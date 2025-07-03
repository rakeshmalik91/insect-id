package com.rakeshmalik.insectid.prediction;

import static com.rakeshmalik.insectid.constants.Constants.*;
import static com.rakeshmalik.insectid.utils.CommonUtils.getGenus;
import static com.rakeshmalik.insectid.utils.CommonUtils.getImagoClassName;
import static com.rakeshmalik.insectid.utils.CommonUtils.isDerivedClass;
import static com.rakeshmalik.insectid.utils.CommonUtils.isEarlyStage;
import static com.rakeshmalik.insectid.utils.CommonUtils.isPossibleDuplicate;

import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;
import android.util.Log;

import androidx.annotation.NonNull;

import com.rakeshmalik.insectid.MainActivity;
import com.rakeshmalik.insectid.enums.Operation;
import com.rakeshmalik.insectid.filemanager.MetadataManager;
import com.rakeshmalik.insectid.filemanager.ModelLoader;
import com.rakeshmalik.insectid.enums.ModelType;
import com.rakeshmalik.insectid.utils.CommonUtils;
import com.rakeshmalik.insectid.utils.ImageUtils;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class PredictionManager {

    private final MainActivity context;
    private final MetadataManager metadataManager;
    private final ModelLoader modelLoader;

    public PredictionManager(Context context, MetadataManager metadataManager, ModelLoader modelLoader) {
        this.context = (MainActivity) context;
        this.metadataManager = metadataManager;
        this.modelLoader = modelLoader;
    }

    private List<String> predictRootClasses(ModelType modelType, Tensor inputTensor) {
        // run through root classifier model
        String modelName = String.format(MODEL_FILE_NAME_FMT, ROOT_CLASSIFIER);
        String modelPath = modelLoader.loadFromCache(context, modelName);
        Log.d(LOG_TAG, modelPath);
        Module model = Module.load(modelPath);
        Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
        float[] logitScores = outputTensor.getDataAsFloatArray();
        float[] softMaxScores = CommonUtils.toSoftMax(logitScores.clone());
        List<String> classLabels = CommonUtils.toList(metadataManager.getMetadata().optJSONObject(ROOT_CLASSIFIER).optJSONArray(FIELD_CLASSES));
        double minAcceptedSoftmax = metadataManager.getMetadata().optJSONObject(ROOT_CLASSIFIER).optDouble(FIELD_MIN_ACCEPTED_SOFTMAX);

        // sort by top score and filter using min accepted softmax
        Integer[] predictedClassIdx = CommonUtils.getSortedIndices(softMaxScores);
        List<String> predictedClass = Arrays.stream(predictedClassIdx)
                .filter(i -> softMaxScores[i] >= minAcceptedSoftmax)
                .map(classLabels::get).collect(Collectors.toList());
        Log.d(LOG_TAG, String.format("Root classifier prediction: %s\n    class labels: %s\n    logits: %s\n    softmax: %s\n    min accepted softmax: %f",
                predictedClass, classLabels, Arrays.toString(logitScores), Arrays.toString(softMaxScores), minAcceptedSoftmax));
        return predictedClass;
    }

    public String predict(ModelType modelType, Uri photoUri) {
        Log.d(LOG_TAG, "inside PredictionManager.predict(" + modelType + ", " + photoUri + ")");

        String modelFileName = String.format(MODEL_FILE_NAME_FMT, modelType.modelName);
        String classListName = String.format(CLASSES_FILE_NAME_FMT, modelType.modelName);
        String classDetailsName = String.format(CLASS_DETAILS_FILE_NAME_FMT, modelType.modelName);

        try {
            String modelPath = modelLoader.loadFromCache(context, modelFileName);
            Module model;
            try {
                model = Module.load(modelPath);
            } catch (Throwable ex) {
                Log.e(LOG_TAG, "Exception loading model", ex);
                throw ex;
            }
            Log.d(LOG_TAG, "Model loaded successfully from " + modelPath);
            List<String> classLabels = modelLoader.getClassLabels(context, classListName);
            Map<String, Map<String, Object>> classDetails = modelLoader.getClassDetails(context, classDetailsName);

            Log.d(LOG_TAG, "Loading photo: " + photoUri);
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(context.getContentResolver(), photoUri);

            // preprocess image
            bitmap = ImageUtils.removeBlackBorders(bitmap, 10, Operation.MEDIAN);
            if(ImageUtils.isScreenCapture(bitmap, 0.25)) {
                bitmap = ImageUtils.applyGaussianBlur(bitmap, 0.01);
            }
            previewPreprocessedImage(bitmap);

            // convert to tensor
            bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
            Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                    new float[]{0.485f, 0.456f, 0.406f},
                    new float[]{0.229f, 0.224f, 0.225f});

            // run through root classifier model
            List<String> predictedRootClasses = predictRootClasses(modelType, inputTensor);
            Set<String> acceptedRootClasses = new HashSet<>(CommonUtils.toList(metadataManager.getMetadata(modelType).optJSONArray(FIELD_ACCEPTED_CLASSES)));

            // run through selected model
            Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
            float[] logitScores = outputTensor.getDataAsFloatArray();
            Log.d(LOG_TAG, "scores: " + Arrays.toString(logitScores));
            float[] softMaxScores = CommonUtils.toSoftMax(logitScores.clone());
            Log.d(LOG_TAG, "softMaxScores: " + Arrays.toString(softMaxScores));

            // get top 10 predictions and filter using min accepted softmax and logit
            int k = MAX_PREDICTIONS;
            Integer[] predictedClassIndices = CommonUtils.getTopKIndices(softMaxScores, k);
            Log.d(LOG_TAG, "Top " + k + " scores: " + Arrays.stream(predictedClassIndices).map(c -> logitScores[c]).collect(Collectors.toList()));
            Log.d(LOG_TAG, "Top " + k + " softMaxScores: " + Arrays.stream(predictedClassIndices).map(c -> softMaxScores[c]).collect(Collectors.toList()));
            final double minAcceptedSoftmax = metadataManager.getMetadata(modelType).optDouble(FIELD_MIN_ACCEPTED_SOFTMAX);
            final double minAcceptedLogit = metadataManager.getMetadata(modelType).optDouble(FIELD_MIN_ACCEPTED_LOGIT);
            Log.d(LOG_TAG, "minAcceptedSoftmax: " + minAcceptedSoftmax);
            Log.d(LOG_TAG, "minAcceptedLogit: " + minAcceptedLogit);
            List<String> predictions = Arrays.stream(predictedClassIndices).map(classLabels::get).collect(Collectors.toList());
            Set<String> predictedGenus = predictions.stream().filter(name -> !isDerivedClass(name)).map(CommonUtils::getGenus).collect(Collectors.toSet());
            List<Integer> filteredPredictionsIndex = Arrays.stream(predictedClassIndices)
                    .filter(classIndex -> softMaxScores[classIndex] > minAcceptedSoftmax)
                    .filter(classIndex -> logitScores[classIndex] > minAcceptedLogit)
                    .filter(classIndex -> isUniquePrediction(classIndex, classLabels, predictions, predictedGenus))
                    .collect(Collectors.toList());
            filteredPredictionsIndex = filterPossibleDuplicateSpeciesNames(filteredPredictionsIndex, classLabels, softMaxScores);
            List<String> filteredPredictionsHtml = filteredPredictionsIndex.stream()
                    .peek(classIndex -> Log.d(LOG_TAG, String.format("Predicted class index: %d, class: %s, logit: %f, softMax: %f", classIndex, classLabels.get(classIndex), logitScores[classIndex], softMaxScores[classIndex])))
                    .map(classIndex -> getPredictionHtml(modelType, classIndex, classLabels, classDetails, softMaxScores))
                    .collect(Collectors.toList());

            // if model is confident enough then override root classifier
            double minAcceptedSoftmaxToOverrideRootClassifier = metadataManager.getMetadata(modelType).optDouble(FIELD_MIN_ACCEPTED_SOFTMAX_TO_OVERRIDE_ROOT_CLASSIFIER);
            Log.d(LOG_TAG, "minAcceptedSoftmaxToOverrideRootClassifier: " + minAcceptedSoftmaxToOverrideRootClassifier);
            boolean confident = softMaxScores[predictedClassIndices[0]] > minAcceptedSoftmaxToOverrideRootClassifier;

            // decide result based on root classifier and model predictions
            if(!confident && predictedRootClasses.stream().noneMatch(acceptedRootClasses::contains)) {
                if(predictedRootClasses.size() == 1 && predictedRootClasses.contains(ROOT_CLASS_OTHER)) {
                    return "No match found!<br><font color='#777777'>Possibly not an Insect<br>Crop to fit the insect for better results</font>";
                } else {
                    return "No match found!<br><font color='#777777'>Possibly not a " + modelType.displayName + "<br>Crop to fit the insect for better results</font>";
                }
            } else if(filteredPredictionsHtml.isEmpty()) {
                return "No match found!<br><font color='#777777'>Crop to fit the insect for better results</font>";
            } else {
                return String.join("<br/><br/>", filteredPredictionsHtml);
            }
        } catch(Exception ex) {
            Log.e(LOG_TAG, "Exception during prediction", ex);
        }
        return "Failed to predict!!!";
    }

    @NonNull
    private static List<Integer> filterPossibleDuplicateSpeciesNames(List<Integer> filteredPredictionsIndex, List<String> classLabels, float[] softMaxScores) {
        // ignore possible duplicate species prediction (eg nepita/asura conferta) and add scores to the highest match
        List<Integer> filteredPredictionsIndexNew = new ArrayList<>();
        for(int i1 = 0; i1 < filteredPredictionsIndex.size(); i1++) {
            int classIndex1 = filteredPredictionsIndex.get(i1);
            if(softMaxScores[classIndex1] > 0) {
                for (int i2 = 0; i2 < filteredPredictionsIndex.size(); i2++) {
                    int classIndex2 = filteredPredictionsIndex.get(i2);
                    if (isPossibleDuplicate(classLabels.get(classIndex1), classLabels.get(classIndex2)) && softMaxScores[classIndex1] > softMaxScores[classIndex2]) {
                        softMaxScores[classIndex1] += softMaxScores[classIndex2];
                        softMaxScores[classIndex2] = 0;
                    }
                }
                filteredPredictionsIndexNew.add(classIndex1);
            }
        }
        return filteredPredictionsIndexNew;
    }

    private static boolean isUniquePrediction(Integer classIndex, List<String> classLabels, List<String> predictions, Set<String> predictedGenus) {
        String name = classLabels.get(classIndex);
        if(isEarlyStage(name)) {
            // imago stage class already predicted, ignore early stage class
            return !predictions.contains(getImagoClassName(name));
        } else if(isDerivedClass(name)) {
            // species level already predicted, ignore genera/spp class
            return !predictedGenus.contains(getGenus(name));
        }
        return true;
    }

    @NonNull
    private String getPredictionHtml(ModelType modelType, Integer classIndex, List<String> classLabels, Map<String, Map<String, Object>> classDetails, float[] softMaxScores) {
        return getScientificNameHtml(classLabels.get(classIndex))
                + getSpeciesNameHtml(classLabels.get(classIndex), classDetails)
                + getScoreHtml(softMaxScores[classIndex])
                + getSpeciesImageList(modelType.modelName, classLabels.get(classIndex));
    }

    private String getScientificName(String className) {
        return className.replaceAll(EARLY_STAGE_CLASS_SUFFIX + "$", "").replaceFirst("-", " ");
    }

    private String getScientificNameHtml(String className) {
        String sciName = getScientificName(className);
        return "<font color='#FF7755'><i>" + sciName + "</i></font><br>";
    }

    private String getSpeciesNameHtml(String className, Map<String, Map<String, Object>> classDetails) {
        String speciesName = "";
        try {
            boolean isEarlyStage = className.endsWith(EARLY_STAGE_CLASS_SUFFIX);
            className = className.replaceAll(EARLY_STAGE_CLASS_SUFFIX + "$", "");
            if (classDetails.containsKey(className) && classDetails.get(className).containsKey(NAME)) {
                // get species name name if available in class details
                speciesName = (String) classDetails.get(className).get(NAME);
            } else {
                // generate species name using scientific name
                String sciName = getScientificName(className);
                speciesName = (sciName.substring(0, 1).toUpperCase() + sciName.substring(1))
                        .replaceAll("(?i)-genera", " Genera")
                        .replaceAll("(?i)-spp$", " spp.");
            }
            if(isEarlyStage) {
                speciesName += EARLY_STAGE_DISPLAY_SUFFIX;
            }
        } catch(Exception ex) {
            Log.e(LOG_TAG, "Exception during species name extraction", ex);
            speciesName = className;
        }
        return "<font color='#FFFFFF'>" + speciesName + "</font><br>";
    }

    private String getScoreHtml(Float score) {
        return String.format(Locale.getDefault(), "<font color='#777777'>~%.2f%% match</font><br><br>", score * 100);
    }

    private String getSpeciesImageList(String modelName, String className) {
        return "<img src='" + modelName + "/" + className + "'/>";
    }

    private void previewPreprocessedImage(final Bitmap bitmap) {
        if (CommonUtils.isDebugMode(context)) {
            context.runOnUiThread(() -> context.getImageView().setImageBitmap(bitmap));
        }
    }

}
