package com.rakeshmalik.insectid;

import static com.rakeshmalik.insectid.Constants.*;

import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class PredictionManager {

    private final Context context;
    private final MetadataManager metadataManager;
    private final ModelLoader modelLoader;

    public PredictionManager(Context context, MetadataManager metadataManager) {
        this.context = context;
        this.metadataManager = metadataManager;
        this.modelLoader = new ModelLoader(context);
    }

    private List<String> predictRootClasses(ModelType modelType, Tensor inputTensor) {
        // run through root classifier model
        String modelName = String.format(MODEL_FILE_NAME_FMT, ROOT_CLASSIFIER);
        String modelPath = modelLoader.loadFromCache(context, modelName);
        Log.d(LOG_TAG, modelPath);
        Module model = Module.load(modelPath);
        Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
        float[] logitScores = outputTensor.getDataAsFloatArray();
        float[] softMaxScores = Utils.toSoftMax(logitScores.clone());
        List<String> classLabels = Utils.toList(metadataManager.getMetadata().optJSONObject(ROOT_CLASSIFIER).optJSONArray(FIELD_CLASSES));
        double minAcceptedSoftmax = metadataManager.getMetadata().optJSONObject(ROOT_CLASSIFIER).optDouble(FIELD_MIN_ACCEPTED_SOFTMAX);

        // sort by top score and filter using min accepted softmax
        Integer[] predictedClassIdx = Utils.getSortedIndices(softMaxScores);
        List<String> predictedClass = Arrays.stream(predictedClassIdx)
                .filter(i -> softMaxScores[i] >= minAcceptedSoftmax)
                .map(classLabels::get).collect(Collectors.toList());
        Log.d(LOG_TAG, String.format("Root classifier prediction: %s\n    class labels: %s\n    logits: %s\n    softmax: %s\n    min accepted softmax: %f",
                predictedClass, classLabels, Arrays.toString(logitScores), Arrays.toString(softMaxScores), minAcceptedSoftmax));
        return predictedClass;
    }

    public String predict(ModelType modelType, Uri photoUri) {
        String modelName = String.format(MODEL_FILE_NAME_FMT, modelType.modelName);
        String classListName = String.format(CLASSES_FILE_NAME_FMT, modelType.modelName);
        String classDetailsName = String.format(CLASS_DETAILS_FILE_NAME_FMT, modelType.modelName);

        try {
            String modelPath = modelLoader.loadFromCache(context, modelName);
            Module model = Module.load(modelPath);
            List<String> classLabels = modelLoader.getClassLabels(context, classListName);
            Map<String, Map<String, Object>> classDetails = modelLoader.getClassDetails(context, classDetailsName);

            Log.d(LOG_TAG, "Loading photo: " + photoUri);
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(context.getContentResolver(), photoUri);
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
            Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap,
                    new float[]{0.485f, 0.456f, 0.406f},
                    new float[]{0.229f, 0.224f, 0.225f});

            // run through root classifier model
            List<String> predictedRootClasses = predictRootClasses(modelType, inputTensor);
            Set<String> acceptedRootClasses = new HashSet<>(Utils.toList(metadataManager.getMetadata(modelType).optJSONArray(FIELD_ACCEPTED_CLASSES)));

            // run through selected model
            Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
            float[] logitScores = outputTensor.getDataAsFloatArray();
            Log.d(LOG_TAG, "scores: " + Arrays.toString(logitScores));
            float[] softMaxScores = Utils.toSoftMax(logitScores.clone());
            Log.d(LOG_TAG, "softMaxScores: " + Arrays.toString(softMaxScores));

            // get top 10 predictions and filter using min accepted softmax and logit
            int k = MAX_PREDICTIONS;
            Integer[] predictedClass = Utils.getTopKIndices(softMaxScores, k);
            Log.d(LOG_TAG, "Top " + k + " scores: " + Arrays.stream(predictedClass).map(c -> logitScores[c]).collect(Collectors.toList()));
            Log.d(LOG_TAG, "Top " + k + " softMaxScores: " + Arrays.stream(predictedClass).map(c -> softMaxScores[c]).collect(Collectors.toList()));
            final double minAcceptedSoftmax = metadataManager.getMetadata(modelType).optDouble(FIELD_MIN_ACCEPTED_SOFTMAX);
            final double minAcceptedLogit = metadataManager.getMetadata(modelType).optDouble(FIELD_MIN_ACCEPTED_LOGIT);
            Log.d(LOG_TAG, "minAcceptedSoftmax: " + minAcceptedSoftmax);
            Log.d(LOG_TAG, "minAcceptedLogit: " + minAcceptedLogit);
            List<String> predictions = Arrays.stream(predictedClass)
                    .filter(c -> softMaxScores[c] > minAcceptedSoftmax)
                    .filter(c -> logitScores[c] > minAcceptedLogit)
                    .map(c -> getScientificNameHtml(classLabels.get(c))
                            + getSpeciesNameHtml(classLabels.get(c), classDetails)
                            + getScoreHtml(softMaxScores[c])
                            + getSpeciesImageList(classLabels.get(c), classDetails))
                    .collect(Collectors.toList());
            Log.d(LOG_TAG, "Predicted class: " + predictions);

            // if model is confident enough then override root classifier
            double minAcceptedSoftmaxToOverrideRootClassifier = metadataManager.getMetadata(modelType).optDouble(FIELD_MIN_ACCEPTED_SOFTMAX_TO_OVERRIDE_ROOT_CLASSIFIER);
            Log.d(LOG_TAG, "minAcceptedSoftmaxToOverrideRootClassifier: " + minAcceptedSoftmaxToOverrideRootClassifier);
            boolean confident = softMaxScores[predictedClass[0]] > minAcceptedSoftmaxToOverrideRootClassifier;

            // decide result based on root classifier and model predictions
            if(!confident && predictedRootClasses.stream().noneMatch(acceptedRootClasses::contains)) {
                if(predictedRootClasses.size() == 1 && predictedRootClasses.contains(ROOT_CLASS_OTHER)) {
                    return "No match found!<br><font color='#777777'>Possibly not an Insect<br>Crop to fit the insect for better results</font>";
                } else {
                    return "No match found!<br><font color='#777777'>Possibly not a " + modelType.displayName + "<br>Crop to fit the insect for better results</font>";
                }
            } else if(predictions.isEmpty()) {
                return "No match found!<br><font color='#777777'>Crop to fit the insect for better results</font>";
            } else {
                return String.join("<br/><br/>", predictions);
            }
        } catch(Exception ex) {
            Log.e(LOG_TAG, "Exception during prediction", ex);
        }
        return "Failed to predict!!!";
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
                speciesName = Arrays.stream(sciName.split(" "))
                        .map(s -> s.substring(0, 1).toUpperCase() + s.substring(1))
                        .collect(Collectors.joining(" "))
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

    private String getSpeciesImageList(String className, Map<String, Map<String, Object>> classDetails) {
        try {
            if (classDetails.containsKey(className) && classDetails.get(className).containsKey(IMAGES)
                    && !((List<String>)classDetails.get(className).get(IMAGES)).isEmpty()) {
                String urls = ((List<String>)classDetails.get(className).get(IMAGES)).stream()
                        .limit(MAX_IMAGES_IN_PREDICTION)
                        .collect(Collectors.joining(","));
                return "<img src='" + urls + "'/>";
            }
        } catch(Exception ex) {
            Log.e(LOG_TAG, "Exception fetching species image urls", ex);
        }
        return HTML_NO_IMAGE_AVAILABLE;
    }

}
