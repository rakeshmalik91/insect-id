package com.rakeshmalik.insectid;

import static com.rakeshmalik.insectid.Constants.*;

import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

public class PredictionManager {

    private final MetadataManager metadataManager;

    private final ModelLoader modelLoader = new ModelLoader();

    public PredictionManager(MetadataManager metadataManager) {
        this.metadataManager = metadataManager;
    }

    public String predict(Context context, ModelType modelType, Uri photoUri) {
        String modelName = String.format(Constants.MODEL_FILE_NAME_FMT, modelType.modelName);
        String classListName = String.format(Constants.CLASSES_FILE_NAME_FMT, modelType.modelName);
        String classDetailsName = String.format(Constants.CLASS_DETAILS_FILE_NAME_FMT, modelType.modelName);

        try {
            String modelPath = modelLoader.load(context, modelName);
            Module model = Module.load(modelPath);
            List<String> classLabels = modelLoader.getClassLabels(context, classListName);
            Map<String, Map<String, String>> classDetails = modelLoader.getClassDetails(context, classDetailsName);

            Log.d(LOG_TAG, "Loading photo: " + photoUri);
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(context.getContentResolver(), photoUri);
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
            Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap,
                    new float[]{0.485f, 0.456f, 0.406f},
                    new float[]{0.229f, 0.224f, 0.225f});

            Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
            float[] logitScores = outputTensor.getDataAsFloatArray();
            Log.d(LOG_TAG, "scores: " + Arrays.toString(logitScores));
            float[] softMaxScores = toSoftMax(logitScores.clone());
            Log.d(LOG_TAG, "softMaxScores: " + Arrays.toString(softMaxScores));

            int k = MAX_PREDICTIONS;
            Integer[] predictedClass = getTopKIndices(softMaxScores, k);
            Log.d(LOG_TAG, "Top " + k + " scores: " + Arrays.stream(predictedClass).map(c -> logitScores[c]).collect(Collectors.toList()));
            Log.d(LOG_TAG, "Top " + k + " softMaxScores: " + Arrays.stream(predictedClass).map(c -> softMaxScores[c]).collect(Collectors.toList()));

            final double minAcceptedSoftmax = metadataManager.getMetadata(modelType).optDouble(FIELD_MIN_ACCEPTED_SOFTMAX);
            final double minAcceptedLogit = metadataManager.getMetadata(modelType).optDouble(FIELD_MIN_ACCEPTED_LOGIT);
            List<String> predictions = Arrays.stream(predictedClass)
                    .filter(c -> softMaxScores[c] > minAcceptedSoftmax)
                    .filter(c -> logitScores[c] > minAcceptedLogit)
                    .map(c -> getScientificNameHtml(classLabels.get(c))
                            + getSpeciesNameHtml(classLabels.get(c), classDetails)
                            + getScoreHtml(softMaxScores[c]))
                    .collect(Collectors.toList());
            Log.d(LOG_TAG, "Predicted class: " + predictions);
            return predictions.isEmpty() ? context.getString(R.string.no_match_found) : String.join("\n", predictions);
        } catch(Exception ex) {
            Log.e(LOG_TAG, "Exception during prediction", ex);
        }
        return "Failed to predict!!!";
    }

    private static String getScientificNameHtml(String className) {
        className = className.replaceAll("-early$", "");
        return "<font color='#FF7755'><i>" + className + "</i></font><br>";
    }

    private static String getSpeciesNameHtml(String className, Map<String, Map<String, String>> classDetails) {
        String speciesName = "";
        try {
            for (String suffix : CLASS_SUFFIXES.keySet()) {
                if (className.endsWith(suffix)) {
                    String imagoClassName = className.substring(0, className.length() - suffix.length());
                    if (classDetails.containsKey(imagoClassName) && classDetails.get(imagoClassName).containsKey(NAME)) {
                        speciesName = classDetails.get(imagoClassName).get(NAME) + CLASS_SUFFIXES.get(suffix);
                    }
                }
            }
            if (speciesName.isBlank() && classDetails.containsKey(className) && classDetails.get(className).containsKey(NAME)) {
                speciesName = classDetails.get(className).get(NAME);
            }
            if (speciesName.isBlank()) {
                speciesName = Arrays.stream(className.split("-"))
                        .map(s -> s.substring(0, 1).toUpperCase() + s.substring(1))
                        .collect(Collectors.joining(" "))
                        .replaceAll("(?i) spp$", " spp.");
                for (String suffix : CLASS_SUFFIXES.keySet()) {
                    speciesName = speciesName.replaceAll("(?i) " + suffix + "$", CLASS_SUFFIXES.get(suffix));
                }
            }
        } catch(Exception ex) {
            Log.e(LOG_TAG, "Exception during species name extraction", ex);
            speciesName = className;
        }
        return "<font color='#FFFFFF'>" + speciesName + "</font><br>";
    }

    private static String getScoreHtml(Float score) {
        return String.format(Locale.getDefault(), "<font color='#777777'>~%.2f%% match</font><br><br>", score * 100);
    }

    private static float[] toSoftMax(float[] scores) {
        float sumExp = 0.0f;
        for (float score : scores) {
            sumExp += (float) Math.exp(score);
        }
        for (int i = 0; i < scores.length; i++) {
            scores[i] = (float) Math.exp(scores[i]) / sumExp;
        }
        return scores;
    }

    public static Integer[] getTopKIndices(float[] array, int k) {
        Integer[] indices = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, (i1, i2) -> Double.compare(array[i2], array[i1]));
        return Arrays.copyOfRange(indices, 0, k);
    }

}
