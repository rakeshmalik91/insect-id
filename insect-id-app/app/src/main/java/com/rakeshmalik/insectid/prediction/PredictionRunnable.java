package com.rakeshmalik.insectid.prediction;

import static com.rakeshmalik.insectid.constants.Constants.HTML_NO_IMAGE_AVAILABLE;
import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;

import android.graphics.drawable.Drawable;
import android.text.Html;
import android.text.Spanned;
import android.util.Log;

import com.rakeshmalik.insectid.MainActivity;
import com.rakeshmalik.insectid.R;
import com.rakeshmalik.insectid.filemanager.ModelDownloader;
import com.rakeshmalik.insectid.filemanager.ModelLoader;

import java.util.List;

public class PredictionRunnable implements Runnable {

    private final PredictionManager predictionManager;
    private final MainActivity mainActivity;
    private final ModelLoader modelLoader;
    private final ModelDownloader modelDownloader;

    public PredictionRunnable(MainActivity mainActivity, PredictionManager predictionManager, ModelLoader modelLoader, ModelDownloader modelDownloader) {
        this.mainActivity = mainActivity;
        this.predictionManager = predictionManager;
        this.modelLoader = modelLoader;
        this.modelDownloader = modelDownloader;
    }

    @Override
    public void run() {
        Log.d(LOG_TAG, "Inside PredictionRunnable.run()");
        try {
            mainActivity.lockUI();
            
            if (modelDownloader.isModelDownloadOrUpdateRequired(mainActivity.getSelectedModel())) {
                List<com.rakeshmalik.insectid.filemanager.DownloadItem> plan = modelDownloader.generateDownloadPlan(
                        java.util.Collections.singletonList(mainActivity.getSelectedModel()), true);
                if (!plan.isEmpty()) {
                    mainActivity.showMessage("Downloading model. Please wait...");
                    mainActivity.initDownloadList(plan);
                    mainActivity.showDownloadProgressContainer();
                    mainActivity.switchToModelsTab();
                }
            } else {
                mainActivity.showMessage(mainActivity.getString(R.string.predicting));
            }
            
            modelDownloader.downloadModel(mainActivity.getSelectedModel(), () -> {
                mainActivity.switchToIdentifyTab();
                runPrediction();
            }, mainActivity::unlockUI, 1, 1);
        } catch(Exception ex) {
            mainActivity.unlockUI();
        }
    }

    private void runPrediction() {
        Log.d(LOG_TAG, "Inside PredictionRunnable.runPrediction()");
        try {
            mainActivity.showMessage(mainActivity.getString(R.string.predicting));
            PredictionResponse response = predictionManager.predict(mainActivity.getSelectedModel(), mainActivity.getPhotoUri());

            if (response.predictions != null) {
                for (PredictionResult result : response.predictions) {
                    try {
                        org.json.JSONObject meta = mainActivity.getMetadataManager().getMetadata(result.modelName);
                        String imagesUrl = com.rakeshmalik.insectid.pojo.InsectModel.fromJson(result.modelName, meta).getImagesUrl();
                        result.images = modelLoader.getImagesFromZip(mainActivity, result.modelName, imagesUrl, result.className);
                    } catch (Exception ex) {
                        Log.e(LOG_TAG, "Failed to load images for " + result.className, ex);
                    }
                }
            }
            mainActivity.showPredictionResponse(response);
        } catch(Exception ex) {
            Log.e(LOG_TAG, "Exception during prediction", ex);
            mainActivity.showMessage("Failed to predict!!!");
        } finally {
            mainActivity.unlockUI();
        }
    }

}