package com.rakeshmalik.insectid.filemanager;

import static com.rakeshmalik.insectid.constants.Constants.*;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.TextView;

import com.rakeshmalik.insectid.pojo.InsectModel;

import java.io.File;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.List;

import okhttp3.*;

//TODO have cancel/retry/resume download button
//TODO have auto-update, pre-download & verify-integrity/re-download model settings
public class ModelDownloader {

    private final OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(MODEL_LOAD_TIMEOUT, TimeUnit.MILLISECONDS)
            .readTimeout(MODEL_LOAD_TIMEOUT, TimeUnit.MILLISECONDS)
            .retryOnConnectionFailure(true)
            .addInterceptor(chain -> {
                Request request = chain.request().newBuilder()
                        .header("Connection", "keep-alive")
                        .header("Keep-Alive", "timeout=60, max=100")
                        .build();
                return chain.proceed(request);
            })
            .build();
    private final Context context;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final SharedPreferences prefs;
    private final TextView outputText;
    private final MetadataManager metadataManager;

    public ModelDownloader(Context context, TextView outputText, MetadataManager metadataManager) {
        this.context = context;
        this.outputText = outputText;
        this.prefs = context.getSharedPreferences(PREF, Context.MODE_PRIVATE);
        this.metadataManager = metadataManager;
    }

    public boolean isModelDownloadOrUpdateRequired(InsectModel model) {
        return !isModelAlreadyDownloaded(model) || isModelToBeUpdated(model);
    }

    public boolean isModelAlreadyDownloaded(InsectModel model) {
        String classesFileName = String.format(CLASSES_FILE_NAME_FMT, model.getModelName());
        String classDetailsFileName = String.format(CLASS_DETAILS_FILE_NAME_FMT, model.getModelName());
        String imagesArchiveFileName = String.format(IMAGES_ARCHIVE_FILE_NAME_FMT, model.getModelName());
        String modelFileName = String.format(MODEL_FILE_NAME_FMT, model.getModelName());
        return isFileAlreadyDownloaded(classesFileName)
                && isFileAlreadyDownloaded(classDetailsFileName)
                && isFileAlreadyDownloaded(imagesArchiveFileName)
                && isFileAlreadyDownloaded(modelFileName);
    }

    public boolean isModelToBeUpdated(InsectModel model) {
        int currentVersion = prefs.getInt(modelVersionPrefName(model.getModelName()), 0);
        int latestVersion = metadataManager.getMetadata(model.getModelName()).optInt(FIELD_VERSION, 0);
        Log.d(LOG_TAG, String.format("Model type: %s, current version: %d, latest version: %d", model.getModelName(), currentVersion, latestVersion));
        return currentVersion < latestVersion;
    }

    public void downloadModel(InsectModel model, Runnable onSuccess, Runnable onFailure, int modelDownloadSeq, int totalModelDownloads) {
        downloadModel(model, onSuccess, onFailure, false, modelDownloadSeq, totalModelDownloads);
    }

    private boolean isRootClassifierDownloadRequired() {
        String modelFileName = String.format(MODEL_FILE_NAME_FMT, ROOT_CLASSIFIER);
        int currentVersion = prefs.getInt(modelVersionPrefName(ROOT_CLASSIFIER), 0);
        int latestVersion = metadataManager.getMetadata(ROOT_CLASSIFIER).optInt(FIELD_VERSION, 0);
        Log.d(LOG_TAG, String.format("Model type: %s, current version: %d, latest version: %d", ROOT_CLASSIFIER, currentVersion, latestVersion));
        return !isFileAlreadyDownloaded(modelFileName) || currentVersion < latestVersion;
    }

    private void downloadRootClassifier(Runnable onSuccess, Runnable onFailure, boolean forceUpdate, int modelDownloadSeq, int totalModelDownloads) {
        Log.d(LOG_TAG, "Inside ModelDownloader.downloadRootClassifier()");
        String modelFileName = String.format(MODEL_FILE_NAME_FMT, ROOT_CLASSIFIER);
        if(isRootClassifierDownloadRequired()) {
            String fileUrl = metadataManager.getMetadata(ROOT_CLASSIFIER).optString(FIELD_MODEL_URL, null);
            downloadFile(modelFileName, fileUrl, onSuccess, onFailure, "Root Classifier model", ROOT_CLASSIFIER, forceUpdate,
                    1, 5, modelDownloadSeq, totalModelDownloads);
        } else {
            Log.d(LOG_TAG, "Model " + ROOT_CLASSIFIER + " already downloaded.");
            onSuccess.run();
        }
    }

    public void downloadModel(InsectModel model, Runnable onSuccess, Runnable onFailure, boolean forceUpdate, int modelDownloadSeq, int totalModelDownloads) {
        Log.d(LOG_TAG, "Inside ModelDownloader.downloadModel()");
        try {

            if(forceUpdate) {
                metadataManager.getMetadata(true);
            }

            final String classesFileName = String.format(CLASSES_FILE_NAME_FMT, model.getModelName());
            final String classDetailsFileName = String.format(CLASS_DETAILS_FILE_NAME_FMT, model.getModelName());
            final String modelFileName = String.format(MODEL_FILE_NAME_FMT, model.getModelName());
            final String imagesFileName = String.format(IMAGES_ARCHIVE_FILE_NAME_FMT, model.getModelName());

            final String classesFileUrl = metadataManager.getMetadata(model.getModelName()).optString(FIELD_CLASSES_URL, null);
            final String classDetailsFileUrl = metadataManager.getMetadata(model.getModelName()).optString(FIELD_CLASS_DETAILS_URL, null);
            final String imagesFileUrl = metadataManager.getMetadata(model.getModelName()).optString(FIELD_IMAGES_URL, null);
            final String modelFileUrl = metadataManager.getMetadata(model.getModelName()).optString(FIELD_MODEL_URL, null);

            Runnable downloadModelData = () -> {
                final boolean updateRequired;
                if (isModelAlreadyDownloaded(model)) {
                    updateRequired = forceUpdate && isModelToBeUpdated(model);
                    if(updateRequired) {
                        Log.d(LOG_TAG, "Going to update " + model.getModelName() + " model");
                    } else {
                        if(forceUpdate) {
                            int currentVersion = prefs.getInt(modelVersionPrefName(model.getModelName()), 0);
                            mainHandler.post(() -> outputText.setText(getModelUpToDateMessage(model, currentVersion)));
                        }
                        Log.d(LOG_TAG, "Model " + model.getModelName() + " already downloaded.");
                        onSuccess.run();
                        return;
                    }
                } else {
                    updateRequired = false;
                    Log.d(LOG_TAG, "Going to download " + model.getModelName() + " model");
                }

                Runnable downloadModelFile = () -> downloadFile(modelFileName, modelFileUrl, onSuccess, onFailure,
                        model.getDisplayName() + " model", model.getModelName(), updateRequired,
                        5, 5, modelDownloadSeq, totalModelDownloads);

                Runnable downloadImageArchive = () -> downloadFile(imagesFileName, imagesFileUrl, downloadModelFile, onFailure,
                        model.getDisplayName() + " images", model.getModelName(), updateRequired,
                        4, 5, modelDownloadSeq, totalModelDownloads);

                Runnable downloadClassDetails = () -> downloadFile(classDetailsFileName, classDetailsFileUrl, downloadImageArchive, onFailure,
                        model.getDisplayName() + " metadata", model.getModelName(), updateRequired,
                        3, 5, modelDownloadSeq, totalModelDownloads);

                downloadFile(classesFileName, classesFileUrl, downloadClassDetails, onFailure,
                        model.getDisplayName() + " classes", model.getModelName(), updateRequired,
                        2, 5, modelDownloadSeq, totalModelDownloads);
            };

            downloadRootClassifier(downloadModelData, onFailure, forceUpdate, modelDownloadSeq, totalModelDownloads);
        } catch(Exception ex) {
            Log.e(LOG_TAG, "Exception downloading model " + model, ex);
            if(onFailure != null) {
                onFailure.run();
            }
        }
    }

    private void downloadFile(String fileName, String fileUrl, Runnable onSuccess, Runnable onFailure,
                              String downloadName, String modelName, boolean updateRequired,
                              int fileDownloadSeq, int totalFileDownloads,
                              int modelDownloadSeq, int totalModelDownloads) {
        try {
            Log.d(LOG_TAG, "Downloading " + downloadName + " " + fileName + " from " + fileUrl + "...");
            DownloadFileHttpCallback callback = new DownloadFileHttpCallback(context, outputText, metadataManager, mainHandler, prefs,
                    fileName, onSuccess, onFailure, downloadName, modelName, updateRequired,
                    fileDownloadSeq, totalFileDownloads, modelDownloadSeq, totalModelDownloads);
            client.newCall(new Request.Builder().url(fileUrl).build()).enqueue(callback);
        } catch(Exception ex) {
            Log.e(LOG_TAG, "Exception downloading " + fileUrl, ex);
            if(onFailure != null) {
                onFailure.run();
            }
        }
    }

    private boolean isFileAlreadyDownloaded(String fileName) {
        File file = new File(context.getFilesDir(), fileName);
        return file.exists() && prefs.getBoolean(fileDownloadedPrefName(fileName), false);
    }

    @SuppressLint("DefaultLocale")
    private String getModelUpToDateMessage(InsectModel model, int currentVersion) {
        long speciesCount = metadataManager.getMetadata(model.getModelName()).optJSONObject(FIELD_STATS).optLong("species_count", 0);
        long dataCount = metadataManager.getMetadata(model.getModelName()).optJSONObject(FIELD_STATS).optLong("data_count", 0);
        String msg = String.format("Model already up to date\nModel name: %s\nVersion: %d\n\nModel info:\nSpecies count: %d\nData count: %dk",
                model.getDisplayName(), currentVersion, speciesCount, dataCount/1000);
        msg += Optional.ofNullable(metadataManager.getMetadata(model.getModelName()).optJSONObject(FIELD_STATS).optString("accuracy", null))
                .map((accuracy) -> String.format("\nOverall accuracy: %s", accuracy)).orElse("");
        msg += Optional.ofNullable(metadataManager.getMetadata(model.getModelName()).optJSONObject(FIELD_STATS).optString("accuracy_top3", null))
                .map((accuracyTop3) -> String.format("\nTop-3 accuracy: %s", accuracyTop3)).orElse("");
        msg += Optional.ofNullable(metadataManager.getMetadata(model.getModelName()).optJSONObject(FIELD_STATS).optString("last_updated_date", null))
                .map((lastUpdatedDate) -> String.format("\nLast updated on: %s", lastUpdatedDate)).orElse("");
        return msg;
    }

    public static String fileDownloadedPrefName(String fileName) {
        return PREF_FILE_DOWNLOADED + "::" + fileName;
    }

    public static String modelVersionPrefName(String modelName) {
        return PREF_MODEL_VERSION + "::" + modelName;
    }

    public long getModelDownloadSizeInMB(InsectModel model) {
        long size = 0;
        if(isRootClassifierDownloadRequired()) {
            size += metadataManager.getModelSize(ROOT_CLASSIFIER);
        }
        if(isModelDownloadOrUpdateRequired(model)) {
            size += metadataManager.getModelSize(model.getModelName());
        }
        return size / 1000 / 1000;
    }

    public long getTotalModelDownloadSizeInMB(boolean onlyNonLegacy) {
        long totalSize = 0;
        if(isRootClassifierDownloadRequired()) {
            long size = metadataManager.getModelSize(ROOT_CLASSIFIER);
            if(size <= 0) return 0;
            totalSize += size;
        }
        List<InsectModel> availableModels = metadataManager.getAvailableModels();
        for(InsectModel model: availableModels) {
            if ((!onlyNonLegacy || !model.isLegacy()) && isModelDownloadOrUpdateRequired(model)) {
                long size = metadataManager.getModelSize(model.getModelName());
                if(size <= 0) return 0;
                totalSize += size;
            }
        }
        return totalSize / 1000 / 1000;
    }

}
