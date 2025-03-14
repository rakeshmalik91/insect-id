package com.rakeshmalik.insectid;

import static com.rakeshmalik.insectid.Constants.*;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.TimeUnit;

import okhttp3.*;

//TODO have cancel/retry/resume download button
//TODO have auto-update, pre-download & verify-integrity/re-download model settings
public class ModelDownloader {

    private final OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(MODEL_LOAD_TIMEOUT, TimeUnit.MILLISECONDS)
            .readTimeout(MODEL_LOAD_TIMEOUT, TimeUnit.MILLISECONDS)
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

    public boolean isModelDownloadOrUpdateRequired(ModelType modelType) {
        return !isModelAlreadyDownloaded(modelType) || isModelToBeUpdated(modelType);
    }

    private boolean isModelAlreadyDownloaded(ModelType modelType) {
        String classesFileName = String.format(CLASSES_FILE_NAME_FMT, modelType.modelName);
        String classDetailsFileName = String.format(CLASS_DETAILS_FILE_NAME_FMT, modelType.modelName);
        String modelFileName = String.format(MODEL_FILE_NAME_FMT, modelType.modelName);
        return isFileAlreadyDownloaded(classesFileName)
                && isFileAlreadyDownloaded(classDetailsFileName)
                && isFileAlreadyDownloaded(modelFileName);
    }

    private boolean isModelToBeUpdated(ModelType modelType) {
        int currentVersion = prefs.getInt(modelVersionPrefName(modelType.modelName), 0);
        int latestVersion = metadataManager.getMetadata(modelType).optInt(FIELD_VERSION, 0);
        Log.d(LOG_TAG, String.format("Model type: %s, current version: %d, latest version: %d", modelType.modelName, currentVersion, latestVersion));
        return currentVersion < latestVersion;
    }

    public void downloadModel(ModelType modelType, Runnable onSuccess, Runnable onFailure) {
        downloadModel(modelType, onSuccess, onFailure, false);
    }

    private void downloadRootClassifier(Runnable onSuccess, Runnable onFailure, boolean forceUpdate) {
        String modelFileName = String.format(MODEL_FILE_NAME_FMT, ROOT_CLASSIFIER);
        int currentVersion = prefs.getInt(modelVersionPrefName(ROOT_CLASSIFIER), 0);
        int latestVersion = metadataManager.getMetadata(ROOT_CLASSIFIER).optInt(FIELD_VERSION, 0);
        Log.d(LOG_TAG, String.format("Model type: %s, current version: %d, latest version: %d", ROOT_CLASSIFIER, currentVersion, latestVersion));
        if(isFileAlreadyDownloaded(modelFileName) && currentVersion >= latestVersion) {
            Log.d(LOG_TAG, "Model " + ROOT_CLASSIFIER + " already downloaded.");
            onSuccess.run();
        } else {
            String fileUrl = metadataManager.getMetadata(ROOT_CLASSIFIER).optString(FIELD_MODEL_URL, null);
            downloadFile(modelFileName, fileUrl, onSuccess, onFailure, "model Root Classifier", ROOT_CLASSIFIER, forceUpdate);
        }
    }

    public void downloadModel(ModelType modelType, Runnable onSuccess, Runnable onFailure, boolean forceUpdate) {
        try {
            if(forceUpdate) {
                metadataManager.getMetadata(true);
            }

            final String classesFileName = String.format(CLASSES_FILE_NAME_FMT, modelType.modelName);
            final String classDetailsFileName = String.format(CLASS_DETAILS_FILE_NAME_FMT, modelType.modelName);
            final String modelFileName = String.format(MODEL_FILE_NAME_FMT, modelType.modelName);

            final String classesFileUrl = metadataManager.getMetadata(modelType).optString(FIELD_CLASSES_URL, null);
            final String classDetailsFileUrl = metadataManager.getMetadata(modelType).optString(FIELD_CLASS_DETAILS_URL, null);
            final String modelFileUrl = metadataManager.getMetadata(modelType).optString(FIELD_MODEL_URL, null);

            // download root classifier
            downloadRootClassifier(() -> {
                final boolean updateRequired;
                if (isModelAlreadyDownloaded(modelType)) {
                    updateRequired = forceUpdate && isModelToBeUpdated(modelType);
                    if(updateRequired) {
                        Log.d(LOG_TAG, "Going to update " + modelType.modelName + " model");
                    } else {
                        if(forceUpdate) {
                            int currentVersion = prefs.getInt(modelVersionPrefName(modelType.modelName), 0);
                            mainHandler.post(() -> outputText.setText("Model already up to date\nModel name: " + modelType.displayName + "\nVersion: " + currentVersion));
                        }
                        Log.d(LOG_TAG, "Model " + modelType.modelName + " already downloaded.");
                        onSuccess.run();
                        return;
                    }
                } else {
                    updateRequired = false;
                    Log.d(LOG_TAG, "Going to download " + modelType.modelName + " model");
                }

                // download class list
                downloadFile(classesFileName, classesFileUrl, () -> {
                    // download class details
                    downloadFile(classDetailsFileName, classDetailsFileUrl, () -> {
                        // download model
                        downloadFile(modelFileName, modelFileUrl, onSuccess, onFailure, "model " + modelType.displayName, modelType.modelName, updateRequired);
                    }, onFailure, "class details", modelType.modelName, updateRequired);
                }, onFailure, "class list", modelType.modelName, updateRequired);
            }, onFailure, forceUpdate);
        } catch(Exception ex) {
            if(onFailure != null) {
                onFailure.run();
            }
        }
    }

    private void downloadFile(String fileName, String fileUrl, Runnable onSuccess, Runnable onFailure, String fileType, String modelName, boolean updateRequired) {
        Log.d(LOG_TAG, "Downloading " + fileType + " " + fileName + " from " + fileUrl + "...");
        client.newCall(new Request.Builder().url(fileUrl).build()).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                Log.e(LOG_TAG, "Download " + fileType + " failed: " + e.getMessage());
                mainHandler.post(() -> outputText.setText("Download " + fileType + " failed!"));
                if(onFailure != null) {
                    onFailure.run();
                }
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (!response.isSuccessful()) {
                    Log.e(LOG_TAG, "Server error: " + response.code());
                    mainHandler.post(() -> outputText.setText("Download " + fileType + " failed!"));
                    return;
                }
                File cacheDir = context.getCacheDir();
                File file = new File(cacheDir, fileName);
                long startTime = System.currentTimeMillis();
                try(InputStream inputStream = response.body().byteStream();
                    FileOutputStream outputStream = new FileOutputStream(file); ) {
                    byte[] buffer = new byte[4096];
                    long totalBytes = response.body().contentLength();
                    long downloadedBytes = 0;
                    int bytesRead;
                    while ((bytesRead = inputStream.read(buffer)) != -1) {
                        outputStream.write(buffer, 0, bytesRead);
                        downloadedBytes += bytesRead;
                        int progress = Math.max(0, (int) ((downloadedBytes * 100) / totalBytes));
                        long elapsedTime = System.currentTimeMillis() - startTime;
                        long eta = Math.max(0, (totalBytes - downloadedBytes) * elapsedTime / downloadedBytes);
                        String msg;
                        if(updateRequired) {
                            int latestVersion = metadataManager.getMetadata(modelName).optInt(FIELD_VERSION, 0);
                            int currentVersion = prefs.getInt(modelVersionPrefName(modelName), 0);
                            msg = String.format("Updating %s...\n%d min %d sec remaining\n%d%% (%d/%d MB)\nCurrent version: %d, Latest version: %d",
                                    fileType, eta / 60000, (eta % 60000) / 1000, progress, downloadedBytes / 1024 / 1024, totalBytes / 1024 / 1024,
                                    currentVersion, latestVersion);
                        } else {
                            msg = String.format("Downloading %s...\n%d min %d sec remaining\n%d%% (%d/%d MB)",
                                    fileType, eta / 60000, (eta % 60000) / 1000, progress, downloadedBytes / 1024 / 1024, totalBytes / 1024 / 1024);
                        }
                        mainHandler.post(() -> outputText.setText(msg));
                    }
                    Log.d(LOG_TAG, "File downloaded successfully: " + file.getAbsolutePath());
                    mainHandler.post(() -> outputText.setText("Downloaded " + fileType + " successfully"));
                    if(onSuccess != null) {
                        onSuccess.run();
                    }
                    prefs.edit().putBoolean(fileDownloadedPrefName(fileName), true).apply();
                    if(fileType.toLowerCase().contains("model")) {
                        int version = metadataManager.getMetadata(modelName).optInt(FIELD_VERSION, 0);
                        prefs.edit().putInt(modelVersionPrefName(modelName), version).apply();
                    }
                }
            }
        });
    }

    private boolean isFileAlreadyDownloaded(String fileName) {
        File file = new File(context.getCacheDir(), fileName);
        return file.exists() && prefs.getBoolean(fileDownloadedPrefName(fileName), false);
    }

    private String fileDownloadedPrefName(String fileName) {
        return PREF_FILE_DOWNLOADED + "::" + fileName;
    }

    private String modelVersionPrefName(String modelName) {
        return PREF_MODEL_VERSION + "::" + modelName;
    }

}
