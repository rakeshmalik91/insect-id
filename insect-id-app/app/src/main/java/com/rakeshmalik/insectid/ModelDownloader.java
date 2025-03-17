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
import java.util.Objects;
import java.util.concurrent.TimeUnit;

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

    public boolean isModelDownloadOrUpdateRequired(ModelType modelType) {
        return !isModelAlreadyDownloaded(modelType) || isModelToBeUpdated(modelType);
    }

    private boolean isModelAlreadyDownloaded(ModelType modelType) {
        String classesFileName = String.format(CLASSES_FILE_NAME_FMT, modelType.modelName);
        String classDetailsFileName = String.format(CLASS_DETAILS_FILE_NAME_FMT, modelType.modelName);
        String imagesFileName = String.format(IMAGES_FILE_NAME_FMT, modelType.modelName);
        String modelFileName = String.format(MODEL_FILE_NAME_FMT, modelType.modelName);
        return isFileAlreadyDownloaded(classesFileName)
                && isFileAlreadyDownloaded(classDetailsFileName)
                && isFileAlreadyDownloaded(imagesFileName)
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
            downloadFile(modelFileName, fileUrl, onSuccess, onFailure, "Root Classifier model", ROOT_CLASSIFIER, forceUpdate, 1, 5, true);
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
            final String imagesFileName = String.format(IMAGES_FILE_NAME_FMT, modelType.modelName);

            final String classesFileUrl = metadataManager.getMetadata(modelType).optString(FIELD_CLASSES_URL, null);
            final String classDetailsFileUrl = metadataManager.getMetadata(modelType).optString(FIELD_CLASS_DETAILS_URL, null);
            final String imagesFileUrl = metadataManager.getMetadata(modelType).optString(FIELD_IMAGES_URL, null);
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
                            mainHandler.post(() -> outputText.setText(getModelUpToDateMessage(modelType, currentVersion)));
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
                        // download image archive
                        downloadFile(imagesFileName, imagesFileUrl, () -> {
                            // download model
                            downloadFile(modelFileName, modelFileUrl, onSuccess,
                                    onFailure, modelType.displayName + " model", modelType.modelName, updateRequired, 5, 5, true);
                        }, onFailure, "image archives", modelType.modelName, updateRequired, 4, 5, false);
                    }, onFailure, "class details", modelType.modelName, updateRequired, 3, 5, false);
                }, onFailure, "class list", modelType.modelName, updateRequired, 2, 5, false);
            }, onFailure, forceUpdate);
        } catch(Exception ex) {
            if(onFailure != null) {
                onFailure.run();
            }
        }
    }

    private void downloadFile(String fileName, String fileUrl, Runnable onSuccess, Runnable onFailure,
                              String fileType, String modelName, boolean updateRequired,
                              int downloadSeq, int totalDownloads, boolean updatePrefs) {
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
                        final long finalDownloadedBytes = downloadedBytes;
                        mainHandler.post(() -> outputText.setText(getDownloadInProgressMessage(eta, progress, finalDownloadedBytes, totalBytes)));
                    }
                    Log.d(LOG_TAG, "File downloaded successfully: " + file.getAbsolutePath());
                    mainHandler.post(() -> outputText.setText(getDownloadCompletedMessage()));
                    if(onSuccess != null) {
                        onSuccess.run();
                    }
                    if(updatePrefs) {
                        updatePrefs();
                    }
                } catch (Exception e) {
                    Log.e(LOG_TAG, "Download " + fileType + " failed: ", e);
                    if(Objects.equals(e.getMessage(), "Software caused connection abort")) {
                        mainHandler.post(() -> outputText.setText("Download " + fileType + " failed!\nPlease restart the download and do not minimize or close the app or lock the screen."));
                    } else {
                        mainHandler.post(() -> outputText.setText("Download " + fileType + " failed!"));
                    }
                    if(onFailure != null) {
                        onFailure.run();
                    }
                }
            }

            private String getDownloadCompletedMessage() {
                return String.format("Downloaded %s successfully\nDownloads: %d/%d", fileType, downloadSeq, totalDownloads);
            }

            private String getDownloadInProgressMessage(long eta, int progress, long downloadedBytes, long totalBytes) {
                String msg;
                if(updateRequired) {
                    int latestVersion = metadataManager.getMetadata(modelName).optInt(FIELD_VERSION, 0);
                    int currentVersion = prefs.getInt(modelVersionPrefName(modelName), 0);
                    msg = String.format("Updating %s...\n%d min %d sec remaining\n%d%% (%d/%d MB)\nVersion: %d -> %d\nDownloads: %d/%d",
                            fileType, eta / 60000, (eta % 60000) / 1000, progress, downloadedBytes / 1024 / 1024, totalBytes / 1024 / 1024,
                            currentVersion, latestVersion, downloadSeq, totalDownloads);
                } else {
                    msg = String.format("Downloading %s...\n%d min %d sec remaining\n%d%% (%d/%d MB)\nDownloads: %d/%d",
                            fileType, eta / 60000, (eta % 60000) / 1000, progress, downloadedBytes / 1024 / 1024, totalBytes / 1024 / 1024,
                            downloadSeq, totalDownloads);
                }
                return msg;
            }

            private void updatePrefs() {
                prefs.edit().putBoolean(fileDownloadedPrefName(fileName), true).apply();
                if(fileType.toLowerCase().contains("model")) {
                    int version = metadataManager.getMetadata(modelName).optInt(FIELD_VERSION, 0);
                    prefs.edit().putInt(modelVersionPrefName(modelName), version).apply();
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

    private String getModelUpToDateMessage(ModelType modelType, int currentVersion) {
        long speciesCount = metadataManager.getMetadata(modelType).optJSONObject(FIELD_STATS).optLong("species_count", 0);
        long dataCount = metadataManager.getMetadata(modelType).optJSONObject(FIELD_STATS).optLong("data_count", 0);
        return String.format("Model already up to date\nModel name: %s\nVersion: %d\n\nModel info:\nSpecies count: %d\nData count: %d",
                modelType.displayName, currentVersion, speciesCount, dataCount);
    }

}
