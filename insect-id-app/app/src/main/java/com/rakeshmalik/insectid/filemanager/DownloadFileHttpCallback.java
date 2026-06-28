package com.rakeshmalik.insectid.filemanager;

import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;
import static com.rakeshmalik.insectid.constants.Constants.WAKE_LOCK_NAME;
import static com.rakeshmalik.insectid.constants.Constants.WAKE_LOCK_TIME;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Handler;
import android.os.PowerManager;
import android.util.Log;

import androidx.annotation.NonNull;

import com.rakeshmalik.insectid.R;
import com.rakeshmalik.insectid.pojo.InsectModel;
import com.rakeshmalik.insectid.ui.UIController;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.Response;

class DownloadFileHttpCallback implements Callback {

    private Context context;
    private UIController uiController;
    private MetadataManager metadataManager;
    private Handler mainHandler;
    private SharedPreferences prefs;

    private String fileName;
    private Runnable onSuccess;
    private Runnable onFailure;
    private String downloadName;
    private String modelName;
    private boolean updateRequired;
    private int fileDownloadSeq;
    private int totalFileDownloads;
    private int modelDownloadSeq;
    private int totalModelDownloads;

    PowerManager.WakeLock wakeLock;

    private DownloadFileHttpCallback() {}

    public DownloadFileHttpCallback(Context context, UIController uiController, MetadataManager metadataManager, Handler mainHandler, SharedPreferences prefs,
                                    String fileName, Runnable onSuccess, Runnable onFailure,
                                    String downloadName, String modelName, boolean updateRequired,
                                    int fileDownloadSeq, int totalFileDownloads, int modelDownloadSeq, int totalModelDownloads) {
        this.fileName = fileName;
        this.onSuccess = onSuccess;
        this.onFailure = onFailure;
        this.downloadName = downloadName;
        this.modelName = modelName;
        this.updateRequired = updateRequired;
        this.fileDownloadSeq = fileDownloadSeq;
        this.totalFileDownloads = totalFileDownloads;
        this.modelDownloadSeq = modelDownloadSeq;
        this.totalModelDownloads = totalModelDownloads;
        this.context = context;
        this.uiController = uiController;
        this.metadataManager = metadataManager;
        this.mainHandler = mainHandler;
        this.prefs = prefs;

        //TODO move to foreground with PARTIAL_WAKE_LOCK
        PowerManager powerManager = (PowerManager) context.getSystemService(context.POWER_SERVICE);
        this.wakeLock = powerManager.newWakeLock(PowerManager.FULL_WAKE_LOCK, WAKE_LOCK_NAME);
        this.wakeLock.acquire(WAKE_LOCK_TIME);
    }

    @Override
    public void onFailure(@NonNull Call call, IOException e) {
        Log.e(LOG_TAG, "Download " + downloadName + " failed: " + e.getMessage());
        mainHandler.post(() -> uiController.showMessage(context.getString(R.string.download_failed, downloadName)));
        if(onFailure != null) {
            onFailure.run();
        }
        if (wakeLock != null && wakeLock.isHeld()) {
            wakeLock.release();
        }
    }

    @Override
    public void onResponse(@NonNull Call call, Response response) throws IOException {
        if (!response.isSuccessful()) {
            Log.e(LOG_TAG, "Server error: " + response.code());
            mainHandler.post(() -> uiController.showMessage(context.getString(R.string.download_failed, downloadName)));
            return;
        }
        File filesDir = context.getFilesDir();
        File file = new File(filesDir, fileName);
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
                mainHandler.post(() -> reportDownloadProgress(eta, progress, finalDownloadedBytes, totalBytes));
            }
            Log.d(LOG_TAG, "File downloaded successfully: " + file.getAbsolutePath());
            updatePrefs();
            if(onSuccess != null) {
                onSuccess.run();
            }
        } catch (Exception e) {
            Log.e(LOG_TAG, "Download " + downloadName + " failed: ", e);
            if(Objects.equals(e.getMessage(), "Software caused connection abort")) {
                mainHandler.post(() -> uiController.showMessage(context.getString(R.string.download_connection_aborted, downloadName)));
            } else {
                mainHandler.post(() -> uiController.showMessage(context.getString(R.string.download_failed, downloadName)));
            }
            if(onFailure != null) {
                onFailure.run();
            }
        } finally {
            if (wakeLock != null && wakeLock.isHeld()) {
                wakeLock.release();
            }
        }
    }



    @SuppressLint("DefaultLocale")
    private void reportDownloadProgress(long eta, int progress, long downloadedBytes, long totalBytes) {
        String title;
        String sizeInfo = String.format("%d/%d MB", downloadedBytes / 1024 / 1024, totalBytes / 1024 / 1024);
        String etaInfo = String.format("%d min %d sec remaining", eta / 60000, (eta % 60000) / 1000);
        String countInfo = String.format("File %d of %d", 
                totalFileDownloads * (modelDownloadSeq - 1) + fileDownloadSeq, totalFileDownloads * totalModelDownloads);

        if(updateRequired) {
            int latestVersion = InsectModel.fromJson(modelName, metadataManager.getMetadata(modelName)).getVersion();
            int currentVersion = prefs.getInt(ModelDownloader.modelVersionPrefName(modelName), 0);
            if (currentVersion == 0) {
                title = String.format("Updating %s...\n(Unknown version \u2192 v%d)", downloadName, latestVersion);
            } else {
                title = String.format("Updating %s...\n(v%d \u2192 v%d)", downloadName, currentVersion, latestVersion);
            }
            uiController.showDownloadProgress(title, progress, etaInfo, sizeInfo, countInfo, modelName, downloadName);
        } else {
            title = String.format("Downloading %s...", downloadName);
            uiController.showDownloadProgress(title, progress, etaInfo, sizeInfo, countInfo, modelName, downloadName);
        }
    }

    private void updatePrefs() {
        prefs.edit().putBoolean(ModelDownloader.fileDownloadedPrefName(fileName), true).apply();
        if(downloadName.contains("model")) {
            int version = InsectModel.fromJson(modelName, metadataManager.getMetadata(modelName)).getVersion();
            prefs.edit().putInt(ModelDownloader.modelVersionPrefName(modelName), version).apply();
        }
    }

}
