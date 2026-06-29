package com.rakeshmalik.insectid.ui;

import com.rakeshmalik.insectid.prediction.PredictionResult;
import com.rakeshmalik.insectid.filemanager.DownloadItem;
import java.util.List;

public interface UIController {
    void showMessage(CharSequence msg);
    void showPredictionResponse(com.rakeshmalik.insectid.prediction.PredictionResponse response);
    void initDownloadList(List<DownloadItem> items);
    void showDownloadProgress(String title, int progress, String eta, String sizeInfo, String countInfo, String modelName, String downloadName);
    void hideDownloadProgress();
    void showDownloadFailedPopup(String title, String message, Runnable onResume, Runnable onCancel);
    void showToast(String message);
}
