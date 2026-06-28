package com.rakeshmalik.insectid.ui;

import com.rakeshmalik.insectid.prediction.PredictionResult;
import com.rakeshmalik.insectid.filemanager.DownloadItem;
import java.util.List;

public interface UIController {
    void showMessage(CharSequence msg);
    void showPredictions(List<PredictionResult> predictions);
    void initDownloadList(List<DownloadItem> items);
    void showDownloadProgress(String title, int progress, String eta, String sizeInfo, String countInfo, String modelName, String downloadName);
    void hideDownloadProgress();
}
