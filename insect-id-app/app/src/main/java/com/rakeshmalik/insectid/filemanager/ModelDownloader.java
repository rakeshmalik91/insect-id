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
import com.rakeshmalik.insectid.ui.UIController;

import java.io.File;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.ArrayList;
import java.util.List;

import com.rakeshmalik.insectid.constants.Constants;

import okhttp3.*;

//TODO have auto-update, pre-download & verify-integrity/re-download model settings
public class ModelDownloader {

    private final List<DownloadRequest> downloadQueue = new ArrayList<>();
    private Call currentCall;
    private boolean isDownloading = false;
    private DownloadRequest currentDownloadRequest = null;

    private static class DownloadRequest {
        InsectModel model;
        Runnable onSuccess;
        Runnable onFailure;
        boolean forceUpdate;
        int modelDownloadSeq;
        int totalModelDownloads;

        DownloadRequest(InsectModel model, Runnable onSuccess, Runnable onFailure, boolean forceUpdate, int modelDownloadSeq, int totalModelDownloads) {
            this.model = model;
            this.onSuccess = onSuccess;
            this.onFailure = onFailure;
            this.forceUpdate = forceUpdate;
            this.modelDownloadSeq = modelDownloadSeq;
            this.totalModelDownloads = totalModelDownloads;
        }
    }

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
    private final UIController uiController;
    private final MetadataManager metadataManager;

    public ModelDownloader(Context context, UIController uiController, MetadataManager metadataManager) {
        this.context = context;
        this.uiController = uiController;
        this.prefs = context.getSharedPreferences(PREF, Context.MODE_PRIVATE);
        this.metadataManager = metadataManager;
    }

    public boolean isModelDownloadOrUpdateRequired(InsectModel model) {
        return !isModelAlreadyDownloaded(model) || isModelToBeUpdated(model);
    }

    public boolean isModelAlreadyDownloaded(InsectModel model) {
        String classesFileName = String.format(CLASSES_FILE_NAME_FMT, model.getModelName());
        String classDetailsFileName = String.format(CLASS_DETAILS_FILE_NAME_FMT, model.getModelName());
        String imagesArchiveFileName = Constants.getImagesArchiveFileName(context, model.getModelName(), model.getImagesUrl());
        String modelFileName = String.format(MODEL_FILE_NAME_FMT, model.getModelName());
        return isFileAlreadyDownloaded(classesFileName)
                && isFileAlreadyDownloaded(classDetailsFileName)
                && isFileAlreadyDownloaded(imagesArchiveFileName)
                && isFileAlreadyDownloaded(modelFileName);
    }

    public boolean isModelToBeUpdated(InsectModel model) {
        int currentVersion = prefs.getInt(modelVersionPrefName(model.getModelName()), 0);
        int latestVersion = model.getVersion();
        Log.d(LOG_TAG, String.format("Model type: %s, current version: %d, latest version: %d", model.getModelName(), currentVersion, latestVersion));
        return currentVersion < latestVersion;
    }

    public void downloadModel(InsectModel model, Runnable onSuccess, Runnable onFailure, int modelDownloadSeq, int totalModelDownloads) {
        downloadModel(model, onSuccess, onFailure, false, modelDownloadSeq, totalModelDownloads);
    }

    private boolean isRootClassifierDownloadRequired() {
        String modelFileName = String.format(MODEL_FILE_NAME_FMT, ROOT_CLASSIFIER);
        int currentVersion = prefs.getInt(modelVersionPrefName(ROOT_CLASSIFIER), 0);
        int latestVersion = InsectModel.fromJson(ROOT_CLASSIFIER, metadataManager.getMetadata(ROOT_CLASSIFIER)).getVersion();
        Log.d(LOG_TAG, String.format("Model type: %s, current version: %d, latest version: %d", ROOT_CLASSIFIER, currentVersion, latestVersion));
        return !isFileAlreadyDownloaded(modelFileName) || currentVersion < latestVersion;
    }

    private void downloadRootClassifier(Runnable onSuccess, Runnable onFailure, boolean forceUpdate, int modelDownloadSeq, int totalModelDownloads) {
        Log.d(LOG_TAG, "Inside ModelDownloader.downloadRootClassifier()");
        String modelFileName = String.format(MODEL_FILE_NAME_FMT, ROOT_CLASSIFIER);
        if(isRootClassifierDownloadRequired()) {
            String fileUrl = InsectModel.fromJson(ROOT_CLASSIFIER, metadataManager.getMetadata(ROOT_CLASSIFIER)).getModelUrl();
            boolean updateRequired = isFileAlreadyDownloaded(modelFileName);
            downloadFile(modelFileName, fileUrl, onSuccess, onFailure, "Root Classifier model", ROOT_CLASSIFIER, updateRequired,
                    1, 5, modelDownloadSeq, totalModelDownloads);
        } else {
            Log.d(LOG_TAG, "Model " + ROOT_CLASSIFIER + " already downloaded.");
            onSuccess.run();
        }
    }

    public void downloadModel(InsectModel model, Runnable onSuccess, Runnable onFailure, boolean forceUpdate, int modelDownloadSeq, int totalModelDownloads) {
        downloadQueue.add(new DownloadRequest(model, onSuccess, onFailure, forceUpdate, modelDownloadSeq, totalModelDownloads));
        if (!isDownloading) {
            processNextDownload();
        }
    }
    
    public boolean isModelQueuedOrDownloading(InsectModel model) {
        if (currentDownloadRequest != null && currentDownloadRequest.model.getModelName().equals(model.getModelName())) {
            return true;
        }
        for (DownloadRequest req : downloadQueue) {
            if (req.model.getModelName().equals(model.getModelName())) {
                return true;
            }
        }
        return false;
    }
    
    private void processNextDownload() {
        if (downloadQueue.isEmpty()) {
            isDownloading = false;
            currentDownloadRequest = null;
            return;
        }
        isDownloading = true;
        currentDownloadRequest = downloadQueue.remove(0);
        executeDownload(currentDownloadRequest);
    }
    
    public boolean isDownloading() {
        return isDownloading;
    }
    
    public List<InsectModel> getQueuedAndDownloadingModels() {
        List<InsectModel> list = new ArrayList<>();
        if (currentDownloadRequest != null) {
            list.add(currentDownloadRequest.model);
        }
        for (DownloadRequest req : downloadQueue) {
            list.add(req.model);
        }
        return list;
    }
    
    public void cancelDownload() {
        downloadQueue.clear();
        if (currentCall != null) {
            currentCall.cancel();
            currentCall = null;
        }
        isDownloading = false;
        currentDownloadRequest = null;
        mainHandler.post(() -> uiController.hideDownloadProgress());
    }

    private void executeDownload(DownloadRequest request) {
        new Thread(() -> {
            InsectModel model = request.model;
            Runnable originalOnSuccess = request.onSuccess;
            Runnable originalOnFailure = request.onFailure;
            boolean forceUpdate = request.forceUpdate;
            int modelDownloadSeq = request.modelDownloadSeq;
            int totalModelDownloads = request.totalModelDownloads;

            Runnable wrappedOnSuccess = () -> {
                mainHandler.post(this::processNextDownload);
                if (originalOnSuccess != null) originalOnSuccess.run();
            };
            
            Runnable wrappedOnFailure = () -> {
                mainHandler.post(this::processNextDownload);
                if (originalOnFailure != null) originalOnFailure.run();
            };

            Log.d(LOG_TAG, "Inside ModelDownloader.executeDownload()");
            try {
                if(forceUpdate) {
                    metadataManager.getMetadata(true);
                }

                final String classesFileName = String.format(CLASSES_FILE_NAME_FMT, model.getModelName());
                final String classDetailsFileName = String.format(CLASS_DETAILS_FILE_NAME_FMT, model.getModelName());
                final String modelFileName = String.format(MODEL_FILE_NAME_FMT, model.getModelName());
                final String imagesFileName = Constants.getImagesArchiveFileName(context, model.getModelName(), model.getImagesUrl());

                final String classesFileUrl = model.getClassesUrl();
                final String classDetailsFileUrl = model.getClassDetailsUrl();
                final String imagesFileUrl = model.getImagesUrl();
                final String modelFileUrl = model.getModelUrl();

                Runnable downloadModelData = () -> {
                    final boolean updateRequired;
                    if (isModelAlreadyDownloaded(model)) {
                        updateRequired = forceUpdate && isModelToBeUpdated(model);
                        if(updateRequired) {
                            Log.d(LOG_TAG, "Going to update " + model.getModelName() + " model");
                        } else {
                            if(forceUpdate) {
                                int currentVersion = prefs.getInt(modelVersionPrefName(model.getModelName()), 0);
                                Log.d(LOG_TAG, "Model already up to date. Version: " + currentVersion);
                            }
                            Log.d(LOG_TAG, "Model " + model.getModelName() + " already downloaded.");
                            wrappedOnSuccess.run();
                            return;
                        }
                    } else {
                        updateRequired = false;
                        Log.d(LOG_TAG, "Going to download " + model.getModelName() + " model");
                    }

                    Runnable downloadModelFile = () -> downloadFile(modelFileName, modelFileUrl, wrappedOnSuccess, wrappedOnFailure,
                            model.getDisplayName() + " model", model.getModelName(), updateRequired,
                            5, 5, modelDownloadSeq, totalModelDownloads);

                    Runnable downloadImageArchive = () -> downloadFile(imagesFileName, imagesFileUrl, downloadModelFile, wrappedOnFailure,
                            model.getDisplayName() + " images", model.getModelName(), updateRequired,
                            4, 5, modelDownloadSeq, totalModelDownloads);

                    Runnable downloadClassDetails = () -> downloadFile(classDetailsFileName, classDetailsFileUrl, downloadImageArchive, wrappedOnFailure,
                            model.getDisplayName() + " metadata", model.getModelName(), updateRequired,
                            3, 5, modelDownloadSeq, totalModelDownloads);

                    downloadFile(classesFileName, classesFileUrl, downloadClassDetails, wrappedOnFailure,
                            model.getDisplayName() + " classes", model.getModelName(), updateRequired,
                            2, 5, modelDownloadSeq, totalModelDownloads);
                };

                downloadRootClassifier(downloadModelData, wrappedOnFailure, forceUpdate, modelDownloadSeq, totalModelDownloads);
            } catch(Exception ex) {
                Log.e(LOG_TAG, "Exception downloading model", ex);
                wrappedOnFailure.run();
            }
        }).start();
    }

    public void offloadModel(InsectModel model) {
        String classesFileName = String.format(CLASSES_FILE_NAME_FMT, model.getModelName());
        String classDetailsFileName = String.format(CLASS_DETAILS_FILE_NAME_FMT, model.getModelName());
        String modelFileName = String.format(MODEL_FILE_NAME_FMT, model.getModelName());
        String imagesFileName = Constants.getImagesArchiveFileName(context, model.getModelName(), model.getImagesUrl());

        // 1. Delete model specific files
        deleteFileAndPref(classesFileName);
        deleteFileAndPref(classDetailsFileName);
        deleteFileAndPref(modelFileName);

        // 2. Delete images archive only if no other downloaded model uses it
        boolean isImagesUsed = false;
        for (InsectModel otherModel : metadataManager.getAvailableModels()) {
            if (otherModel.getModelName().equals(model.getModelName())) continue;
            
            if (isModelAlreadyDownloaded(otherModel)) {
                String otherImagesFileName = Constants.getImagesArchiveFileName(context, otherModel.getModelName(), otherModel.getImagesUrl());
                if (imagesFileName.equals(otherImagesFileName)) {
                    isImagesUsed = true;
                    Log.d(LOG_TAG, "Images archive " + imagesFileName + " is shared with " + otherModel.getModelName() + " and will not be deleted.");
                    break;
                }
            }
        }
        
        if (!isImagesUsed) {
            deleteFileAndPref(imagesFileName);
        }
        
        // 3. Remove version info
        prefs.edit().remove(modelVersionPrefName(model.getModelName())).apply();
    }

    private void deleteFileAndPref(String fileName) {
        File file = new File(context.getFilesDir(), fileName);
        if (file.exists()) {
            boolean deleted = file.delete();
            Log.d(LOG_TAG, "Deleted file: " + fileName + " success: " + deleted);
        }
        prefs.edit().remove(fileDownloadedPrefName(fileName)).apply();
    }

    private void downloadFile(String fileName, String fileUrl, Runnable onSuccess, Runnable onFailure,
                              String downloadName, String modelName, boolean updateRequired,
                              int fileDownloadSeq, int totalFileDownloads,
                              int modelDownloadSeq, int totalModelDownloads) {
        if (!updateRequired && isFileAlreadyDownloaded(fileName)) {
            Log.d(LOG_TAG, "File " + fileName + " is already downloaded, skipping...");
            if (onSuccess != null) onSuccess.run();
            return;
        }

        Runnable popupOnFailure = () -> {
            mainHandler.post(() -> {
                uiController.showDownloadFailedPopup(
                        "Download Failed",
                        "Failed to download " + downloadName + ". Do you want to retry?",
                        () -> downloadFile(fileName, fileUrl, onSuccess, onFailure, downloadName, modelName, updateRequired, fileDownloadSeq, totalFileDownloads, modelDownloadSeq, totalModelDownloads),
                        onFailure
                );
            });
        };

        try {
            Log.d(LOG_TAG, "Downloading " + downloadName + " " + fileName + " from " + fileUrl + "...");
            DownloadFileHttpCallback callback = new DownloadFileHttpCallback(context, uiController, metadataManager, mainHandler, prefs,
                    fileName, onSuccess, popupOnFailure, downloadName, modelName, updateRequired,
                    fileDownloadSeq, totalFileDownloads, modelDownloadSeq, totalModelDownloads);
            currentCall = client.newCall(new Request.Builder().url(fileUrl).build());
            currentCall.enqueue(callback);
        } catch(Exception ex) {
            Log.e(LOG_TAG, "Exception downloading " + fileUrl, ex);
            if(popupOnFailure != null) {
                popupOnFailure.run();
            }
        }
    }

    private boolean isFileAlreadyDownloaded(String fileName) {
        File file = new File(context.getFilesDir(), fileName);
        return file.exists() && prefs.getBoolean(fileDownloadedPrefName(fileName), false);
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

    public List<DownloadItem> generateDownloadPlan(List<InsectModel> models, boolean forceUpdate) {
        List<DownloadItem> plan = new ArrayList<>();
        
        if (isRootClassifierDownloadRequired()) {
            DownloadItem rootModel = new DownloadItem(DownloadItem.TYPE_MODEL, "Root Classifier", ROOT_CLASSIFIER);
            String modelFileName = String.format(MODEL_FILE_NAME_FMT, ROOT_CLASSIFIER);
            DownloadItem fileItem = new DownloadItem(DownloadItem.TYPE_FILE, "Root Classifier model", ROOT_CLASSIFIER + "_" + modelFileName);
            fileItem.parent = rootModel;
            rootModel.children.add(fileItem);
            plan.add(rootModel);
        }
        
        for (InsectModel model : models) {
            boolean updateRequired = false;
            if (isModelAlreadyDownloaded(model)) {
                updateRequired = forceUpdate && isModelToBeUpdated(model);
                if (!updateRequired) {
                    continue;
                }
            } else {
                updateRequired = true;
            }
            
            if (updateRequired) {
                DownloadItem modelItem = new DownloadItem(DownloadItem.TYPE_MODEL, model.getDisplayName(), model.getModelName());
                
                String classesFileName = String.format(CLASSES_FILE_NAME_FMT, model.getModelName());
                DownloadItem classesFile = new DownloadItem(DownloadItem.TYPE_FILE, model.getDisplayName() + " classes", model.getModelName() + "_" + classesFileName);
                classesFile.parent = modelItem;
                modelItem.children.add(classesFile);
                
                String classDetailsFileName = String.format(CLASS_DETAILS_FILE_NAME_FMT, model.getModelName());
                DownloadItem detailsFile = new DownloadItem(DownloadItem.TYPE_FILE, model.getDisplayName() + " metadata", model.getModelName() + "_" + classDetailsFileName);
                detailsFile.parent = modelItem;
                modelItem.children.add(detailsFile);
                
                String imagesFileName = Constants.getImagesArchiveFileName(context, model.getModelName(), model.getImagesUrl());
                DownloadItem imagesFile = new DownloadItem(DownloadItem.TYPE_FILE, model.getDisplayName() + " images", model.getModelName() + "_" + imagesFileName);
                imagesFile.parent = modelItem;
                modelItem.children.add(imagesFile);
                
                String modelFileName = String.format(MODEL_FILE_NAME_FMT, model.getModelName());
                DownloadItem modelFile = new DownloadItem(DownloadItem.TYPE_FILE, model.getDisplayName() + " model", model.getModelName() + "_" + modelFileName);
                modelFile.parent = modelItem;
                modelItem.children.add(modelFile);
                
                plan.add(modelItem);
            }
        }
        return plan;
    }

}
