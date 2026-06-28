package com.rakeshmalik.insectid;

import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;

import com.rakeshmalik.insectid.ui.DownloadListAdapter;
import com.rakeshmalik.insectid.ui.ManageModelsAdapter;
import com.rakeshmalik.insectid.constants.Constants;

import com.rakeshmalik.insectid.ui.ModelSelectorHelper;
import com.rakeshmalik.insectid.ui.UIStateManager;
import com.rakeshmalik.insectid.utils.PhotoPickerHelper;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.google.android.material.button.MaterialButton;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import android.text.Html;

import androidx.recyclerview.widget.RecyclerView;
import com.getkeepsafe.relinker.ReLinker;

import com.rakeshmalik.insectid.filemanager.MetadataManager;
import com.rakeshmalik.insectid.filemanager.ModelDownloader;
import com.rakeshmalik.insectid.filemanager.DownloadItem;
import com.rakeshmalik.insectid.filemanager.ModelLoader;
import com.rakeshmalik.insectid.pojo.InsectModel;
import com.rakeshmalik.insectid.prediction.PredictionRunnable;
import com.rakeshmalik.insectid.prediction.PredictionManager;
import com.rakeshmalik.insectid.ui.PredictionAdapter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

import com.rakeshmalik.insectid.ui.UIController;

public class MainActivity extends AppCompatActivity implements UIController {

    private ImageView imageView;
    private Button buttonPickImage;
    private ImageView welcomeIcon;
    private TextView outputText;
    private View outputTextContainer;
    private View downloadProgressContainer;
    private TextView downloadTitle;
    private com.google.android.material.progressindicator.LinearProgressIndicator downloadProgressBar;
    private TextView downloadEta;
    private TextView downloadSize;
    private TextView downloadCount;
    private RecyclerView downloadListRecyclerView;
    private DownloadListAdapter downloadListAdapter;
    private RecyclerView predictionsRecyclerView;
    private PredictionAdapter predictionAdapter;
    
    private final Map<String, com.rakeshmalik.insectid.prediction.PredictionResponse> predictionCache = new HashMap<>();
    private RecyclerView manageModelsRecyclerView;
    private ManageModelsAdapter manageModelsAdapter;
    private ModelLoader modelLoader;
    private MetadataManager metadataManager;
    private ModelDownloader modelDownloader;
    private PredictionManager predictionManager;
    
    private BottomNavigationView bottomNavigation;
    private MaterialButton btnDownloadAll;
    private MaterialButton btnCancelDownload;
    
    private com.google.android.material.switchmaterial.SwitchMaterial switchShowLegacy;
    private com.google.android.material.switchmaterial.SwitchMaterial switchShowExperimental;
    private com.google.android.material.switchmaterial.SwitchMaterial switchApplyBlur;

    private final ExecutorService executorService = Executors.newSingleThreadExecutor();
    private final ArrayBlockingQueue<Future<?>> runningTasks = new ArrayBlockingQueue<>(10);

    public static final String PREF = "InsectIdPrefs";
    
    private UIStateManager uiStateManager;
    private PhotoPickerHelper photoPickerHelper;
    private ModelSelectorHelper modelSelectorHelper;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        try {
            super.onCreate(savedInstanceState);
            setContentView(R.layout.activity_main);

            this.buttonPickImage = findViewById(R.id.buttonPickImage);
            this.imageView = findViewById(R.id.imageView);
            this.welcomeIcon = findViewById(R.id.welcomeIcon);
            this.outputText = findViewById(R.id.outputText);
            this.outputText.setMovementMethod(android.text.method.LinkMovementMethod.getInstance());
            this.outputTextContainer = findViewById(R.id.outputTextContainer);
            this.downloadProgressContainer = findViewById(R.id.downloadProgressContainer);
            this.downloadTitle = findViewById(R.id.downloadTitle);
            this.downloadProgressBar = findViewById(R.id.downloadProgressBar);
            this.downloadEta = findViewById(R.id.downloadEta);
            this.downloadSize = findViewById(R.id.downloadSize);
            this.downloadCount = findViewById(R.id.downloadCount);
            this.downloadListRecyclerView = findViewById(R.id.downloadListRecyclerView);
            this.predictionsRecyclerView = findViewById(R.id.predictionsRecyclerView);
            
            // No headers to collapse/expand anymore, legacy section is toggled by the card
            
            this.manageModelsRecyclerView = findViewById(R.id.manageModelsRecyclerView);
            this.manageModelsRecyclerView.setLayoutManager(new androidx.recyclerview.widget.LinearLayoutManager(this));
            
            this.metadataManager = new MetadataManager(this, this);
            this.modelLoader = new ModelLoader(this);
            this.modelDownloader = new ModelDownloader(this, this, metadataManager);
            this.predictionManager = new PredictionManager(this, metadataManager, modelLoader);

            
            this.bottomNavigation = findViewById(R.id.bottom_navigation);
            View tabIdentify = findViewById(R.id.tab_identify);
            View tabModels = findViewById(R.id.tab_models);
            View tabSettings = findViewById(R.id.tab_settings);
            
            this.uiStateManager = new UIStateManager(this, bottomNavigation, tabIdentify, tabModels, tabSettings);
            
            this.bottomNavigation.setOnItemSelectedListener(item -> {
                int itemId = item.getItemId();
                if (itemId == R.id.navigation_identify) {
                    tabIdentify.setVisibility(View.VISIBLE);
                    tabModels.setVisibility(View.GONE);
                    tabSettings.setVisibility(View.GONE);
                    if (modelDownloader != null && modelDownloader.isDownloading()) {
                        showMessage("Identify is temporarily disabled during downloads.");
                    }
                    return true;
                } else if (itemId == R.id.navigation_models) {
                    tabIdentify.setVisibility(View.GONE);
                    tabModels.setVisibility(View.VISIBLE);
                    tabSettings.setVisibility(View.GONE);
                    return true;
                } else if (itemId == R.id.navigation_settings) {
                    tabIdentify.setVisibility(View.GONE);
                    tabModels.setVisibility(View.GONE);
                    tabSettings.setVisibility(View.VISIBLE);
                    return true;
                }
                return false;
            });
            
            this.btnDownloadAll = findViewById(R.id.btnDownloadAll);
            this.btnDownloadAll.setOnClickListener(v -> downloadOrUpdateAllModels());
            
            this.btnCancelDownload = findViewById(R.id.btnCancelDownload);
            if (this.btnCancelDownload != null) {
                this.btnCancelDownload.setOnClickListener(v -> {
                    modelDownloader.cancelDownload();
                    hideDownloadProgress();
                    uiStateManager.unlockUI();
                    refreshManageModelsList();
                });
            }
            
            this.photoPickerHelper = new PhotoPickerHelper(this, new PhotoPickerHelper.PhotoPickerCallback() {
                @Override
                public void onPhotoCropped(Uri uri) {
                    predictionCache.clear();
                    imageView.setImageURI(uri);
                    downloadModelAndRunPredictionAsync();
                }

                @Override
                public void onShowMessage(String message) {
                    showMessage(message);
                }

                @Override
                public boolean isUiLocked() {
                    return uiStateManager.isUiLocked();
                }
            });

            this.buttonPickImage.setOnClickListener(v -> {
                if (modelDownloader != null && modelDownloader.isDownloading()) {
                    showMessage("Please wait for downloads to complete.");
                    return;
                }
                photoPickerHelper.showImagePickerDialog();
            });

            android.widget.LinearLayout modelSelectorContainer = findViewById(R.id.modelSelectorContainer);
            TextView identifyModelWarning = findViewById(R.id.identifyModelWarning);
            TextView selectedModelName = findViewById(R.id.selectedModelName);

            this.modelSelectorHelper = new ModelSelectorHelper(this, modelSelectorContainer, identifyModelWarning, selectedModelName, metadataManager, modelDownloader, new ModelSelectorHelper.ModelSelectorCallback() {
                @Override
                public void onModelSelected(InsectModel model) {
                    if (photoPickerHelper.getPhotoUri() != null) {
                        runOnUiThread(() -> {
                            if (welcomeIcon != null) {
                                welcomeIcon.setVisibility(View.GONE);
                            }
                            outputText.setText("");
                        });
                        if (predictionCache.containsKey(model.getModelName())) {
                            showPredictionResponse(predictionCache.get(model.getModelName()));
                        } else {
                            downloadModelAndRunPredictionAsync();
                        }
                    }
                }

                @Override
                public boolean isUiLocked() {
                    return uiStateManager.isUiLocked();
                }

                @Override
                public void showMessage(String message) {
                    MainActivity.this.showMessage(message);
                }
            });

            this.modelSelectorHelper.initMinHeight();

            this.uiStateManager.setLockableViews(buttonPickImage, btnDownloadAll, btnCancelDownload, enabled -> {
                modelSelectorHelper.setChipGroupsEnabled(enabled);
            });
            
            this.switchShowLegacy = findViewById(R.id.switchShowLegacy);
            this.switchShowExperimental = findViewById(R.id.switchShowExperimental);
            this.switchApplyBlur = findViewById(R.id.switchApplyBlur);
            
            SharedPreferences prefs = getSharedPreferences(PREF, Context.MODE_PRIVATE);
            switchShowLegacy.setChecked(prefs.getBoolean("show_legacy_models", false));
            switchShowExperimental.setChecked(prefs.getBoolean("show_experimental_models", true));
            switchApplyBlur.setChecked(prefs.getBoolean("apply_blur_screen_captures", true));
            
            android.widget.CompoundButton.OnCheckedChangeListener switchListener = (buttonView, isChecked) -> {
                SharedPreferences.Editor editor = prefs.edit();
                if (buttonView.getId() == R.id.switchShowLegacy) {
                    editor.putBoolean("show_legacy_models", isChecked);
                } else if (buttonView.getId() == R.id.switchShowExperimental) {
                    editor.putBoolean("show_experimental_models", isChecked);
                } else if (buttonView.getId() == R.id.switchApplyBlur) {
                    editor.putBoolean("apply_blur_screen_captures", isChecked);
                }
                editor.apply();
                
                // Refresh UI
                modelSelectorHelper.populateModelSpinner();
                refreshManageModelsList();
            };
            
            switchShowLegacy.setOnCheckedChangeListener(switchListener);
            switchShowExperimental.setOnCheckedChangeListener(switchListener);
            switchApplyBlur.setOnCheckedChangeListener(switchListener);

            //System.loadLibrary("opencv_java4");
            ReLinker.loadLibrary(this, "opencv_java4");

            new Thread(() -> {
                metadataManager.getMetadata();
                runOnUiThread(() -> {
                    modelSelectorHelper.populateModelSpinner();
                    showWelcome();
                    
                    boolean anyDownloaded = false;
                    for (InsectModel m : modelSelectorHelper.getVisibleModels()) {
                        if (modelDownloader.isModelAlreadyDownloaded(m)) {
                            anyDownloaded = true;
                            break;
                        }
                    }
                    if (!anyDownloaded) {
                        new AlertDialog.Builder(MainActivity.this)
                            .setTitle("Welcome")
                            .setMessage("Please download a model to start identifying insects.")
                            .setPositiveButton("Go to Models", (dialog, which) -> {
                                bottomNavigation.setSelectedItemId(R.id.navigation_models);
                            })
                            .setCancelable(false)
                            .show();
                    }
                });
                executorService.submit(() -> modelSelectorHelper.populateModelSpinner());
            }).start();
            
            this.manageModelsAdapter = new ManageModelsAdapter(new ArrayList<>(), modelDownloader, getSharedPreferences(Constants.PREF, Context.MODE_PRIVATE), this::downloadOrUpdateModel);
            this.manageModelsRecyclerView.setAdapter(this.manageModelsAdapter);
            
            executorService.submit(this::refreshManageModelsList);
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception in MainActivity.onCreate()", ex);
            throw ex;
        }
    }

    public void switchToIdentifyTab() {
        uiStateManager.switchToIdentifyTab();
    }
    
    public void showDownloadProgressContainer() {
        runOnUiThread(() -> {
            downloadProgressContainer.setVisibility(View.VISIBLE);
            uiStateManager.startDownloadIconAnimation();
        });
    }
    

    public void switchToModelsTab() {
        uiStateManager.switchToModelsTab();
    }

    public InsectModel getSelectedModel() {
        return modelSelectorHelper.getSelectedModel();
    }

    public Uri getPhotoUri() {
        return photoPickerHelper.getPhotoUri();
    }

    public TextView getOutputText() {
        return outputText;
    }

    public ImageView getImageView() {
        return imageView;
    }

    public void refreshManageModelsList() {
        List<InsectModel> currentModels = modelSelectorHelper.getVisibleModels();
        runOnUiThread(() -> {
            if (manageModelsAdapter != null) {
                manageModelsAdapter.updateModels(currentModels);
            }
        });
    }

    public synchronized void lockUI() {
        uiStateManager.lockUI();
    }

    public synchronized void unlockUI() {
        uiStateManager.unlockUI();
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        photoPickerHelper.handleActivityResult(requestCode, resultCode, data);
    }

    private void downloadModelAndRunPredictionAsync() {
        if (modelDownloader.isDownloading()) {
            showMessage("Identify is disabled while downloads are in progress.");
            return;
        }
        if(!runningTasks.isEmpty()) {
            Log.d(LOG_TAG, "Previous tasks still running. Going to try killing them.");
            showMessage(getString(R.string.please_wait));
        }
        while(!runningTasks.isEmpty()) {
            Future<?> future = runningTasks.poll();
            if(future != null) {
                future.cancel(true);
                Log.d(LOG_TAG, "Task " + future + " killed");
            }
        }
        PredictionRunnable runnable = new PredictionRunnable(this, predictionManager, modelLoader, modelDownloader);
        Future<?> future = executorService.submit(runnable);
        runningTasks.add(future);
    }



    private void downloadOrUpdateModel() {
        if (modelSelectorHelper.getSelectedModel() != null) {
            downloadOrUpdateModel(modelSelectorHelper.getSelectedModel());
        }
    }

    private void downloadOrUpdateModel(InsectModel model) {
        showDownloadProgressContainer();
        
        Runnable onSuccess = () -> {
            runOnUiThread(() -> {
                refreshManageModelsList();
                if (!modelDownloader.isDownloading()) {
                    hideDownloadProgress();
                    showMessage("Downloads completed");
                } else {
                    refreshActiveDownloadsPlan();
                }
            });
        };
        
        Runnable onFailure = () -> {
            runOnUiThread(() -> {
                refreshManageModelsList();
                if (!modelDownloader.isDownloading()) {
                    hideDownloadProgress();
                } else {
                    refreshActiveDownloadsPlan();
                }
            });
        };
        
        modelDownloader.downloadModel(model, onSuccess, onFailure, true, 1, 1);
        runOnUiThread(() -> {
            refreshActiveDownloadsPlan();
            refreshManageModelsList();
        });
    }

    private void refreshActiveDownloadsPlan() {
        List<InsectModel> activeModels = modelDownloader.getQueuedAndDownloadingModels();
        List<DownloadItem> plan = modelDownloader.generateDownloadPlan(activeModels, true);
        initDownloadList(plan);
        if (activeModels.isEmpty()) {
            hideDownloadProgress();
        } else {
            showDownloadProgressContainer();
        }
    }

    private void downloadOrUpdateAllModels() {
        showDownloadProgressContainer();
        
        List<InsectModel> modelsToDownload = modelSelectorHelper.getVisibleModels().stream()
                .filter(m -> modelDownloader.isModelDownloadOrUpdateRequired(m) && !modelDownloader.isModelQueuedOrDownloading(m))
                .collect(Collectors.toList());
        
        if (modelsToDownload.isEmpty()) {
            hideDownloadProgress();
            showMessage("All models are already up to date or queued");
            return;
        }

        Runnable onSuccess = () -> {
            runOnUiThread(() -> {
                refreshManageModelsList();
                if (!modelDownloader.isDownloading()) {
                    hideDownloadProgress();
                    showMessage("All downloads completed");
                } else {
                    refreshActiveDownloadsPlan();
                }
            });
        };
        
        Runnable onFailure = () -> {
            runOnUiThread(() -> {
                refreshManageModelsList();
                if (!modelDownloader.isDownloading()) {
                    hideDownloadProgress();
                } else {
                    refreshActiveDownloadsPlan();
                }
            });
        };

        for(int i = 0; i < modelsToDownload.size(); i++) {
            InsectModel model = modelsToDownload.get(i);
            modelDownloader.downloadModel(model, onSuccess, onFailure, true, i + 1, modelsToDownload.size());
        }
        
        runOnUiThread(() -> {
            refreshActiveDownloadsPlan();
            refreshManageModelsList();
        });
    }

    @Override
    public void showMessage(CharSequence msg) {
        runOnUiThread(() -> {
            if (welcomeIcon != null) {
                welcomeIcon.setVisibility(View.GONE);
            }
            outputText.setText(msg);
            outputTextContainer.setVisibility(View.VISIBLE);
            if (predictionsRecyclerView != null) {
                predictionsRecyclerView.setVisibility(View.GONE);
            }
        });
    }

    private void showWelcome() {
        runOnUiThread(() -> {
            if (welcomeIcon != null) {
                welcomeIcon.setVisibility(View.VISIBLE);
            }
            outputText.setText("Ready to identify!\nSelect an image to begin.");
            outputTextContainer.setVisibility(View.VISIBLE);
            if (predictionsRecyclerView != null) {
                predictionsRecyclerView.setVisibility(View.GONE);
            }
        });
    }

    @Override
    public void showPredictionResponse(com.rakeshmalik.insectid.prediction.PredictionResponse response) {
        runOnUiThread(() -> {
            if (modelSelectorHelper.getSelectedModel() != null && response != null) {
                predictionCache.put(modelSelectorHelper.getSelectedModel().getModelName(), response);
            }
            if (response.errorMessage != null) {
                if (welcomeIcon != null) {
                    welcomeIcon.setVisibility(View.GONE);
                }
                outputText.setText(Html.fromHtml(response.errorMessage, Html.FROM_HTML_MODE_COMPACT, null, null));
                outputTextContainer.setVisibility(View.VISIBLE);
                if (predictionsRecyclerView != null) {
                    predictionsRecyclerView.setVisibility(View.GONE);
                }
                return;
            }

            outputTextContainer.setVisibility(View.GONE);
            predictionsRecyclerView.setVisibility(View.VISIBLE);
            
            if (predictionAdapter == null) {
                predictionAdapter = new PredictionAdapter(this, response.predictions);
                predictionsRecyclerView.setAdapter(predictionAdapter);
            } else {
                predictionAdapter.updateData(response.predictions);
            }
            predictionsRecyclerView.scrollToPosition(0);
        });
    }

    @Override
    public void initDownloadList(List<DownloadItem> items) {
        runOnUiThread(() -> {
            if (downloadListAdapter == null) {
                downloadListAdapter = new DownloadListAdapter(items);
                downloadListRecyclerView.setAdapter(downloadListAdapter);
            } else {
                downloadListAdapter.setItems(items);
            }
        });
    }

    @Override
    public void showDownloadProgress(String title, int progress, String eta, String sizeInfo, String countInfo, String modelName, String downloadName) {
        runOnUiThread(() -> {
            
            downloadTitle.setText(title);
            if (progress >= 0) {
                downloadProgressBar.setIndeterminate(false);
                downloadProgressBar.setProgressCompat(progress, true);
            } else {
                downloadProgressBar.setIndeterminate(true);
            }
            downloadEta.setText(eta);
            downloadSize.setText(sizeInfo);
            
            if (downloadListAdapter != null) {
                String realCountInfo = downloadListAdapter.getCountInfo(modelName, downloadName);
                if (!realCountInfo.isEmpty()) {
                    downloadCount.setText(realCountInfo);
                } else {
                    downloadCount.setText(countInfo);
                }
                downloadListAdapter.updateProgress(modelName, downloadName);
            } else {
                downloadCount.setText(countInfo);
            }
        });
    }

    @Override
    public void hideDownloadProgress() {
        runOnUiThread(() -> {
            downloadProgressContainer.setVisibility(View.GONE);
            downloadTitle.setText("No active downloads");
            downloadProgressBar.setIndeterminate(false);
            downloadProgressBar.setProgressCompat(0, true);
            downloadEta.setText("");
            downloadSize.setText("");
            downloadCount.setText("Up to date");
            uiStateManager.stopDownloadIconAnimation();
        });
    }

    @Override
    public void showDownloadFailedPopup(String title, String message, Runnable onResume, Runnable onCancel) {
        new AlertDialog.Builder(this)
                .setTitle(title)
                .setMessage(message)
                .setPositiveButton("Retry", (dialog, which) -> {
                    if (onResume != null) onResume.run();
                })
                .setNegativeButton("Cancel", (dialog, which) -> {
                    if (onCancel != null) onCancel.run();
                })
                .setCancelable(false)
                .show();
    }
    
    public com.rakeshmalik.insectid.filemanager.MetadataManager getMetadataManager() {
        return metadataManager;
    }
}
