package com.rakeshmalik.insectid;

import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;

import com.rakeshmalik.insectid.adapters.DownloadListAdapter;
import com.rakeshmalik.insectid.adapters.ManageModelsAdapter;
import com.rakeshmalik.insectid.constants.Constants;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;


import android.widget.Button;
import android.widget.ImageView;

import android.widget.TextView;
import android.widget.Toast;
import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.google.android.material.button.MaterialButton;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import android.text.Html;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
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
import com.yalantis.ucrop.UCrop;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
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
    private Uri photoUri;
    private android.widget.LinearLayout modelSelectorContainer;
    private TextView identifyModelWarning;
    private TextView selectedModelName;
    private InsectModel selectedModel;
    private ModelLoader modelLoader;
    private MetadataManager metadataManager;
    private ModelDownloader modelDownloader;
    private PredictionManager predictionManager;
    
    private View tabIdentify;
    private View tabModels;
    private View tabSettings;
    private BottomNavigationView bottomNavigation;
    private MaterialButton btnDownloadAll;
    private MaterialButton btnCancelDownload;
    
    private com.google.android.material.switchmaterial.SwitchMaterial switchShowLegacy;
    private com.google.android.material.switchmaterial.SwitchMaterial switchShowExperimental;
    private com.google.android.material.switchmaterial.SwitchMaterial switchApplyBlur;

    private final ExecutorService executorService = Executors.newSingleThreadExecutor();
    private final ArrayBlockingQueue<Future<?>> runningTasks = new ArrayBlockingQueue<>(10);

    private boolean uiLocked = false;
    public static final String PREF = "InsectIdPrefs";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        try {
            super.onCreate(savedInstanceState);
            setContentView(R.layout.activity_main);

            this.buttonPickImage = findViewById(R.id.buttonPickImage);
            this.imageView = findViewById(R.id.imageView);
            this.buttonPickImage.setOnClickListener(v -> showImagePickerDialog());
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
            this.modelSelectorContainer = findViewById(R.id.modelSelectorContainer);
            this.identifyModelWarning = findViewById(R.id.identifyModelWarning);
            this.selectedModelName = findViewById(R.id.selectedModelName);
            
            // No headers to collapse/expand anymore, legacy section is toggled by the card
            
            this.manageModelsRecyclerView = findViewById(R.id.manageModelsRecyclerView);
            this.manageModelsRecyclerView.setLayoutManager(new androidx.recyclerview.widget.LinearLayoutManager(this));
            
            this.metadataManager = new MetadataManager(this, this);
            this.modelLoader = new ModelLoader(this);
            this.modelDownloader = new ModelDownloader(this, this, metadataManager);
            this.predictionManager = new PredictionManager(this, metadataManager, modelLoader);

            this.tabIdentify = findViewById(R.id.tab_identify);
            this.tabModels = findViewById(R.id.tab_models);
            this.tabSettings = findViewById(R.id.tab_settings);
            
            this.bottomNavigation = findViewById(R.id.bottom_navigation);
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
                    unlockUI();
                    refreshManageModelsList();
                });
            }
            
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
                populateModelSpinner();
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
                    createModelTypeSpinner();
                    showWelcome();
                    
                    boolean anyDownloaded = false;
                    for (InsectModel m : getVisibleModels()) {
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
                executorService.submit(this::populateModelSpinner);
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
        runOnUiThread(() -> {
            bottomNavigation.setSelectedItemId(R.id.navigation_identify);
        });
    }
    
    public void showDownloadProgressContainer() {
        runOnUiThread(() -> {
            downloadProgressContainer.setVisibility(View.VISIBLE);
            startDownloadIconAnimation();
        });
    }
    
    private android.animation.ObjectAnimator downloadIconAnimator;
    private android.animation.ObjectAnimator identifyIconAnimator;
    
    private void startDownloadIconAnimation() {
        try {
            com.google.android.material.badge.BadgeDrawable badge = bottomNavigation.getOrCreateBadge(R.id.navigation_models);
            badge.setVisible(true);
            badge.setBackgroundColor(android.graphics.Color.parseColor("#4CAF50"));
            badge.clearNumber();
            
            View modelsTab = bottomNavigation.findViewById(R.id.navigation_models);
            if (modelsTab != null && downloadIconAnimator == null) {
                downloadIconAnimator = android.animation.ObjectAnimator.ofFloat(modelsTab, "alpha", 1f, 0.4f);
                downloadIconAnimator.setDuration(600);
                downloadIconAnimator.setRepeatCount(android.animation.ValueAnimator.INFINITE);
                downloadIconAnimator.setRepeatMode(android.animation.ValueAnimator.REVERSE);
                downloadIconAnimator.start();
            }
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception animating download icon", ex);
        }
    }
    
    private void stopDownloadIconAnimation() {
        try {
            bottomNavigation.removeBadge(R.id.navigation_models);
            if (downloadIconAnimator != null) {
                downloadIconAnimator.cancel();
                downloadIconAnimator = null;
            }
            View modelsTab = bottomNavigation.findViewById(R.id.navigation_models);
            if (modelsTab != null) {
                modelsTab.setAlpha(1f);
            }
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception stopping download icon animation", ex);
        }
    }
    
    private void startIdentifyIconAnimation() {
        try {
            com.google.android.material.badge.BadgeDrawable badge = bottomNavigation.getOrCreateBadge(R.id.navigation_identify);
            badge.setVisible(true);
            badge.setBackgroundColor(android.graphics.Color.parseColor("#FF9800"));
            badge.clearNumber();
            
            View identifyTab = bottomNavigation.findViewById(R.id.navigation_identify);
            if (identifyTab != null && identifyIconAnimator == null) {
                identifyIconAnimator = android.animation.ObjectAnimator.ofFloat(identifyTab, "alpha", 1f, 0.4f);
                identifyIconAnimator.setDuration(600);
                identifyIconAnimator.setRepeatCount(android.animation.ValueAnimator.INFINITE);
                identifyIconAnimator.setRepeatMode(android.animation.ValueAnimator.REVERSE);
                identifyIconAnimator.start();
            }
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception animating identify icon", ex);
        }
    }
    
    private void stopIdentifyIconAnimation() {
        try {
            bottomNavigation.removeBadge(R.id.navigation_identify);
            if (identifyIconAnimator != null) {
                identifyIconAnimator.cancel();
                identifyIconAnimator = null;
            }
            View identifyTab = bottomNavigation.findViewById(R.id.navigation_identify);
            if (identifyTab != null) {
                identifyTab.setAlpha(1f);
            }
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception stopping identify icon animation", ex);
        }
    }
    
    private List<InsectModel> getVisibleModels() {
        SharedPreferences prefs = getSharedPreferences(PREF, Context.MODE_PRIVATE);
        boolean showLegacy = prefs.getBoolean("show_legacy_models", false);
        boolean showExperimental = prefs.getBoolean("show_experimental_models", true);
        
        return metadataManager.getAvailableModels().stream()
                .filter(m -> {
                    if (m.isLegacy() && !showLegacy) return false;
                    if (m.isExperimental() && !showExperimental) return false;
                    return true;
                })
                .collect(Collectors.toList());
    }

    public void switchToModelsTab() {
        runOnUiThread(() -> {
            bottomNavigation.setSelectedItemId(R.id.navigation_models);
        });
    }

    public InsectModel getSelectedModel() {
        return selectedModel;
    }

    public Uri getPhotoUri() {
        return photoUri;
    }

    public TextView getOutputText() {
        return outputText;
    }

    public ImageView getImageView() {
        return imageView;
    }

    private void createModelTypeSpinner() {
        try {
            populateModelSpinner();
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception during model selector creation", ex);
        }
    }
    
    private void populateModelSpinner() {
        List<InsectModel> availableModels = getVisibleModels();
        runOnUiThread(() -> {
            modelSelectorContainer.removeAllViews();

            // Use post to ensure the container has been measured
            modelSelectorContainer.post(() -> {
                // Get the HorizontalScrollView's content width (accounts for all outer padding already)
                View scrollParent = (View) modelSelectorContainer.getParent();
                int availableWidth = scrollParent.getWidth() - scrollParent.getPaddingLeft() - scrollParent.getPaddingRight();
                // Each card from item_model_selector.xml has: marginStart=4dp + marginEnd=12dp = 16dp
                float density = getResources().getDisplayMetrics().density;
                int perCardMargin = (int) (16 * density);
                int cardSize = (availableWidth - (perCardMargin * 4)) / 4;
                // Ensure a reasonable minimum
                int minSize = (int) (48 * density);
                if (cardSize < minSize) cardSize = minSize;

                for (int i = 0; i < availableModels.size(); i++) {
                    InsectModel model = availableModels.get(i);
                    View cardView = getLayoutInflater().inflate(R.layout.item_model_selector, modelSelectorContainer, false);
                    com.google.android.material.card.MaterialCardView card = cardView.findViewById(R.id.modelCard);
                    ImageView icon = cardView.findViewById(R.id.modelIcon);
                    ImageView typeIcon = cardView.findViewById(R.id.typeIcon);

                    // Set dynamic card size
                    android.view.ViewGroup.LayoutParams lp = card.getLayoutParams();
                    lp.width = cardSize;
                    lp.height = cardSize;
                    card.setLayoutParams(lp);

                    if (model.getIcon() != null && !model.getIcon().isEmpty()) {
                        int resId = getResources().getIdentifier(model.getIcon(), "drawable", getPackageName());
                        if (resId != 0) {
                            icon.setImageResource(resId);
                        }
                    }

                    if (model.getIconColor() != null && !model.getIconColor().isEmpty()) {
                        try {
                            icon.setColorFilter(android.graphics.Color.parseColor(model.getIconColor()));
                        } catch (Exception ignored) {}
                    }

                    if (model.isLegacy()) {
                        typeIcon.setImageResource(R.drawable.ic_legacy);
                        typeIcon.setColorFilter(android.graphics.Color.parseColor("#FF9800"));
                        typeIcon.setVisibility(View.VISIBLE);
                    } else if (model.isExperimental()) {
                        typeIcon.setImageResource(R.drawable.ic_experimental);
                        typeIcon.setColorFilter(android.graphics.Color.parseColor("#9C27B0"));
                        typeIcon.setVisibility(View.VISIBLE);
                    } else {
                        typeIcon.setVisibility(View.GONE);
                    }

                    ImageView downloadStatusIcon = cardView.findViewById(R.id.downloadStatusIcon);
                    if (!modelDownloader.isModelAlreadyDownloaded(model)) {
                        downloadStatusIcon.setVisibility(View.VISIBLE);
                        downloadStatusIcon.setColorFilter(android.graphics.Color.parseColor("#9E9E9E"));
                    } else {
                        downloadStatusIcon.setVisibility(View.GONE);
                    }

                    card.setOnClickListener(v -> onModelChipSelected(model));
                    card.setTag(model);
                    modelSelectorContainer.addView(card);

                    if (i == 0 && selectedModel == null) {
                        onModelChipSelected(model);
                    }
                }
            
                // Re-select current model if still available
                if (selectedModel != null) {
                    selectChipForModel(selectedModel);
                }
            });
        });
    }
    
    private void onModelChipSelected(InsectModel model) {
        if (uiLocked) {
            Log.d(LOG_TAG, "Already predicting...");
            if (selectedModel != null) {
                selectChipForModel(selectedModel);
            }
            return;
        }
        if (modelDownloader != null && modelDownloader.isDownloading()) {
            showMessage("Please wait for downloads to complete.");
            if (selectedModel != null) {
                selectChipForModel(selectedModel); // Keep previous selection
            }
            return;
        }
        try {
            // Deselect chips in other groups
            deselectAllChipsExcept(model);
            selectedModel = model;
            
            runOnUiThread(() -> {
                if (selectedModelName != null) {
                    selectedModelName.setText("Identify a " + selectedModel.getDisplayName());
                }
                if (selectedModel.isLegacy()) {
                    identifyModelWarning.setText("This is a legacy model, may not perform up to the mark.");
                    identifyModelWarning.setVisibility(View.VISIBLE);
                } else if (selectedModel.isExperimental()) {
                    identifyModelWarning.setText("This is an experimental model, may not perform up to the mark.");
                    identifyModelWarning.setVisibility(View.VISIBLE);
                } else {
                    identifyModelWarning.setVisibility(View.GONE);
                }
            });

            if (photoUri != null) {
                runOnUiThread(() -> {
                    if (welcomeIcon != null) {
                        welcomeIcon.setVisibility(View.GONE);
                    }
                    outputText.setText("");
                });
                if (predictionCache.containsKey(selectedModel.getModelName())) {
                    showPredictionResponse(predictionCache.get(selectedModel.getModelName()));
                } else {
                    downloadModelAndRunPredictionAsync();
                }
            }
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception during model chip selection", ex);
        }
    }
    
    private void deselectAllChipsExcept(InsectModel selected) {
        // Handle Cards
        for (int i = 0; i < modelSelectorContainer.getChildCount(); i++) {
            View child = modelSelectorContainer.getChildAt(i);
            if (child instanceof com.google.android.material.card.MaterialCardView) {
                com.google.android.material.card.MaterialCardView card = (com.google.android.material.card.MaterialCardView) child;
                InsectModel cardModel = (InsectModel) card.getTag();
                if (cardModel != null) {
                    boolean isSelected = cardModel.getModelName().equals(selected.getModelName());
                    int defaultColor = com.google.android.material.color.MaterialColors.getColor(card, com.google.android.material.R.attr.colorOutlineVariant);
                    card.setStrokeColor(isSelected ? ContextCompat.getColor(this, R.color.primaryGreen) : defaultColor);
                    card.setStrokeWidth(isSelected ? 4 : 1);
                }
            }
        }
    }
    
    private void selectChipForModel(InsectModel model) {
        deselectAllChipsExcept(model);
    }
    
    private void setChipGroupsEnabled(boolean enabled) {
        for (int i = 0; i < modelSelectorContainer.getChildCount(); i++) {
            modelSelectorContainer.getChildAt(i).setEnabled(enabled);
        }
    }

    public void refreshManageModelsList() {
        List<InsectModel> currentModels = getVisibleModels();
        runOnUiThread(() -> {
            if (manageModelsAdapter != null) {
                manageModelsAdapter.updateModels(currentModels);
            }
        });
    }

    // Launcher for picking an image from the gallery
    private final ActivityResultLauncher<PickVisualMediaRequest> pickMedia =
            registerForActivityResult(new ActivityResultContracts.PickVisualMedia(), uri -> {
                if (uri != null) {
                    photoUri = uri;
                    launchImageCrop();
                } else {
                    Log.d("PhotoPicker", "No media selected");
                }
            });

    // Launcher for taking a photo with the camera
    private final ActivityResultLauncher<Intent> cameraLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                try {
                    if (result.getResultCode() == RESULT_OK && photoUri != null) {
                        launchImageCrop();
                    } else {
                        Toast.makeText(this, "Failed to capture image", Toast.LENGTH_SHORT).show();
                    }
                } catch (Exception ex) {
                    Log.e(LOG_TAG, "Exception in camera launcher activity result", ex);
                    throw ex;
                }
            });

    // Request camera permission
    private final ActivityResultLauncher<String> cameraPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                try {
                    if (isGranted) {
                        openCamera();
                    } else {
                        Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show();
                    }
                } catch (Exception ex) {
                    Log.e(LOG_TAG, "Exception in camera permission launcher activity result", ex);
                    throw ex;
                }
            });

    // Show dialog to choose Gallery or Camera
    private void showImagePickerDialog() {
        if(uiLocked) {
            return;
        }
        if (modelDownloader != null && modelDownloader.isDownloading()) {
            showMessage("Please wait for downloads to complete.");
            return;
        }
        try {
            String[] options = {"Gallery", "Camera"};
            AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setTitle("Select an Option");
            builder.setItems(options, (dialog, which) -> {
                if (which == 0) {
                    openGallery();
                } else {
                    checkCameraPermission();
                }
            });
            builder.show();
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception during show image picker dialog", ex);
            throw ex;
        }
    }

    // Open the Gallery
    private void openGallery() {
        try {
            pickMedia.launch(new PickVisualMediaRequest.Builder()
                    .setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly.INSTANCE)
                    .build());
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception during open gallery", ex);
            throw ex;
        }
    }

    // Check and request Camera permission
    private void checkCameraPermission() {
        try {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                openCamera();
            } else {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA);
            }
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception during check camera permission", ex);
            throw ex;
        }
    }

    // Open the Camera
    private void openCamera() {
        try {
            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            if (intent.resolveActivity(getPackageManager()) != null) {
                Log.d(LOG_TAG, "Camera app found!");
                // Stores the photo file
                File photoFile = createImageFile();
                if (photoFile != null) {
                    photoUri = FileProvider.getUriForFile(this, getApplicationContext().getPackageName() + ".provider", photoFile);
                    intent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
                    cameraLauncher.launch(intent);
                } else {
                    Toast.makeText(this, "Error creating file", Toast.LENGTH_SHORT).show();
                }
            } else {
                Log.d(LOG_TAG, "Camera app not found!");
                Toast.makeText(this, "Camera app not found", Toast.LENGTH_SHORT).show();
            }
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception during open camera", ex);
            throw ex;
        }
    }

    // Create a temporary file to store the captured image
    private File createImageFile() {
        try {
            String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
            File storageDir = getExternalFilesDir(null);
            return File.createTempFile(timeStamp, "_tmp.jpg", storageDir);
        } catch (IOException ex) {
            Log.e(LOG_TAG, "Exception during image creation", ex);
            Toast.makeText(this, "Failed to create image file", Toast.LENGTH_SHORT).show();
            return null;
        }
    }

    private void launchImageCrop() {
        try {
            String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
            File croppedFile = new File(getCacheDir(), timeStamp + "_cropped.jpg");
            if (croppedFile.exists()) {
                croppedFile.delete();
            }
            Uri croppedUri = Uri.fromFile(new File(getCacheDir(), timeStamp + "_cropped.jpg"));
            UCrop.of(photoUri, croppedUri)
                    .withAspectRatio(1, 1)
                    .withMaxResultSize(300, 300)
                    .start(this);
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception during image crop", ex);
            Toast.makeText(this, "Failed to crop image", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        try {
            if (requestCode == UCrop.REQUEST_CROP && resultCode == RESULT_OK) {
                photoUri = UCrop.getOutput(data);
                if (photoUri != null) {
                    predictionCache.clear();
                    imageView.setImageURI(photoUri);
                    downloadModelAndRunPredictionAsync();
                }
            }
            //TODO else auto-crop
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception during image crop", ex);
            Toast.makeText(this, "Failed to crop image", Toast.LENGTH_SHORT).show();
        }
    }

    public synchronized void lockUI() {
        uiLocked = true;
        runOnUiThread(() -> {
            setChipGroupsEnabled(false);
            buttonPickImage.setEnabled(false);
            btnDownloadAll.setEnabled(false);
            if (btnCancelDownload != null) btnCancelDownload.setEnabled(true);
            startIdentifyIconAnimation();
        });
    }

    public synchronized void unlockUI() {
        uiLocked = false;
        runOnUiThread(() -> {
            setChipGroupsEnabled(true);
            buttonPickImage.setEnabled(true);
            btnDownloadAll.setEnabled(true);
            stopIdentifyIconAnimation();
        });
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
        if (selectedModel != null) {
            downloadOrUpdateModel(selectedModel);
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
        
        List<InsectModel> modelsToDownload = getVisibleModels().stream()
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
            if (selectedModel != null && response != null) {
                predictionCache.put(selectedModel.getModelName(), response);
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
            stopDownloadIconAnimation();
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
