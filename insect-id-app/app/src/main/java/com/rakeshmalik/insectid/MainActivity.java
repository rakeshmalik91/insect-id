package com.rakeshmalik.insectid;

import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;
import com.rakeshmalik.insectid.constants.Constants;

import android.Manifest;
import android.annotation.SuppressLint;
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
import android.widget.ImageButton;
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
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.recyclerview.widget.RecyclerView;
import com.getkeepsafe.relinker.ReLinker;

import com.rakeshmalik.insectid.filemanager.MetadataManager;
import com.rakeshmalik.insectid.filemanager.ModelDownloader;
import com.rakeshmalik.insectid.filemanager.DownloadItem;
import com.rakeshmalik.insectid.filemanager.ModelLoader;
import com.rakeshmalik.insectid.pojo.InsectModel;
import com.rakeshmalik.insectid.prediction.PredictionResult;
import com.rakeshmalik.insectid.prediction.PredictionRunnable;
import com.rakeshmalik.insectid.prediction.PredictionManager;
import com.rakeshmalik.insectid.ui.PredictionAdapter;
import com.yalantis.ucrop.UCrop;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;
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
    private RecyclerView manageModelsRecyclerView;
    private ManageModelsAdapter manageModelsAdapter;
    private Uri photoUri;
    private com.google.android.material.chip.ChipGroup chipGroupNormal;
    private com.google.android.material.chip.ChipGroup chipGroupLegacy;
    private com.google.android.material.chip.ChipGroup chipGroupPrototype;
    private View legacySectionContainer;
    private View prototypeSectionContainer;
    private ImageView legacyExpandIcon;
    private ImageView prototypeExpandIcon;
    private TextView identifyModelWarning;
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
    private MaterialButton btnDownloadNonLegacy;
    
    private com.google.android.material.switchmaterial.SwitchMaterial switchShowLegacy;
    private com.google.android.material.switchmaterial.SwitchMaterial switchShowPrototype;
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
            this.chipGroupNormal = findViewById(R.id.chipGroupNormal);
            this.chipGroupLegacy = findViewById(R.id.chipGroupLegacy);
            this.chipGroupPrototype = findViewById(R.id.chipGroupPrototype);
            this.legacySectionContainer = findViewById(R.id.legacySectionContainer);
            this.prototypeSectionContainer = findViewById(R.id.prototypeSectionContainer);
            this.legacyExpandIcon = findViewById(R.id.legacyExpandIcon);
            this.prototypeExpandIcon = findViewById(R.id.prototypeExpandIcon);
            this.identifyModelWarning = findViewById(R.id.identifyModelWarning);
            
            // Set up collapsible legacy section
            findViewById(R.id.legacyHeader).setOnClickListener(v -> {
                boolean isVisible = chipGroupLegacy.getVisibility() == View.VISIBLE;
                chipGroupLegacy.setVisibility(isVisible ? View.GONE : View.VISIBLE);
                legacyExpandIcon.setImageResource(isVisible ? R.drawable.ic_expand_more : R.drawable.ic_expand_less);
            });
            
            // Set up collapsible prototype section
            findViewById(R.id.prototypeHeader).setOnClickListener(v -> {
                boolean isVisible = chipGroupPrototype.getVisibility() == View.VISIBLE;
                chipGroupPrototype.setVisibility(isVisible ? View.GONE : View.VISIBLE);
                prototypeExpandIcon.setImageResource(isVisible ? R.drawable.ic_expand_more : R.drawable.ic_expand_less);
            });
            
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
            this.btnDownloadAll.setOnClickListener(v -> downloadOrUpdateAllModels(false));
            
            this.btnDownloadNonLegacy = findViewById(R.id.btnDownloadNonLegacy);
            this.btnDownloadNonLegacy.setOnClickListener(v -> downloadOrUpdateAllModels(true));
            
            this.switchShowLegacy = findViewById(R.id.switchShowLegacy);
            this.switchShowPrototype = findViewById(R.id.switchShowPrototype);
            this.switchApplyBlur = findViewById(R.id.switchApplyBlur);
            
            SharedPreferences prefs = getSharedPreferences(PREF, Context.MODE_PRIVATE);
            switchShowLegacy.setChecked(prefs.getBoolean("show_legacy_models", true));
            switchShowPrototype.setChecked(prefs.getBoolean("show_prototype_models", false));
            switchApplyBlur.setChecked(prefs.getBoolean("apply_blur_screen_captures", true));
            
            android.widget.CompoundButton.OnCheckedChangeListener switchListener = (buttonView, isChecked) -> {
                SharedPreferences.Editor editor = prefs.edit();
                if (buttonView.getId() == R.id.switchShowLegacy) {
                    editor.putBoolean("show_legacy_models", isChecked);
                } else if (buttonView.getId() == R.id.switchShowPrototype) {
                    editor.putBoolean("show_prototype_models", isChecked);
                } else if (buttonView.getId() == R.id.switchApplyBlur) {
                    editor.putBoolean("apply_blur_screen_captures", isChecked);
                }
                editor.apply();
                
                // Refresh UI
                populateModelSpinner();
                refreshManageModelsList();
            };
            
            switchShowLegacy.setOnCheckedChangeListener(switchListener);
            switchShowPrototype.setOnCheckedChangeListener(switchListener);
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
        });
    }
    
    private List<InsectModel> getVisibleModels() {
        SharedPreferences prefs = getSharedPreferences(PREF, Context.MODE_PRIVATE);
        boolean showLegacy = prefs.getBoolean("show_legacy_models", true);
        boolean showPrototype = prefs.getBoolean("show_prototype_models", false);
        
        return metadataManager.getAvailableModels().stream()
                .filter(m -> {
                    if (m.isLegacy() && !showLegacy) return false;
                    if (m.isPrototype() && !showPrototype) return false;
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
            chipGroupNormal.removeAllViews();
            chipGroupLegacy.removeAllViews();
            chipGroupPrototype.removeAllViews();
            
            List<InsectModel> normalModels = new ArrayList<>();
            List<InsectModel> legacyModels = new ArrayList<>();
            List<InsectModel> prototypeModels = new ArrayList<>();
            
            for (InsectModel model : availableModels) {
                if (model.isLegacy()) {
                    legacyModels.add(model);
                } else if (model.isPrototype()) {
                    prototypeModels.add(model);
                } else {
                    normalModels.add(model);
                }
            }
            
            // Add normal model chips
            for (int i = 0; i < normalModels.size(); i++) {
                InsectModel model = normalModels.get(i);
                com.google.android.material.chip.Chip chip = new com.google.android.material.chip.Chip(this);
                chip.setText(model.getDisplayName());
                chip.setCheckable(true);
                chip.setTag(model);
                chip.setOnClickListener(v -> onModelChipSelected(model));
                chipGroupNormal.addView(chip);
                if (i == 0 && selectedModel == null) {
                    chip.setChecked(true);
                    onModelChipSelected(model);
                }
            }
            
            // Show/hide legacy section
            legacySectionContainer.setVisibility(legacyModels.isEmpty() ? View.GONE : View.VISIBLE);
            for (InsectModel model : legacyModels) {
                com.google.android.material.chip.Chip chip = new com.google.android.material.chip.Chip(this);
                chip.setText(model.getDisplayName());
                chip.setCheckable(true);
                chip.setChipIconResource(R.drawable.ic_legacy);
                chip.setChipIconVisible(true);
                chip.setTag(model);
                chip.setOnClickListener(v -> onModelChipSelected(model));
                chipGroupLegacy.addView(chip);
            }
            
            // Show/hide prototype section
            prototypeSectionContainer.setVisibility(prototypeModels.isEmpty() ? View.GONE : View.VISIBLE);
            for (InsectModel model : prototypeModels) {
                com.google.android.material.chip.Chip chip = new com.google.android.material.chip.Chip(this);
                chip.setText(model.getDisplayName());
                chip.setCheckable(true);
                chip.setChipIconResource(R.drawable.ic_prototype);
                chip.setChipIconVisible(true);
                chip.setTag(model);
                chip.setOnClickListener(v -> onModelChipSelected(model));
                chipGroupPrototype.addView(chip);
            }
            
            // Re-select current model if still available
            if (selectedModel != null) {
                selectChipForModel(selectedModel);
            }
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
        try {
            // Deselect chips in other groups
            deselectAllChipsExcept(model);
            selectedModel = model;
            
            runOnUiThread(() -> {
                if (selectedModel.isLegacy()) {
                    identifyModelWarning.setText("This is a legacy model, may not contain up to the mark.");
                    identifyModelWarning.setVisibility(View.VISIBLE);
                } else if (selectedModel.isPrototype()) {
                    identifyModelWarning.setText("This is an experimental model, may not contain up to the mark.");
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
                downloadModelAndRunPredictionAsync();
            }
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception during model chip selection", ex);
        }
    }
    
    private void deselectAllChipsExcept(InsectModel selected) {
        com.google.android.material.chip.ChipGroup[] groups = {chipGroupNormal, chipGroupLegacy, chipGroupPrototype};
        for (com.google.android.material.chip.ChipGroup group : groups) {
            for (int i = 0; i < group.getChildCount(); i++) {
                View child = group.getChildAt(i);
                if (child instanceof com.google.android.material.chip.Chip) {
                    com.google.android.material.chip.Chip chip = (com.google.android.material.chip.Chip) child;
                    InsectModel chipModel = (InsectModel) chip.getTag();
                    chip.setChecked(chipModel != null && chipModel.getModelName().equals(selected.getModelName()));
                }
            }
        }
    }
    
    private void selectChipForModel(InsectModel model) {
        deselectAllChipsExcept(model);
    }
    
    private void setChipGroupsEnabled(boolean enabled) {
        com.google.android.material.chip.ChipGroup[] groups = {chipGroupNormal, chipGroupLegacy, chipGroupPrototype};
        for (com.google.android.material.chip.ChipGroup group : groups) {
            for (int i = 0; i < group.getChildCount(); i++) {
                group.getChildAt(i).setEnabled(enabled);
            }
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
            btnDownloadNonLegacy.setEnabled(false);
        });
    }

    public synchronized void unlockUI() {
        uiLocked = false;
        runOnUiThread(() -> {
            setChipGroupsEnabled(true);
            buttonPickImage.setEnabled(true);
            btnDownloadAll.setEnabled(true);
            btnDownloadNonLegacy.setEnabled(true);
        });
    }

    private void downloadModelAndRunPredictionAsync() {
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
        lockUI();
        imageView.setImageURI(null);
        photoUri = null;
        
        downloadProgressContainer.setVisibility(View.VISIBLE);
        
        List<DownloadItem> plan = modelDownloader.generateDownloadPlan(Collections.singletonList(model), true);
        initDownloadList(plan);
        
        Runnable onSuccess = () -> {
            runOnUiThread(() -> {
                hideDownloadProgress();
                refreshManageModelsList();
                showMessage(model.getDisplayName() + " model downloaded successfully");
            });
            unlockUI();
        };
        executorService.submit(() -> modelDownloader.downloadModel(model, onSuccess, this::unlockUI, true, 1, 1));
    }

    private void downloadOrUpdateAllModels(boolean onlyCoreModels) {
        lockUI();
        imageView.setImageURI(null);
        photoUri = null;
        
        downloadProgressContainer.setVisibility(View.VISIBLE);
        
        List<InsectModel> modelsToDownload = getVisibleModels().stream()
                .filter(m -> (!onlyCoreModels || (!m.isLegacy() && !m.isPrototype())) && modelDownloader.isModelDownloadOrUpdateRequired(m))
                .sorted()
                .collect(Collectors.toList());
        
        List<DownloadItem> plan = modelDownloader.generateDownloadPlan(modelsToDownload, true);
        initDownloadList(plan);

        int modelDownloadSeq = modelsToDownload.size();
        Runnable runnable = () -> {
            int availableCount = getVisibleModels().size();
            runOnUiThread(() -> {
                hideDownloadProgress();
                refreshManageModelsList();
                showMessage("All " + availableCount + " models are now up to date");
            });
            unlockUI();
        };
        for(int i = modelsToDownload.size() - 1; i >= 0; i--) {
            InsectModel model = modelsToDownload.get(i);
            Runnable onSuccess = runnable;
            final int modelDownloadSeqFinal = modelDownloadSeq;
            runnable = () -> modelDownloader.downloadModel(model, onSuccess, this::unlockUI, true, modelDownloadSeqFinal, modelsToDownload.size());
            modelDownloadSeq--;
        }
        executorService.submit(runnable);
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
    public void showPredictions(List<PredictionResult> predictions) {
        runOnUiThread(() -> {
            outputTextContainer.setVisibility(View.GONE);
            predictionsRecyclerView.setVisibility(View.VISIBLE);
            
            if (predictionAdapter == null) {
                predictionAdapter = new PredictionAdapter(this, predictions);
                predictionsRecyclerView.setAdapter(predictionAdapter);
            } else {
                predictionAdapter.updateData(predictions);
            }
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
        });
    }

}
