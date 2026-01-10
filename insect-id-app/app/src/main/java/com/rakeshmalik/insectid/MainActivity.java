package com.rakeshmalik.insectid;

import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.getkeepsafe.relinker.ReLinker;
import com.rakeshmalik.insectid.filemanager.MetadataManager;
import com.rakeshmalik.insectid.filemanager.ModelDownloader;
import com.rakeshmalik.insectid.filemanager.ModelLoader;
import com.rakeshmalik.insectid.enums.ModelType;
import com.rakeshmalik.insectid.prediction.PredictionRunnable;
import com.rakeshmalik.insectid.prediction.PredictionManager;
import com.yalantis.ucrop.UCrop;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
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

public class MainActivity extends AppCompatActivity {

    private ImageView imageView;
    private Button buttonPickImage;
    private TextView outputText;
    private Uri photoUri;
    private Spinner modelTypeSpinner;
    private ModelType selectedModelType;
    private ModelLoader modelLoader;
    private ModelDownloader modelDownloader;
    private PredictionManager predictionManager;
    private ImageButton buttonUpdateModel;

    private final ExecutorService executorService = Executors.newSingleThreadExecutor();
    private final ArrayBlockingQueue<Future<?>> runningTasks = new ArrayBlockingQueue<>(10);

    private boolean uiLocked = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        try {
            super.onCreate(savedInstanceState);
            setContentView(R.layout.activity_main);

            this.buttonPickImage = findViewById(R.id.buttonPickImage);
            this.imageView = findViewById(R.id.imageView);
            this.buttonPickImage.setOnClickListener(v -> showImagePickerDialog());
            this.outputText = findViewById(R.id.outputText);
            this.modelTypeSpinner = findViewById(R.id.modelTypeSpinner);
            createModelTypeSpinner();

            MetadataManager metadataManager = new MetadataManager(this, outputText);
            this.modelLoader = new ModelLoader(this);
            this.modelDownloader = new ModelDownloader(this, outputText, metadataManager);
            this.predictionManager = new PredictionManager(this, metadataManager, modelLoader);

            this.buttonUpdateModel = findViewById(R.id.buttonUpdateModel);
            this.buttonUpdateModel.setOnClickListener(v -> showDownloadOrUpdateModelDialog());

            //System.loadLibrary("opencv_java4");
            ReLinker.loadLibrary(this, "opencv_java4");

            new Thread(metadataManager::getMetadata).start();
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception in MainActivity.onCreate()", ex);
            throw ex;
        }
    }

    public ModelType getSelectedModelType() {
        return selectedModelType;
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
            // Convert enum values to a string array for the Spinner
            ModelType[] modelTypes = ModelType.values();
            String[] modelTypeNames = new String[modelTypes.length];
            for (int i = 0; i < modelTypes.length; i++) {
                modelTypeNames[i] = modelTypes[i].getModelDisplayName();
            }

            // Create and set adapter for the Spinner
            ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, modelTypeNames);
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
            modelTypeSpinner.setAdapter(adapter);

            // Set listener for Spinner item selection
            modelTypeSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
                private int previousSelection = 0;
                @Override
                public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                    if(uiLocked) {
                        Log.d(LOG_TAG, "Already predicting...");
                        modelTypeSpinner.setSelection(previousSelection);
                        return;
                    }
                    try {
                        runOnUiThread(() -> outputText.setText(""));
                        selectedModelType = modelTypes[position];
                        previousSelection = position;
                        if(photoUri != null) {
                            downloadModelAndRunPredictionAsync();
                        }
                    } catch (Exception ex) {
                        Log.e(LOG_TAG, "Exception during model type spinner item selection", ex);
                        throw ex;
                    }
                }
                @Override
                public void onNothingSelected(AdapterView<?> parent) {
                    Log.d(LOG_TAG, "nothing selected on model type spinner");
                    selectedModelType = null;
                }
            });
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception during spinner creation", ex);
            throw ex;
        }
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
            modelTypeSpinner.setEnabled(false);
            buttonPickImage.setEnabled(false);
            buttonUpdateModel.setEnabled(false);
        });
    }

    public synchronized void unlockUI() {
        uiLocked = false;
        runOnUiThread(() -> {
            modelTypeSpinner.setEnabled(true);
            buttonPickImage.setEnabled(true);
            buttonUpdateModel.setEnabled(true);
        });
    }

    private void downloadModelAndRunPredictionAsync() {
        if(!runningTasks.isEmpty()) {
            Log.d(LOG_TAG, "Previous tasks still running. Going to try killing them.");
            runOnUiThread(() -> outputText.setText(R.string.please_wait));
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

    @SuppressLint("DefaultLocale")
    private void showDownloadOrUpdateModelDialog() {
        if(uiLocked) {
            return;
        }
        try {
            long modelSize = modelDownloader.getModelDownloadSizeInMB(selectedModelType);
            long totalNonLegacyModelSize = modelDownloader.getTotalModelDownloadSizeInMB(true);
            long totalModelSize = modelDownloader.getTotalModelDownloadSizeInMB(false);
            String[] options = {
                    selectedModelType.getModelDisplayName() + " " + (modelSize > 0 ? String.format("(%d MB)", modelSize) : "(Up to date)"),
                    "All Non-legacy Models " + (totalNonLegacyModelSize > 0 ? String.format("(%d MB)", totalNonLegacyModelSize) : "(Up to date)"),
                    "All Models " + (totalModelSize > 0 ? String.format("(%d MB)", totalModelSize) : "(Up to date)")
            };
            AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setTitle("Download/Update");
            builder.setItems(options, (dialog, which) -> {
                switch (which) {
                    case 0: downloadOrUpdateModel(); break;
                    case 1: downloadOrUpdateAllModels(true); break;
                    default: downloadOrUpdateAllModels(false); break;
                }
            });
            builder.show();
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception during show image picker dialog", ex);
            throw ex;
        }
    }

    private void downloadOrUpdateModel() {
        lockUI();
        imageView.setImageURI(null);
        photoUri = null;
        executorService.submit(() -> modelDownloader.downloadModel(selectedModelType, this::unlockUI, this::unlockUI, true, 1, 1));
    }

    private void downloadOrUpdateAllModels(boolean onlyNonLegacy) {
        lockUI();
        imageView.setImageURI(null);
        photoUri = null;
        List<ModelType> modelsToDownload = Arrays.stream(ModelType.values())
                .filter(m -> (!onlyNonLegacy || !m.legacy) && modelDownloader.isModelDownloadOrUpdateRequired(m))
                .sorted(Collections.reverseOrder())
                .collect(Collectors.toList());
        int modelDownloadSeq = modelsToDownload.size();
        Runnable runnable = () -> {
            runOnUiThread(() -> outputText.setText("All " + ModelType.values().length + " models are now up to date"));
            unlockUI();
        };
        for(ModelType modelType : modelsToDownload) {
            Runnable onSuccess = runnable;
            final int modelDownloadSeqFinal = modelDownloadSeq;
            runnable = () -> modelDownloader.downloadModel(modelType, onSuccess, this::unlockUI, true, modelDownloadSeqFinal, modelsToDownload.size());
            modelDownloadSeq--;
        }
        executorService.submit(runnable);
    }

}
