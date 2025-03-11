package com.rakeshmalik.insectid;

import static com.rakeshmalik.insectid.Constants.*;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.Html;
import android.text.Spanned;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.yalantis.ucrop.UCrop;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private ImageView imageView;
    private Button buttonPickImage;
    private TextView outputText;
    private Uri photoUri;
    private ExecutorService executorService;
    private Spinner modelTypeSpinner;
    private ModelType selectedModelType;
    private ModelDownloader modelDownloader;
    private PredictionManager predictionManager;

    private boolean predicting = false;

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

            this.executorService = Executors.newSingleThreadExecutor();

            MetadataManager metadataManager = new MetadataManager(this, outputText);
            this.modelDownloader = new ModelDownloader(this, outputText, metadataManager);
            this.predictionManager = new PredictionManager(this, metadataManager);
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception in MainActivity.onCreate()", ex);
            throw ex;
        }
    }

    private void createModelTypeSpinner() {
        try {
            // Convert enum values to a string array for the Spinner
            ModelType[] modelTypes = ModelType.values();
            String[] modelTypeNames = new String[modelTypes.length];
            for (int i = 0; i < modelTypes.length; i++) {
                modelTypeNames[i] = modelTypes[i].displayName;
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
                    if(predicting) {
                        Log.d(LOG_TAG, "Already predicting...");
                        modelTypeSpinner.setSelection(previousSelection);
                        return;
                    }
                    try {
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
    private final ActivityResultLauncher<Intent> galleryLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                try {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        photoUri = result.getData().getData();
                        if (photoUri != null) {
                            launchImageCrop();
                        }
                    }
                } catch (Exception ex) {
                    Log.e(LOG_TAG, "Exception during gallery launcher", ex);
                    throw ex;
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
        if(predicting) {
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
            Log.e(LOG_TAG, "Exception during sho image picker dialog", ex);
            throw ex;
        }
    }

    // Open the Gallery
    private void openGallery() {
        try {
            Intent intent = new Intent(Intent.ACTION_PICK);
            intent.setType("image/*");
            galleryLauncher.launch(intent);
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

    private void downloadModelAndRunPredictionAsync() {
        executorService.submit(new PredictRunnable());
    }

    private void blockPrediction() {
        predicting = true;
        runOnUiThread(() -> {
            modelTypeSpinner.setEnabled(false);
            buttonPickImage.setEnabled(false);
        });
    }

    private void unblockPrediction() {
        predicting = false;
        runOnUiThread(() -> {
            modelTypeSpinner.setEnabled(true);
            buttonPickImage.setEnabled(true);
        });
    }

    public Drawable predictedImageRenderer(String source) {
        String[] urls = source.split(",");
        int maxColumns = 3, gap = 10;
        int size = (outputText.getWidth() - gap * (maxColumns - 1)) / maxColumns;
        int columns = Math.min(urls.length, maxColumns);
        int rows = (int) Math.ceil((double) urls.length / maxColumns);
        int gridWidth = size * columns + gap * (columns - 1);
        int gridHeight = size * rows + gap * (rows - 1);
        Bitmap bitmap = Bitmap.createBitmap(gridWidth, gridHeight, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        int left = 0, top = 0;
        for (int i = 0; i < urls.length; i++) {
            try (InputStream is = (InputStream) new URL(urls[i]).getContent();) {
                Bitmap img = BitmapFactory.decodeStream(is);
                img = Utils.topBottomEdgeCrop(img, 0.12f);
                img = Utils.centerSquareCrop(img);
                img = Bitmap.createScaledBitmap(img, size, size, true);
                canvas.drawBitmap(img, left, top, null);
                if(i % maxColumns == maxColumns - 1) {
                    top += size + gap;
                    left = 0;
                } else {
                    left += size + gap;
                }
            } catch (Exception ex1) {
                Log.e(LOG_TAG, "Exception loading image " + urls[i] + " as html in textview", ex1);
            }
        }
        Drawable drawable = new BitmapDrawable(getResources(), bitmap);
        drawable.setBounds(0, 0, gridWidth, gridHeight);
        return drawable;
    }

    class PredictRunnable implements Runnable {
        private void runPrediction() {
            try {
                runOnUiThread(() -> outputText.setText(R.string.predicting));
                final ModelType modelType = selectedModelType;
                String predictions = predictionManager.predict(selectedModelType, photoUri);
                if (modelType == selectedModelType) {
                    // set html with alt text while loading images
                    String imageAltHtml = "<font color='#777777'>(Loading images...)</font><br/><br/>";
                    Spanned simpleHtml = Html.fromHtml(predictions.replaceAll("<img [^>]+/>", imageAltHtml), Html.FROM_HTML_MODE_COMPACT, null, null);
                    runOnUiThread(() -> outputText.setText(simpleHtml));
                    // render html with images
                    Spanned html = Html.fromHtml(predictions, Html.FROM_HTML_MODE_COMPACT, MainActivity.this::predictedImageRenderer, null);
                    runOnUiThread(() -> outputText.setText(html));
                }
            } catch(Exception ex) {
                Log.e(LOG_TAG, "Exception during prediction", ex);
            } finally {
                unblockPrediction();
            }
        }
        @Override
        public void run() {
            blockPrediction();
            try {
                if (modelDownloader.isModelDownloadOrUpdateRequired(selectedModelType)) {
                    modelDownloader.downloadModel(selectedModelType, this::runPrediction, MainActivity.this::unblockPrediction);
                } else {
                    runPrediction();
                }
            } catch(Exception ex) {
                unblockPrediction();
            }
        }
    }

}
