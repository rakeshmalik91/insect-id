package com.rakeshmalik.insectid.utils;

import static android.app.Activity.RESULT_OK;
import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.yalantis.ucrop.UCrop;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class PhotoPickerHelper {

    private final AppCompatActivity activity;
    private final PhotoPickerCallback callback;
    private Uri photoUri;

    private final ActivityResultLauncher<PickVisualMediaRequest> pickMedia;
    private final ActivityResultLauncher<Intent> cameraLauncher;
    private final ActivityResultLauncher<String> cameraPermissionLauncher;

    public interface PhotoPickerCallback {
        void onPhotoCropped(Uri uri);
        void onShowMessage(String message);
        boolean isUiLocked();
    }

    public PhotoPickerHelper(AppCompatActivity activity, PhotoPickerCallback callback) {
        this.activity = activity;
        this.callback = callback;

        // Launcher for picking an image from the gallery
        pickMedia = activity.registerForActivityResult(new ActivityResultContracts.PickVisualMedia(), uri -> {
            if (uri != null) {
                photoUri = uri;
                launchImageCrop();
            } else {
                Log.d("PhotoPicker", "No media selected");
            }
        });

        // Launcher for taking a photo with the camera
        cameraLauncher = activity.registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
            try {
                if (result.getResultCode() == RESULT_OK && photoUri != null) {
                    launchImageCrop();
                } else {
                    Toast.makeText(activity, "Failed to capture image", Toast.LENGTH_SHORT).show();
                }
            } catch (Exception ex) {
                Log.e(LOG_TAG, "Exception in camera launcher activity result", ex);
                throw ex;
            }
        });

        // Request camera permission
        cameraPermissionLauncher = activity.registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
            try {
                if (isGranted) {
                    openCamera();
                } else {
                    Toast.makeText(activity, "Camera permission is required", Toast.LENGTH_SHORT).show();
                }
            } catch (Exception ex) {
                Log.e(LOG_TAG, "Exception in camera permission launcher activity result", ex);
                throw ex;
            }
        });
    }

    public void showImagePickerDialog() {
        if (callback.isUiLocked()) {
            return;
        }
        try {
            String[] options = {"Gallery", "Camera"};
            AlertDialog.Builder builder = new AlertDialog.Builder(activity);
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

    private void checkCameraPermission() {
        try {
            if (ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                openCamera();
            } else {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA);
            }
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception during check camera permission", ex);
            throw ex;
        }
    }

    private void openCamera() {
        try {
            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            if (intent.resolveActivity(activity.getPackageManager()) != null) {
                Log.d(LOG_TAG, "Camera app found!");
                File photoFile = createImageFile();
                if (photoFile != null) {
                    photoUri = FileProvider.getUriForFile(activity, activity.getApplicationContext().getPackageName() + ".provider", photoFile);
                    intent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
                    cameraLauncher.launch(intent);
                } else {
                    Toast.makeText(activity, "Error creating file", Toast.LENGTH_SHORT).show();
                }
            } else {
                Log.d(LOG_TAG, "Camera app not found!");
                Toast.makeText(activity, "Camera app not found", Toast.LENGTH_SHORT).show();
            }
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception during open camera", ex);
            throw ex;
        }
    }

    private File createImageFile() {
        try {
            String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
            File storageDir = activity.getExternalFilesDir(null);
            return File.createTempFile(timeStamp, "_tmp.jpg", storageDir);
        } catch (IOException ex) {
            Log.e(LOG_TAG, "Exception during image creation", ex);
            Toast.makeText(activity, "Failed to create image file", Toast.LENGTH_SHORT).show();
            return null;
        }
    }

    private void launchImageCrop() {
        try {
            String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
            File croppedFile = new File(activity.getCacheDir(), timeStamp + "_cropped.jpg");
            if (croppedFile.exists()) {
                croppedFile.delete();
            }
            Uri croppedUri = Uri.fromFile(croppedFile);
            UCrop.of(photoUri, croppedUri)
                    .withAspectRatio(1, 1)
                    .withMaxResultSize(300, 300)
                    .start(activity);
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception during image crop", ex);
            Toast.makeText(activity, "Failed to crop image", Toast.LENGTH_SHORT).show();
        }
    }

    public void handleActivityResult(int requestCode, int resultCode, Intent data) {
        try {
            if (requestCode == UCrop.REQUEST_CROP && resultCode == RESULT_OK) {
                Uri resultUri = UCrop.getOutput(data);
                if (resultUri != null) {
                    photoUri = resultUri;
                    callback.onPhotoCropped(photoUri);
                }
            }
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception during image crop handle", ex);
            Toast.makeText(activity, "Failed to crop image", Toast.LENGTH_SHORT).show();
        }
    }
    
    public Uri getPhotoUri() {
        return photoUri;
    }
}
