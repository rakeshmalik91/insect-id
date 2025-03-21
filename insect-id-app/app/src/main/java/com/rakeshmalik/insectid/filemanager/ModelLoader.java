package com.rakeshmalik.insectid.filemanager;

import static com.rakeshmalik.insectid.constants.Constants.IMAGES_ARCHIVE_FILE_NAME_FMT;
import static com.rakeshmalik.insectid.constants.Constants.MAX_IMAGES_IN_PREDICTION;
import static com.rakeshmalik.insectid.constants.Constants.PREF;
import static com.rakeshmalik.insectid.constants.Constants.PREF_ASSET_TEMP_PATH;

import android.content.Context;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.zip.ZipFile;

import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.util.Log;

import com.google.gson.Gson;
import com.rakeshmalik.insectid.constants.Constants;

public class ModelLoader {

    private final Map<String, List<String>> classLabelsCache = new HashMap<>();
    private final Map<String, Map<String, Map<String, Object>>> classDetailsCache = new HashMap<>();
    private final SharedPreferences prefs;

    public ModelLoader(Context context) {
        this.prefs = context.getSharedPreferences(PREF, Context.MODE_PRIVATE);
    }

    public String loadFromCache(Context context, String fileName) {
        File file = new File(context.getCacheDir(), fileName);
        return file.getAbsolutePath();
    }

    public String loadFromAsset(Context context, String assetName) {
        String prefKey = PREF_ASSET_TEMP_PATH + "::" + assetName;
        if(prefs.contains(prefKey)) {
            File file = new File(prefs.getString(prefKey, ""));
            if(file.exists()) {
                return file.getAbsolutePath();
            }
        }
        try(InputStream is = context.getAssets().open(assetName);) {
            File tempFile = File.createTempFile(assetName, "tmp", context.getCacheDir());
            tempFile.deleteOnExit();
            try(FileOutputStream outputStream = new FileOutputStream(tempFile);) {
                byte[] buffer = new byte[4 * 1024];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
            }
            prefs.edit().putString(prefKey, tempFile.getAbsolutePath()).apply();
            return tempFile.getAbsolutePath();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public List<String> getClassLabels(Context context, String fileName) {
        if(classLabelsCache.containsKey(fileName)) {
            return classLabelsCache.get(fileName);
        }
        List<String> classLabels = loadJsonFromFile(context, fileName, List.of());
        classLabelsCache.put(fileName, classLabels);
        return classLabels;
    }

    public Map<String, Map<String, Object>> getClassDetails(Context context, String fileName) {
        if(classDetailsCache.containsKey(fileName)) {
            return classDetailsCache.get(fileName);
        }
        Map<String, Map<String, Object>> classDetails = loadJsonFromFile(context, fileName, Map.of());
        classDetailsCache.put(fileName, classDetails);
        return classDetails;
    }

    private <T> T loadJsonFromFile(Context context, String fileName, T defaultValue) {
        File file = new File(context.getCacheDir(), fileName);
        try(InputStream is = new FileInputStream(file)) {
            byte[] buffer = new byte[is.available()];
            is.read(buffer);
            is.close();
            String json = new String(buffer, StandardCharsets.UTF_8);
            Gson gson = new Gson();
            if(defaultValue instanceof Map) {
                return (T) gson.fromJson(json, Map.class);
            } else {
                return (T) gson.fromJson(json, List.class);
            }
        } catch (IOException ex) {
            Log.e(Constants.LOG_TAG, "Exception loading class details", ex);
            return defaultValue;
        }
    }

    public List<Bitmap> getImagesFromZip(Context context, String modelName, String className) {
        final String zipFileName = String.format(IMAGES_ARCHIVE_FILE_NAME_FMT, modelName);
        File file = new File(context.getCacheDir(), zipFileName);
        try (ZipFile zipFile = new ZipFile(file.getAbsolutePath())) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                return zipFile.stream()
                        .filter(f -> f.getName().startsWith(className + "/") && !f.isDirectory())
                        .map(f -> {
                            try {
                                InputStream inputStream = zipFile.getInputStream(f);
                                Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
//                                Log.d(Constants.LOG_TAG, "Loaded bitmap " + bitmap + " from path " + f.getName() + " in archive " + zipFileName);
                                return bitmap;
                            } catch (IOException ex) {
                                Log.e(Constants.LOG_TAG, "Exception creating bitmap from file " + f.getName() + " in archive " + zipFileName, ex);
                                return null;
                            }
                        })
                        .filter(Objects::nonNull)
                        .limit(MAX_IMAGES_IN_PREDICTION)
                        .collect(Collectors.toList());
            }
            Log.d(Constants.LOG_TAG, "No images found for model" + modelName + ", class " + className);
        } catch (Exception ex) {
            Log.e(Constants.LOG_TAG, "Exception processing images archive for model" + modelName + ", class " + className, ex);
        }
        return List.of();
    }

}
