package com.rakeshmalik.insectid.filemanager;

import static com.rakeshmalik.insectid.constants.Constants.IMAGES_ARCHIVE_FILE_NAME_FMT;
import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;
import static com.rakeshmalik.insectid.constants.Constants.MAX_IMAGES_IN_PREDICTION;
import static com.rakeshmalik.insectid.constants.Constants.PREF;

import android.content.Context;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.io.File;
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

    public String loadFile(Context context, String fileName) {
        Log.d(LOG_TAG, "inside ModelLoader.loadFromCache(context,"  + fileName + ")");
        File file = new File(context.getFilesDir(), fileName);
        String path = file.getAbsolutePath();
        Log.d(LOG_TAG, "absolute path: " + path + ", exists: " + file.exists() + ", size: " + file.length());
        return path;
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
        File file = new File(context.getFilesDir(), fileName);
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
        File file = new File(context.getFilesDir(), zipFileName);
        try (ZipFile zipFile = new ZipFile(file.getAbsolutePath())) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                return zipFile.stream()
                        .filter(f -> f.getName().startsWith(className + "/") && !f.isDirectory())
                        .map(f -> {
                            try {
                                InputStream inputStream = zipFile.getInputStream(f);
                                return BitmapFactory.decodeStream(inputStream);
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
