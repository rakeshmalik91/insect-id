package com.rakeshmalik.insectid;

import static com.rakeshmalik.insectid.Constants.PREF;
import static com.rakeshmalik.insectid.Constants.PREF_ASSET_TEMP_PATH;

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

import android.content.SharedPreferences;
import android.util.Log;

import com.google.gson.Gson;

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
            Log.d(Constants.LOG_TAG, "Exception loading class details", ex);
            return defaultValue;
        }
    }
}
