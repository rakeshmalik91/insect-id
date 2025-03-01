package com.rakeshmalik.insectid;

import android.content.Context;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import android.util.Log;

import com.google.gson.Gson;

public class ModelLoader {

    private final Map<String, List<String>> classLabelsCache = new HashMap<>();
    private final Map<String, Map<String, Map<String, String>>> classDetailsCache = new HashMap<>();

    public String load(Context context, String fileName) {
        File file = new File(context.getCacheDir(), fileName);
        return file.getAbsolutePath();
    }

    public List<String> getClassLabels(Context context, String fileName) {
        if(classLabelsCache.containsKey(fileName)) {
            return classLabelsCache.get(fileName);
        }
        List<String> classLabels = loadJsonFromFile(context, fileName, List.of());
        classLabelsCache.put(fileName, classLabels);
        return classLabels;
    }

    public Map<String, Map<String, String>> getClassDetails(Context context, String fileName) {
        if(classDetailsCache.containsKey(fileName)) {
            return classDetailsCache.get(fileName);
        }
        Map<String, Map<String, String>> classDetails = loadJsonFromFile(context, fileName, Map.of());
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
