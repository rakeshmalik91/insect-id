package com.rakeshmalik.insectid.filemanager;

import static com.rakeshmalik.insectid.constants.Constants.*;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.TextView;

import com.rakeshmalik.insectid.R;
import com.rakeshmalik.insectid.constants.Constants;
import com.rakeshmalik.insectid.utils.FileUtils;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

import com.rakeshmalik.insectid.pojo.InsectModel;

public class MetadataManager {

    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private JSONObject metadata;
    private final TextView outputText;
    private final Context context;
    private final SharedPreferences prefs;

    public MetadataManager(Context context, TextView outputText) {
        this.context = context;
        this.outputText = outputText;
        this.prefs = context.getSharedPreferences(PREF, Context.MODE_PRIVATE);
    }

    public JSONObject getMetadata(String modelName) {
        JSONObject meta = getMetadata(false);
        if (meta == null) return new JSONObject();
        JSONObject modelMeta = meta.optJSONObject(modelName);
        return modelMeta != null ? modelMeta : new JSONObject();
    }

    public JSONObject getMetadata() {
        return getMetadata(false);
    }

    public JSONObject getMetadata(boolean forceRefresh) {
        if(metadata == null || forceRefresh) {
            mainHandler.post(() -> outputText.setText(R.string.fetching_metadata));
            Log.d(Constants.LOG_TAG, "Fetching metadata");
            try {
                // Try to fetch from network
                JSONObject newMetadata = FileUtils.fetchJSONFromURL(Constants.METADATA_URL);
                if (newMetadata != null) {
                    metadata = newMetadata;
                    prefs.edit().putString(PREF_METADATA, metadata.toString()).apply();
                }
            } catch(Exception ex1) {
                Log.e(Constants.LOG_TAG, "Exception fetching metadata ", ex1);
            }
            
            // If network failed or didn't update, try cache
            if (metadata == null) {
                String cachedJson = prefs.getString(PREF_METADATA, null);
                if (cachedJson != null) {
                    try {
                        metadata = new JSONObject(cachedJson);
                    } catch (JSONException ex2) {
                        Log.e(Constants.LOG_TAG, "Exception parsing cached metadata", ex2);
                    }
                }
            }
            
            if (metadata == null) {
                metadata = new JSONObject();
            }
        }
        return metadata;
    }

    public long getModelSize(String modelName) {
        return getMetadata(modelName).optLong(FIELD_SIZE);
    }
    
    public List<InsectModel> getAvailableModels() {
        List<InsectModel> models = new ArrayList<>();
        JSONObject metadata = getMetadata(false);
        if (metadata != null) {
            Iterator<String> keys = metadata.keys();
            while (keys.hasNext()) {
                String key = keys.next();
                if (key.startsWith("::")) continue; 
                
                JSONObject modelJson = metadata.optJSONObject(key);
                if (modelJson != null && !Constants.ROOT_CLASSIFIER.equals(key)) {
                    if (modelJson.optBoolean(Constants.FIELD_IS_ROOT, false)) {
                        continue;
                    }
                    InsectModel model = InsectModel.fromJson(key, modelJson);
                    if (model.isEnabled()) {
                        models.add(model);
                    }
                }
            }
        }
        Collections.sort(models);
        return models;
    }
}
