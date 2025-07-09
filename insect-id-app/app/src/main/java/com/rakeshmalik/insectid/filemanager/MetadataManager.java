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
import com.rakeshmalik.insectid.enums.ModelType;
import com.rakeshmalik.insectid.utils.FileUtils;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

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

    public JSONObject getMetadata(ModelType modelType) {
        return getMetadata(modelType.modelName);
    }

    public JSONObject getMetadata(String modelName) {
        return getMetadata().optJSONObject(modelName);
    }

    public JSONObject getMetadata() {
        return getMetadata(false);
    }

    public JSONObject getMetadata(boolean forceRefresh) {
        if(metadata == null || forceRefresh) {
            mainHandler.post(() -> outputText.setText(R.string.fetching_metadata));
            Log.d(Constants.LOG_TAG, "Fetching metadata");
            try {
                metadata = FileUtils.fetchJSONFromURL(Constants.METADATA_URL);
                prefs.edit().putString(PREF_METADATA, metadata.toString()).apply();
            } catch(Exception ex1) {
                Log.e(Constants.LOG_TAG, "Exception fetching metadata ", ex1);
                if(prefs.contains(PREF_METADATA)) {
                    try {
                        metadata = new JSONObject(prefs.getString(PREF_METADATA, "{}"));
                    } catch(Exception ex2) {
                        Log.e(Constants.LOG_TAG, "Exception during parsing metadata json from shared prefs", ex2);
                    }
                }
            }
        }
        return metadata;
    }

    public long getModelSize(String modelName) {
        return getMetadata(modelName).optLong(FIELD_SIZE);
    }

}
