package com.rakeshmalik.insectid;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.TextView;

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

    public MetadataManager(TextView outputText) {
        this.outputText = outputText;
    }

    public JSONObject getMetadata(ModelType modelType) {
        return getMetadata().optJSONObject(modelType.modelName);
    }

    public JSONObject getMetadata() {
        if(metadata == null) {
            mainHandler.post(() -> outputText.setText(R.string.fetching_metadata));
            Log.d(Constants.LOG_TAG, "Fetching metadata");
            metadata = fetchJSONFromURL(Constants.METADATA_URL);
        }
        return metadata;
    }

    private JSONObject fetchJSONFromURL(String url) {
        Log.d(Constants.LOG_TAG, "Fetching json file from url: " + url);
        OkHttpClient client = new OkHttpClient();
        Request request = new Request.Builder().url(url).build();
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);
            String data = response.body().string();
            Log.d(Constants.LOG_TAG, "Content: " + data);
            return new JSONObject(data);
        } catch (IOException | JSONException ex) {
            Log.d(Constants.LOG_TAG, "Exception during fetching json file from " + url, ex);
            return null;
        }
    }

}
