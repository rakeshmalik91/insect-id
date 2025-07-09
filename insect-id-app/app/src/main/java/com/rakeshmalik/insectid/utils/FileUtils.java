package com.rakeshmalik.insectid.utils;

import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;

import android.util.Log;

import com.rakeshmalik.insectid.constants.Constants;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

public class FileUtils {

    public static JSONObject fetchJSONFromURL(String url) {
        Log.d(Constants.LOG_TAG, "Fetching json file from url: " + url);
        OkHttpClient client = new OkHttpClient();
        Request request = new Request.Builder().url(url).build();
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);
            String data = response.body().string();
            return new JSONObject(data);
        } catch (IOException | JSONException ex) {
            Log.e(Constants.LOG_TAG, "Exception during fetching json file from " + url, ex);
            return null;
        }
    }

    public static long getFileSizeFromUrl(String fileUrl) {
        long fileSize = 0;
        HttpURLConnection connection = null;
        try {
            URL url = new URL(fileUrl);
            connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("HEAD");
            connection.setConnectTimeout(5000);
            connection.setReadTimeout(5000);
            connection.connect();

            if (connection.getResponseCode() == HttpURLConnection.HTTP_OK) {
                fileSize = connection.getContentLengthLong();
            }
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception in FileUtils.getFileSizeFromUrl() for " + fileUrl, ex);
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
        return fileSize;
    }

}
