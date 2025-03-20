package com.rakeshmalik.insectid.utils;

import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CommonUtils {

    public static float[] toSoftMax(float[] scores) {
        float sumExp = 0.0f;
        for (float score : scores) {
            sumExp += (float) Math.exp(score);
        }
        for (int i = 0; i < scores.length; i++) {
            scores[i] = (float) Math.exp(scores[i]) / sumExp;
        }
        return scores;
    }

    public static Integer[] getSortedIndices(float[] array) {
        return getTopKIndices(array, array.length);
    }

    public static Integer[] getTopKIndices(float[] array, int k) {
        Integer[] indices = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, (i1, i2) -> Double.compare(array[i2], array[i1]));
        return Arrays.copyOfRange(indices, 0, k);
    }

    public static List<String> toList(JSONArray jsonArr) {
        List<String> list = new ArrayList<>();
        if(jsonArr != null) {
            for (int i = 0; i < jsonArr.length(); i++) {
                try {
                    list.add(jsonArr.getString(i));
                } catch (JSONException ex) {
                    Log.e(LOG_TAG, "Exception during JSONArray to List<String> conversion", ex);
                }
            }
        }
        return list;
    }

    public static Bitmap topBottomEdgeCrop(Bitmap img, float cropAmount) {
        return Bitmap.createBitmap(img,
                0, (int) (img.getHeight() * cropAmount),
                img.getWidth(), (int) (img.getHeight() * (1 - cropAmount*2)));
    }

    public static Bitmap centerSquareCrop(Bitmap img) {
        int centerCropSize = Math.min(img.getWidth(), img.getHeight());
        return Bitmap.createBitmap(img,
                (img.getWidth() - centerCropSize) / 2, (img.getHeight() - centerCropSize) / 2,
                centerCropSize, centerCropSize);
    }

    public static Bitmap loadImageFromUrl(String urlString, int timeout) {
        try {
            URL url = new URL(urlString);
            URLConnection connection = url.openConnection();
            connection.setConnectTimeout(timeout);
            connection.setReadTimeout(timeout);
            try (InputStream is = connection.getInputStream();
                 ByteArrayOutputStream buffer = new ByteArrayOutputStream();) {
                byte[] data = new byte[8192];
                int bytesRead;
                while ((bytesRead = is.read(data, 0, data.length)) != -1) {
                    if(Thread.currentThread().isInterrupted()) {
                        throw new RuntimeException("thread interrupted");
                    }
                    buffer.write(data, 0, bytesRead);
                }
                byte[] imageBytes = buffer.toByteArray();
                return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
            }
        } catch (Exception ex2) {
            Log.e(LOG_TAG, "Exception loading image " + urlString, ex2);
        }
        return null;
    }

    // loads partial image on timeout
    public static Bitmap loadImageFromUrlAsStream(String urlString, int timeout) {
        try {
            URL url = new URL(urlString);
            URLConnection connection = url.openConnection();
            connection.setConnectTimeout(timeout);
            connection.setReadTimeout(timeout);
            try (InputStream is = (InputStream) connection.getInputStream()) {
                return BitmapFactory.decodeStream(is);
            } catch (Exception ex1) {
                Log.e(LOG_TAG, "Exception reading image " + urlString, ex1);
            }
        } catch (Exception ex2) {
            Log.e(LOG_TAG, "Exception loading image " + urlString, ex2);
        }
        return null;
    }

}
