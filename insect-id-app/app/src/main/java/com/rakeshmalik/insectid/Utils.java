package com.rakeshmalik.insectid;

import static com.rakeshmalik.insectid.Constants.LOG_TAG;

import android.graphics.Bitmap;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Utils {

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

}
