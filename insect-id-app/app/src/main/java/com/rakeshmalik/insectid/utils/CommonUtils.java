package com.rakeshmalik.insectid.utils;

import static com.rakeshmalik.insectid.constants.Constants.DERIVED_CLASS_REGEX;
import static com.rakeshmalik.insectid.constants.Constants.EARLY_STAGE_CLASS_SUFFIX;
import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;

import android.content.Context;
import android.content.pm.ApplicationInfo;
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

    public static boolean isDebugMode(Context context) {
        return (context.getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) != 0;
    }

    public static float[] toSoftMax(float[] scores) {
        float max = Float.NEGATIVE_INFINITY;
        for (float score : scores) {
            if (score > max) max = score;
        }
        float sumExp = 0.0f;
        for (int i = 0; i < scores.length; i++) {
            scores[i] = (float) Math.exp(scores[i] - max);
            sumExp += scores[i];
        }
        for (int i = 0; i < scores.length; i++) {
            scores[i] /= sumExp;
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

    public static boolean isEarlyStage(String className) {
        return className.endsWith(EARLY_STAGE_CLASS_SUFFIX);
    }

    public static String getImagoClassName(String earlyClassName) {
        return earlyClassName.replaceAll(EARLY_STAGE_CLASS_SUFFIX + "$", "");
    }

    public static boolean isDerivedClass(String className) {
        return className.matches(DERIVED_CLASS_REGEX);
    }

    public static String getGenus(String className) {
        return className.split("-")[0];
    }

    public static boolean isPossibleDuplicate(String className1, String className2) {
        return className1.split("-", 2)[1].equals(className2.split("-", 2)[1]);
    }

}
