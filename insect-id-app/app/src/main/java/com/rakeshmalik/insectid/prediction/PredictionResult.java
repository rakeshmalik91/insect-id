package com.rakeshmalik.insectid.prediction;

import android.graphics.Bitmap;
import java.util.List;

public class PredictionResult {
    public final String scientificName;
    public final String commonName;
    public final float score;
    public final String searchUrl;
    public final String modelName;
    public final String className;
    public List<Bitmap> images; // Loaded asynchronously

    public PredictionResult(String scientificName, String commonName, float score, String searchUrl, String modelName, String className) {
        this.scientificName = scientificName;
        this.commonName = commonName;
        this.score = score;
        this.searchUrl = searchUrl;
        this.modelName = modelName;
        this.className = className;
    }
}
