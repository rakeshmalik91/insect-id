package com.rakeshmalik.insectid.prediction;

import java.util.List;

public class PredictionResponse {
    public final String errorMessage;
    public final List<PredictionResult> predictions;

    public PredictionResponse(String errorMessage) {
        this.errorMessage = errorMessage;
        this.predictions = null;
    }

    public PredictionResponse(List<PredictionResult> predictions) {
        this.errorMessage = null;
        this.predictions = predictions;
    }
}
