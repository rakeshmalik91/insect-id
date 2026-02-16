package com.rakeshmalik.insectid.prediction;

import static com.rakeshmalik.insectid.constants.Constants.HTML_NO_IMAGE_AVAILABLE;
import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.text.Html;
import android.text.Spanned;
import android.util.Log;
import android.widget.TextView;

import com.rakeshmalik.insectid.MainActivity;
import com.rakeshmalik.insectid.R;
import com.rakeshmalik.insectid.filemanager.ModelDownloader;
import com.rakeshmalik.insectid.filemanager.ModelLoader;

import java.util.List;

public class PredictionRunnable implements Runnable {

    private final TextView outputText;
    private final PredictionManager predictionManager;
    private final MainActivity mainActivity;
    private final ModelLoader modelLoader;
    private final ModelDownloader modelDownloader;

    public PredictionRunnable(MainActivity mainActivity, PredictionManager predictionManager, ModelLoader modelLoader, ModelDownloader modelDownloader) {
        this.mainActivity = mainActivity;
        this.outputText = mainActivity.getOutputText();
        this.predictionManager = predictionManager;
        this.modelLoader = modelLoader;
        this.modelDownloader = modelDownloader;
    }

    @Override
    public void run() {
        Log.d(LOG_TAG, "Inside PredictionRunnable.run()");
        try {
            mainActivity.lockUI();
            modelDownloader.downloadModel(mainActivity.getSelectedModel(), this::runPrediction, mainActivity::unlockUI, 1, 1);
        } catch(Exception ex) {
            mainActivity.unlockUI();
        }
    }

    private void runPrediction() {
        Log.d(LOG_TAG, "Inside PredictionRunnable.runPrediction()");
        try {
            mainActivity.runOnUiThread(() -> outputText.setText(R.string.predicting));
            String predictions = predictionManager.predict(mainActivity.getSelectedModel(), mainActivity.getPhotoUri());

            Log.d(LOG_TAG, "Inside PredictionRunnable.runPrediction(): Going to render");
            // set html with alt text while loading images
            Spanned htmlWithoutImage = Html.fromHtml(getHtmlWithoutImage(predictions), Html.FROM_HTML_MODE_COMPACT, null, null);
            mainActivity.runOnUiThread(() -> outputText.setText(htmlWithoutImage));
            mainActivity.unlockUI();

            // render html with images
            Spanned html = Html.fromHtml(predictions, Html.FROM_HTML_MODE_COMPACT, this::predictedImageRenderer, null);
            mainActivity.runOnUiThread(() -> outputText.setText(html));
        } catch(Exception ex) {
            Log.e(LOG_TAG, "Exception during prediction", ex);
        } finally {
            mainActivity.unlockUI();
        }
    }

    public Drawable predictedImageRenderer(String source) {
        try {
            List<Bitmap> images = modelLoader.getImagesFromZip(mainActivity, source.split("/")[0], source.split("/")[1]);
            int maxColumns = 3, gap = 10;
            int size = (outputText.getWidth() - gap * (maxColumns - 1)) / maxColumns;
            int columns = Math.min(images.size(), maxColumns);
            int rows = (int) Math.ceil((double) images.size() / maxColumns);
            int gridWidth = size * columns + gap * (columns - 1);
            int gridHeight = size * rows + gap * (rows - 1);
            Bitmap bitmap = Bitmap.createBitmap(gridWidth, gridHeight, Bitmap.Config.ARGB_8888);
            Canvas canvas = new Canvas(bitmap);
            int left = 0, top = 0;
            for (int i = 0; i < images.size(); i++) {
                Bitmap img = images.get(i);
                img = Bitmap.createScaledBitmap(img, size, size, true);
                canvas.drawBitmap(img, left, top, null);
                if (i % maxColumns == maxColumns - 1) {
                    top += size + gap;
                    left = 0;
                } else {
                    left += size + gap;
                }
            }
            Drawable drawable = new BitmapDrawable(mainActivity.getResources(), bitmap);
            drawable.setBounds(0, 0, gridWidth, gridHeight);
            return drawable;
        } catch(RuntimeException ex) {
            Log.e(LOG_TAG, "Exception in predictedImageRenderer", ex);
            return null;
        }
    }

    private String getHtmlWithoutImage(String predictions) {
        String predictionsWithoutImage = predictions.replaceAll("<img [^>]+/>", "");
        while(predictionsWithoutImage.contains(HTML_NO_IMAGE_AVAILABLE)) {
            predictionsWithoutImage = predictionsWithoutImage.replace(HTML_NO_IMAGE_AVAILABLE, "");
        }
        return predictionsWithoutImage;
    }

}