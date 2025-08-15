package com.rakeshmalik.insectid.utils;

import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;

import android.graphics.Bitmap;
import android.util.Log;

import com.rakeshmalik.insectid.enums.Operation;

import java.util.Arrays;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class CVImageUtils {

    public static Bitmap applyGaussianBlur(Bitmap bitmap, double radiusRatio) {
        return applyGaussianBlur(bitmap, (int) Math.ceil(Math.min(bitmap.getHeight(), bitmap.getWidth()) * radiusRatio));
    }

    public static Bitmap applyGaussianBlur(Bitmap bitmap, int radius) {
        Log.d(LOG_TAG, "Applying openCV gaussian blur");
        Mat mat = new Mat();
        try {
            Utils.bitmapToMat(bitmap, mat);
            int kernelSize = (int) (2 * radius + 1);
            Imgproc.GaussianBlur(mat, mat, new Size(kernelSize, kernelSize), 0);
            Bitmap blurredBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat, blurredBitmap);
            return blurredBitmap;
        } finally {
            mat.release();
        }
    }

    public static boolean isScreenCapture(Bitmap bitmap, double threshold) {
        Mat image = new Mat(), edges = new Mat();
        try {
            Utils.bitmapToMat(bitmap, image);
            Imgproc.Canny(image, edges, 50, 150);
            int edgeCount = Core.countNonZero(edges);
            double edgeRatio = (double) edgeCount / (image.rows() * image.cols());
            boolean isCapture = edgeRatio > threshold;
            if(isCapture) {
                Log.d(LOG_TAG, "Image looks like screen capture. canny edge ratio = " + edgeRatio);
            }
            return isCapture;
        } finally {
            image.release();
            edges.release();
        }
    }

    public static Bitmap removeBlackBorders(Bitmap inputBitmap, int threshold, Operation opType) {
        Mat src = new Mat(), gray = new Mat(), cropped = null;
        try {
            // Convert Bitmap to Mat
            Utils.bitmapToMat(inputBitmap, src);

            // Convert to grayscale
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);

            int rows = gray.rows();
            int cols = gray.cols();
            int[] rowExtract = new int[rows];
            int[] colExtract = new int[cols];

            // Compute row and column intensity medians
            for (int y = 0; y < rows; y++) {
                double[] rowPixels = new double[cols];
                for (int x = 0; x < cols; x++) {
                    rowPixels[x] = gray.get(y, x)[0];
                }
                rowExtract[y] = (int) extract(opType, rowPixels);
            }
            for (int x = 0; x < cols; x++) {
                double[] colPixels = new double[rows];
                for (int y = 0; y < rows; y++) {
                    colPixels[y] = gray.get(y, x)[0];
                }
                colExtract[x] = (int) extract(opType, colPixels);
            }

            // Find first and last non-black row
            int yMin = 0, yMax = rows - 1;
            while (yMin < rows && rowExtract[yMin] < threshold) yMin++;
            while (yMax > yMin && rowExtract[yMax] < threshold) yMax--;

            // Find first and last non-black column
            int xMin = 0, xMax = cols - 1;
            while (xMin < cols && colExtract[xMin] < threshold) xMin++;
            while (xMax > xMin && colExtract[xMax] < threshold) xMax--;

            // Ensure valid crop dimensions
            if (xMax > xMin && yMax > yMin) {
                Rect roi = new Rect(xMin, yMin, xMax - xMin, yMax - yMin);
                cropped = new Mat(src, roi);
                Bitmap outputBitmap = Bitmap.createBitmap(cropped.cols(), cropped.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(cropped, outputBitmap);
                return outputBitmap;
            }

            return inputBitmap; // Return original if no crop found
        } finally {
            src.release();
            gray.release();
            if (cropped != null) {
                cropped.release();
            }
        }
    }

    private static double extract(Operation type, double[] values) {
        switch (type) {
            case MEDIAN:
                Arrays.sort(values);
                int middle = values.length / 2;
                if (values.length % 2 == 0) {
                    return (values[middle - 1] + values[middle]) / 2.0;
                } else {
                    return values[middle];
                }
            case MEAN:
                return Arrays.stream(values).average().orElse(0);
        }
        return 0;
    }

}
