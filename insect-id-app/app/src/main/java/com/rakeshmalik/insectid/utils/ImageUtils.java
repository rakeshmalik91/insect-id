package com.rakeshmalik.insectid.utils;

import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import com.rakeshmalik.insectid.enums.Operation;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.Arrays;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class ImageUtils {

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

    /*
    public static Bitmap applyGaussianBlur(Bitmap src, int radius) {
        int width = src.getWidth();
        int height = src.getHeight();
        Bitmap blurredBitmap = Bitmap.createBitmap(width, height, Objects.requireNonNull(src.getConfig()));

        int[] pixels = new int[width * height];
        src.getPixels(pixels, 0, width, 0, 0, width, height);

        int[] output = new int[width * height];

        // Generate Gaussian Kernel
        float[][] kernel = generateGaussianKernel(radius);
        int kernelSize = kernel.length;
        int halfKernelSize = kernelSize / 2;

        // Apply Gaussian Blur
        for (int y = halfKernelSize; y < height - halfKernelSize; y++) {
            for (int x = halfKernelSize; x < width - halfKernelSize; x++) {
                float r = 0, g = 0, b = 0;
                float sum = 0;

                for (int ky = -halfKernelSize; ky <= halfKernelSize; ky++) {
                    for (int kx = -halfKernelSize; kx <= halfKernelSize; kx++) {
                        int pixel = pixels[(y + ky) * width + (x + kx)];
                        float weight = kernel[ky + halfKernelSize][kx + halfKernelSize];

                        r += Color.red(pixel) * weight;
                        g += Color.green(pixel) * weight;
                        b += Color.blue(pixel) * weight;
                        sum += weight;
                    }
                }

                int newPixel = Color.rgb((int) (r / sum), (int) (g / sum), (int) (b / sum));
                output[y * width + x] = newPixel;
            }
        }

        blurredBitmap.setPixels(output, 0, width, 0, 0, width, height);
        return blurredBitmap;
    }

    private static float[][] generateGaussianKernel(int radius) {
        int size = 2 * radius + 1;
        float[][] kernel = new float[size][size];
        float sigma = radius / 3.0f;
        float sum = 0;

        for (int y = -radius; y <= radius; y++) {
            for (int x = -radius; x <= radius; x++) {
                float value = (float) Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
                kernel[y + radius][x + radius] = value;
                sum += value;
            }
        }

        // Normalize kernel
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                kernel[y][x] /= sum;
            }
        }

        return kernel;
    }
    */

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

    public static Bitmap applyGaussianBlur(Bitmap bitmap, double radiusRatio) {
        return ImageUtils.applyGaussianBlur(bitmap, (int) Math.ceil(Math.min(bitmap.getHeight(), bitmap.getWidth()) * radiusRatio));
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
