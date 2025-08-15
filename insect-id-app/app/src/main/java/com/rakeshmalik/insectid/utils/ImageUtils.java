package com.rakeshmalik.insectid.utils;

import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.Objects;

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

    @Deprecated
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

}
