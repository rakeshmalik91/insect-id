package com.rakeshmalik.insectid.utils;

import android.graphics.Bitmap;
import org.pytorch.executorch.Tensor;

public class TensorImageUtils {

    /**
     * Converts an RGB Bitmap to a Float32 Tensor with shape [1, 3, height, width],
     * normalized using the provided mean and std per channel.
     *
     * @param bitmap input RGB Bitmap
     * @param normMean float[3] array of mean values (e.g. {0.485f, 0.456f, 0.406f})
     * @param normStd float[3] array of std values (e.g. {0.229f, 0.224f, 0.225f})
     * @return ExecuTorch Tensor suitable for inference
     */
    public static Tensor bitmapToFloat32Tensor(Bitmap bitmap, float[] normMean, float[] normStd) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        float[] floatBuffer = new float[3 * width * height];
        int offsetR = 0;
        int offsetG = width * height;
        int offsetB = 2 * width * height;

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            // Extract RGB components (ignore alpha)
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;

            // Normalize each channel
            floatBuffer[offsetR++] = (r - normMean[0]) / normStd[0];
            floatBuffer[offsetG++] = (g - normMean[1]) / normStd[1];
            floatBuffer[offsetB++] = (b - normMean[2]) / normStd[2];
        }

        long[] shape = new long[]{1, 3, height, width};
        return Tensor.fromBlob(floatBuffer, shape);
    }
}
