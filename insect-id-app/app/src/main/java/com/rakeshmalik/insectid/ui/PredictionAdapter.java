package com.rakeshmalik.insectid.ui;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.GridLayout;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.rakeshmalik.insectid.R;
import com.rakeshmalik.insectid.prediction.PredictionResult;

import java.util.List;
import java.util.Locale;

public class PredictionAdapter extends RecyclerView.Adapter<PredictionAdapter.PredictionViewHolder> {

    private final Context context;
    private List<PredictionResult> predictions;

    public PredictionAdapter(Context context, List<PredictionResult> predictions) {
        this.context = context;
        this.predictions = predictions;
    }

    @SuppressLint("NotifyDataSetChanged")
    public void updateData(List<PredictionResult> newPredictions) {
        this.predictions = newPredictions;
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public PredictionViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context).inflate(R.layout.item_prediction, parent, false);
        return new PredictionViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull PredictionViewHolder holder, int position) {
        PredictionResult result = predictions.get(position);

        holder.textScientificName.setText(result.scientificName);
        holder.textScientificName.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(result.searchUrl));
            context.startActivity(intent);
        });

        holder.textCommonName.setText(result.commonName);
        holder.textScore.setText(String.format(Locale.getDefault(), "~%.2f%% match", result.score * 100));

        // Populate Images Grid
        holder.imagesGrid.removeAllViews();
        if (result.images != null && !result.images.isEmpty()) {
            int imageSize = (int) (100 * context.getResources().getDisplayMetrics().density);
            int margin = (int) (4 * context.getResources().getDisplayMetrics().density);

            for (Bitmap bitmap : result.images) {
                ImageView imageView = new ImageView(context);
                imageView.setImageBitmap(bitmap);
                imageView.setScaleType(ImageView.ScaleType.CENTER_CROP);
                
                GridLayout.LayoutParams params = new GridLayout.LayoutParams();
                params.width = imageSize;
                params.height = imageSize;
                params.setMargins(margin, margin, margin, margin);
                
                imageView.setLayoutParams(params);
                holder.imagesGrid.addView(imageView);
            }
        }
    }

    @Override
    public int getItemCount() {
        return predictions == null ? 0 : predictions.size();
    }

    static class PredictionViewHolder extends RecyclerView.ViewHolder {
        final TextView textScientificName;
        final TextView textCommonName;
        final TextView textScore;
        final GridLayout imagesGrid;

        public PredictionViewHolder(@NonNull View itemView) {
            super(itemView);
            textScientificName = itemView.findViewById(R.id.textScientificName);
            textCommonName = itemView.findViewById(R.id.textCommonName);
            textScore = itemView.findViewById(R.id.textScore);
            imagesGrid = itemView.findViewById(R.id.imagesGrid);
        }
    }
}
