package com.rakeshmalik.insectid.ui;

import android.content.SharedPreferences;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.RecyclerView;
import com.google.android.material.button.MaterialButton;
import com.rakeshmalik.insectid.R;
import com.rakeshmalik.insectid.filemanager.ModelDownloader;
import com.rakeshmalik.insectid.pojo.InsectModel;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class ManageModelsAdapter extends RecyclerView.Adapter<ManageModelsAdapter.ViewHolder> {

    public interface OnModelDownloadClickListener {
        void onDownloadClick(InsectModel model);
    }

    public interface OnModelOffloadClickListener {
        void onOffloadClick(InsectModel model);
    }

    private List<InsectModel> models;
    private final ModelDownloader modelDownloader;
    private final SharedPreferences prefs;
    private final OnModelDownloadClickListener downloadClickListener;
    private final OnModelOffloadClickListener offloadClickListener;
    private final Set<Integer> expandedPositions = new HashSet<>();

    public ManageModelsAdapter(List<InsectModel> models, ModelDownloader modelDownloader, SharedPreferences prefs, 
                               OnModelDownloadClickListener downloadClickListener, OnModelOffloadClickListener offloadClickListener) {
        this.models = models;
        this.modelDownloader = modelDownloader;
        this.prefs = prefs;
        this.downloadClickListener = downloadClickListener;
        this.offloadClickListener = offloadClickListener;
    }

    public void updateModels(List<InsectModel> models) {
        this.models = models;
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_manage_model, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        InsectModel model = models.get(position);
        holder.bind(model, modelDownloader, prefs, downloadClickListener, offloadClickListener);
    }

    @Override
    public int getItemCount() {
        return models.size();
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        TextView modelName;
        ImageView statusIcon;
        TextView modelStatus;
        TextView modelLatestVersion;
        TextView modelWarning;
        ImageView iconLegacy;
        ImageView iconExperimental;
        MaterialButton btnDownload;
        MaterialButton btnOffload;
        
        View expandableInfoContainer;
        TextView infoDescription;
        TextView infoAccuracy;
        TextView infoCounts;
        TextView infoArchitecture;
        TextView infoDataSources;

        ViewHolder(View itemView) {
            super(itemView);
            modelName = itemView.findViewById(R.id.modelName);
            statusIcon = itemView.findViewById(R.id.statusIcon);
            modelStatus = itemView.findViewById(R.id.modelStatus);
            modelLatestVersion = itemView.findViewById(R.id.modelLatestVersion);
            modelWarning = itemView.findViewById(R.id.modelWarning);
            iconLegacy = itemView.findViewById(R.id.iconLegacy);
            iconExperimental = itemView.findViewById(R.id.iconExperimental);
            btnDownload = itemView.findViewById(R.id.btnDownload);
            btnOffload = itemView.findViewById(R.id.btnOffload);
            
            expandableInfoContainer = itemView.findViewById(R.id.expandableInfoContainer);
            infoDescription = itemView.findViewById(R.id.infoDescription);
            infoAccuracy = itemView.findViewById(R.id.infoAccuracy);
            infoCounts = itemView.findViewById(R.id.infoCounts);
            infoArchitecture = itemView.findViewById(R.id.infoArchitecture);
            infoDataSources = itemView.findViewById(R.id.infoDataSources);
        }

        void bind(InsectModel model, ModelDownloader modelDownloader, SharedPreferences prefs, 
                  OnModelDownloadClickListener downloadClickListener, OnModelOffloadClickListener offloadClickListener) {
            modelName.setText(model.getDisplayName());
            
            iconLegacy.setVisibility(model.isLegacy() ? View.VISIBLE : View.GONE);
            iconExperimental.setVisibility(model.isExperimental() ? View.VISIBLE : View.GONE);
            
            if (model.isLegacy()) {
                modelWarning.setText("Legacy model: performance may not be optimal.");
                modelWarning.setVisibility(View.VISIBLE);
            } else if (model.isExperimental()) {
                modelWarning.setText("Experimental model: performance may not be optimal.");
                modelWarning.setVisibility(View.VISIBLE);
            } else {
                modelWarning.setVisibility(View.GONE);
            }
            
            boolean isQueuedOrDownloading = modelDownloader.isModelQueuedOrDownloading(model);
            boolean isDownloaded = modelDownloader.isModelAlreadyDownloaded(model);
            int currentVersion = prefs.getInt(ModelDownloader.modelVersionPrefName(model.getModelName()), 0);
            int latestVersion = model.getVersion();
            
            modelLatestVersion.setText("Latest version: v" + latestVersion);
            
            if (isQueuedOrDownloading) {
                statusIcon.setImageResource(R.drawable.ic_download);
                statusIcon.clearColorFilter();
                modelStatus.setText("Downloading / Queued");
                btnDownload.setText("Downloading");
                btnDownload.setEnabled(false);
                btnDownload.setVisibility(View.VISIBLE);
                btnOffload.setVisibility(View.GONE);
            } else if (isDownloaded) {
                if (currentVersion < latestVersion) {
                    statusIcon.setImageResource(R.drawable.ic_download);
                    statusIcon.clearColorFilter();
                    modelStatus.setText("Downloaded (v" + currentVersion + ") - Update available");
                    btnDownload.setText("Update");
                    btnDownload.setEnabled(true);
                    btnDownload.setVisibility(View.VISIBLE);
                    btnOffload.setVisibility(View.VISIBLE);
                } else {
                    statusIcon.setImageResource(R.drawable.ic_check_circle);
                    statusIcon.setColorFilter(ContextCompat.getColor(itemView.getContext(), R.color.primaryGreen));
                    modelStatus.setText("Downloaded (v" + currentVersion + ")");
                    btnDownload.setVisibility(View.GONE);
                    btnOffload.setVisibility(View.VISIBLE);
                }
            } else {
                statusIcon.setImageResource(R.drawable.ic_download);
                statusIcon.clearColorFilter();
                modelStatus.setText("Not downloaded");
                btnDownload.setText("Download");
                btnDownload.setEnabled(true);
                btnDownload.setVisibility(View.VISIBLE);
                btnOffload.setVisibility(View.GONE);
            }
            
            btnDownload.setOnClickListener(v -> {
                if (downloadClickListener != null) {
                    downloadClickListener.onDownloadClick(model);
                }
            });

            btnOffload.setOnClickListener(v -> {
                if (offloadClickListener != null) {
                    offloadClickListener.onOffloadClick(model);
                }
            });
            
            boolean isExpanded = expandedPositions.contains(getAdapterPosition());
            expandableInfoContainer.setVisibility(isExpanded ? View.VISIBLE : View.GONE);
            
            if (isExpanded) {
                if (model.getDescription() != null && !model.getDescription().isEmpty()) {
                    infoDescription.setText(model.getDescription());
                    infoDescription.setVisibility(View.VISIBLE);
                } else {
                    infoDescription.setVisibility(View.GONE);
                }
                
                InsectModel.ModelStats stats = model.getStats();
                if (stats != null) {
                    if (stats.getAccuracy() != null) {
                        infoAccuracy.setText("Accuracy: " + stats.getAccuracy() + 
                                (stats.getAccuracyTop3() != null ? " (Top-3: " + stats.getAccuracyTop3() + ")" : ""));
                        infoAccuracy.setVisibility(View.VISIBLE);
                    } else {
                        infoAccuracy.setVisibility(View.GONE);
                    }
                    
                    if (stats.getClassCount() > 0) {
                        infoCounts.setText("Classes: " + stats.getClassCount() + 
                                " | Species: " + stats.getSpeciesCount() + 
                                " | Data: " + stats.getDataCount());
                        infoCounts.setVisibility(View.VISIBLE);
                    } else {
                        infoCounts.setVisibility(View.GONE);
                    }
                    
                    if (stats.getModelArch() != null) {
                        infoArchitecture.setText("Architecture: " + stats.getModelArch());
                        infoArchitecture.setVisibility(View.VISIBLE);
                    } else {
                        infoArchitecture.setVisibility(View.GONE);
                    }
                    
                    if (stats.getDataSources() != null && !stats.getDataSources().isEmpty()) {
                        infoDataSources.setText("Sources: " + String.join(", ", stats.getDataSources()));
                        infoDataSources.setVisibility(View.VISIBLE);
                    } else {
                        infoDataSources.setVisibility(View.GONE);
                    }
                } else {
                    infoAccuracy.setVisibility(View.GONE);
                    infoCounts.setVisibility(View.GONE);
                    infoArchitecture.setVisibility(View.GONE);
                    infoDataSources.setVisibility(View.GONE);
                }
            }
            
            itemView.setOnClickListener(v -> {
                int position = getAdapterPosition();
                if (position != RecyclerView.NO_POSITION) {
                    if (expandedPositions.contains(position)) {
                        expandedPositions.remove(position);
                    } else {
                        expandedPositions.add(position);
                    }
                    notifyItemChanged(position);
                }
            });
        }
    }
}
