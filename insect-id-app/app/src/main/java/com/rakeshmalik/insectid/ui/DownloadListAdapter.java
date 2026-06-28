package com.rakeshmalik.insectid.ui;

import android.graphics.Typeface;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.RecyclerView;

import com.rakeshmalik.insectid.R;
import com.rakeshmalik.insectid.filemanager.DownloadItem;

import android.graphics.Color;
import java.util.ArrayList;
import java.util.List;

public class DownloadListAdapter extends RecyclerView.Adapter<RecyclerView.ViewHolder> {
    private List<DownloadItem> originalItems = new ArrayList<>();
    private List<DownloadItem> flatItems = new ArrayList<>();

    public DownloadListAdapter(List<DownloadItem> items) {
        setItems(items);
    }

    public void setItems(List<DownloadItem> items) {
        this.originalItems = items;
        flattenItems();
    }

    private void flattenItems() {
        flatItems.clear();
        for (DownloadItem item : originalItems) {
            flatItems.add(item);
            if (item.isExpanded) {
                flatItems.addAll(item.children);
            }
        }
        notifyDataSetChanged();
    }

    public String getCountInfo(String modelName, String downloadName) {
        int total = 0;
        int current = 0;
        for (DownloadItem model : originalItems) {
            for (DownloadItem file : model.children) {
                total++;
                if (file.title.equals(downloadName) && model.id.equals(modelName)) {
                    current = total;
                }
            }
        }
        if (current > 0) {
            return String.format("File %d of %d", current, total);
        }
        return "";
    }

    public void updateProgress(String modelName, String downloadName) {
        boolean found = false;
        boolean stateChanged = false;
        
        // Let's iterate and find the currently active item
        for (DownloadItem model : originalItems) {
            for (DownloadItem file : model.children) {
                if (file.title.equals(downloadName) && model.id.equals(modelName)) {
                    if (!file.isDownloading) {
                        file.isDownloading = true;
                        file.isCompleted = false;
                        stateChanged = true;
                    }
                    
                    // Mark all previous files in the whole list as completed
                    boolean reachedTarget = false;
                    for (DownloadItem prevModel : originalItems) {
                        for (DownloadItem prevFile : prevModel.children) {
                            if (prevFile == file) {
                                reachedTarget = true;
                                break;
                            }
                            if (!prevFile.isCompleted || prevFile.isDownloading) {
                                prevFile.isCompleted = true;
                                prevFile.isDownloading = false;
                                stateChanged = true;
                            }
                        }
                        if (reachedTarget) {
                            break;
                        }
                    }
                    
                    // Ensure current model is expanded
                    if (!model.isExpanded) {
                        model.isExpanded = true;
                        stateChanged = true;
                    }
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        
        if (found && stateChanged) {
            flattenItems();
        }
    }

    @Override
    public int getItemViewType(int position) {
        return flatItems.get(position).type;
    }

    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        if (viewType == DownloadItem.TYPE_MODEL) {
            View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_download_model, parent, false);
            return new ModelViewHolder(view);
        } else {
            View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_download_file, parent, false);
            return new FileViewHolder(view);
        }
    }

    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
        DownloadItem item = flatItems.get(position);
        if (holder instanceof ModelViewHolder) {
            ModelViewHolder modelHolder = (ModelViewHolder) holder;
            modelHolder.modelTitle.setText(item.title);
            if (item.isExpanded) {
                modelHolder.expandIcon.setImageResource(R.drawable.ic_expand_less);
            } else {
                modelHolder.expandIcon.setImageResource(R.drawable.ic_expand_more);
            }
            
            modelHolder.itemView.setOnClickListener(v -> {
                item.isExpanded = !item.isExpanded;
                flattenItems();
            });
        } else if (holder instanceof FileViewHolder) {
            FileViewHolder fileHolder = (FileViewHolder) holder;
            fileHolder.fileTitle.setText(item.title);
            
            if (item.isDownloading) {
                fileHolder.fileTitle.setTextColor(ContextCompat.getColor(fileHolder.itemView.getContext(), com.google.android.material.R.color.design_default_color_primary));
                fileHolder.fileTitle.setTypeface(null, Typeface.BOLD);
                fileHolder.fileProgressIcon.setVisibility(View.VISIBLE);
                fileHolder.fileStatusIcon.setVisibility(View.GONE);
            } else if (item.isCompleted) {
                fileHolder.fileTitle.setTextColor(Color.GRAY);
                fileHolder.fileTitle.setTypeface(null, Typeface.NORMAL);
                fileHolder.fileProgressIcon.setVisibility(View.GONE);
                fileHolder.fileStatusIcon.setVisibility(View.VISIBLE);
            } else {
                // Not started
                fileHolder.fileTitle.setTextColor(Color.GRAY);
                fileHolder.fileTitle.setTypeface(null, Typeface.NORMAL);
                fileHolder.fileProgressIcon.setVisibility(View.GONE);
                fileHolder.fileStatusIcon.setVisibility(View.GONE);
            }
        }
    }

    @Override
    public int getItemCount() {
        return flatItems.size();
    }

    static class ModelViewHolder extends RecyclerView.ViewHolder {
        TextView modelTitle;
        ImageView expandIcon;

        public ModelViewHolder(@NonNull View itemView) {
            super(itemView);
            modelTitle = itemView.findViewById(R.id.modelTitle);
            expandIcon = itemView.findViewById(R.id.expandIcon);
        }
    }

    static class FileViewHolder extends RecyclerView.ViewHolder {
        TextView fileTitle;
        ImageView fileStatusIcon;
        ProgressBar fileProgressIcon;

        public FileViewHolder(@NonNull View itemView) {
            super(itemView);
            fileTitle = itemView.findViewById(R.id.fileTitle);
            fileStatusIcon = itemView.findViewById(R.id.fileStatusIcon);
            fileProgressIcon = itemView.findViewById(R.id.fileProgressIcon);
        }
    }
}
