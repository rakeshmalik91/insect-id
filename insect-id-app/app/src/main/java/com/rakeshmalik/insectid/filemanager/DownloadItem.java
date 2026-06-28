package com.rakeshmalik.insectid.filemanager;

import java.util.ArrayList;
import java.util.List;

public class DownloadItem {
    public static final int TYPE_MODEL = 0;
    public static final int TYPE_FILE = 1;

    public int type;
    public String title;
    public String id; // modelName for model, modelName_fileName for file
    public boolean isExpanded = false;
    public boolean isDownloading = false;
    public boolean isCompleted = false;

    // For file types
    public DownloadItem parent;
    
    // For model types
    public List<DownloadItem> children = new ArrayList<>();

    public DownloadItem(int type, String title, String id) {
        this.type = type;
        this.title = title;
        this.id = id;
    }
}
