package com.rakeshmalik.insectid.ui;

import static com.rakeshmalik.insectid.MainActivity.PREF;
import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.google.android.material.card.MaterialCardView;
import com.rakeshmalik.insectid.R;
import com.rakeshmalik.insectid.filemanager.MetadataManager;
import com.rakeshmalik.insectid.filemanager.ModelDownloader;
import com.rakeshmalik.insectid.pojo.InsectModel;

import java.util.List;
import java.util.stream.Collectors;

public class ModelSelectorHelper {

    private final AppCompatActivity activity;
    private final LinearLayout modelSelectorContainer;
    private final TextView identifyModelWarning;
    private final TextView selectedModelName;
    private final MetadataManager metadataManager;
    private final ModelDownloader modelDownloader;
    private final ModelSelectorCallback callback;

    private InsectModel selectedModel;

    public interface ModelSelectorCallback {
        void onModelSelected(InsectModel model);
        boolean isUiLocked();
        void showMessage(String message);
    }

    public ModelSelectorHelper(AppCompatActivity activity,
                               LinearLayout modelSelectorContainer,
                               TextView identifyModelWarning,
                               TextView selectedModelName,
                               MetadataManager metadataManager,
                               ModelDownloader modelDownloader,
                               ModelSelectorCallback callback) {
        this.activity = activity;
        this.modelSelectorContainer = modelSelectorContainer;
        this.identifyModelWarning = identifyModelWarning;
        this.selectedModelName = selectedModelName;
        this.metadataManager = metadataManager;
        this.modelDownloader = modelDownloader;
        this.callback = callback;
    }

    public InsectModel getSelectedModel() {
        return selectedModel;
    }

    public void initMinHeight() {
        modelSelectorContainer.post(() -> {
            View scrollParent = (View) modelSelectorContainer.getParent();
            int availableWidth = scrollParent.getWidth() - scrollParent.getPaddingLeft() - scrollParent.getPaddingRight();
            float density = activity.getResources().getDisplayMetrics().density;
            int perCardMargin = (int) (16 * density);
            int cardSize = (availableWidth - (perCardMargin * 4)) / 4;
            int minSize = (int) (48 * density);
            if (cardSize < minSize) cardSize = minSize;
            int verticalMargins = (int) (12 * density); // 6dp top + 6dp bottom from item_model_selector.xml
            modelSelectorContainer.setMinimumHeight(cardSize + verticalMargins);
        });
    }

    public List<InsectModel> getVisibleModels() {
        SharedPreferences prefs = activity.getSharedPreferences(PREF, Context.MODE_PRIVATE);
        boolean showLegacy = prefs.getBoolean("show_legacy_models", false);
        boolean showExperimental = prefs.getBoolean("show_experimental_models", true);

        return metadataManager.getAvailableModels().stream()
                .filter(m -> {
                    if (m.isLegacy() && !showLegacy) return false;
                    if (m.isExperimental() && !showExperimental) return false;
                    return true;
                })
                .collect(Collectors.toList());
    }

    public void populateModelSpinner() {
        List<InsectModel> availableModels = getVisibleModels();
        activity.runOnUiThread(() -> {
            modelSelectorContainer.removeAllViews();

            // Use post to ensure the container has been measured
            modelSelectorContainer.post(() -> {
                View scrollParent = (View) modelSelectorContainer.getParent();
                int availableWidth = scrollParent.getWidth() - scrollParent.getPaddingLeft() - scrollParent.getPaddingRight();
                float density = activity.getResources().getDisplayMetrics().density;
                int perCardMargin = (int) (16 * density);
                int cardSize = (availableWidth - (perCardMargin * 4)) / 4;
                int minSize = (int) (48 * density);
                if (cardSize < minSize) cardSize = minSize;

                for (int i = 0; i < availableModels.size(); i++) {
                    InsectModel model = availableModels.get(i);
                    View cardView = activity.getLayoutInflater().inflate(R.layout.item_model_selector, modelSelectorContainer, false);
                    MaterialCardView card = cardView.findViewById(R.id.modelCard);
                    ImageView icon = cardView.findViewById(R.id.modelIcon);
                    ImageView typeIcon = cardView.findViewById(R.id.typeIcon);

                    // Set dynamic card size
                    ViewGroup.LayoutParams lp = card.getLayoutParams();
                    lp.width = cardSize;
                    lp.height = cardSize;
                    card.setLayoutParams(lp);

                    if (model.getIcon() != null && !model.getIcon().isEmpty()) {
                        int resId = activity.getResources().getIdentifier(model.getIcon(), "drawable", activity.getPackageName());
                        if (resId != 0) {
                            icon.setImageResource(resId);
                        }
                    }

                    if (model.getIconColor() != null && !model.getIconColor().isEmpty()) {
                        try {
                            icon.setColorFilter(Color.parseColor(model.getIconColor()));
                        } catch (Exception ignored) {}
                    }

                    if (model.isLegacy()) {
                        typeIcon.setImageResource(R.drawable.ic_legacy);
                        typeIcon.setColorFilter(Color.parseColor("#FF9800"));
                        typeIcon.setVisibility(View.VISIBLE);
                    } else if (model.isExperimental()) {
                        typeIcon.setImageResource(R.drawable.ic_experimental);
                        typeIcon.setColorFilter(Color.parseColor("#9C27B0"));
                        typeIcon.setVisibility(View.VISIBLE);
                    } else {
                        typeIcon.setVisibility(View.GONE);
                    }

                    ImageView downloadStatusIcon = cardView.findViewById(R.id.downloadStatusIcon);
                    if (!modelDownloader.isModelAlreadyDownloaded(model)) {
                        downloadStatusIcon.setVisibility(View.VISIBLE);
                        downloadStatusIcon.setColorFilter(Color.parseColor("#9E9E9E"));
                    } else {
                        downloadStatusIcon.setVisibility(View.GONE);
                    }

                    card.setOnClickListener(v -> onModelChipSelected(model));
                    card.setTag(model);
                    modelSelectorContainer.addView(card);

                    if (i == 0 && selectedModel == null) {
                        onModelChipSelected(model);
                    }
                }

                // Re-select current model if still available
                if (selectedModel != null) {
                    selectChipForModel(selectedModel);
                }
            });
        });
    }

    private void onModelChipSelected(InsectModel model) {
        if (callback.isUiLocked()) {
            Log.d(LOG_TAG, "Already predicting...");
            if (selectedModel != null) {
                selectChipForModel(selectedModel);
            }
            return;
        }
        if (modelDownloader != null && modelDownloader.isDownloading()) {
            callback.showMessage("Please wait for downloads to complete.");
            if (selectedModel != null) {
                selectChipForModel(selectedModel); // Keep previous selection
            }
            return;
        }
        try {
            deselectAllChipsExcept(model);
            selectedModel = model;

            activity.runOnUiThread(() -> {
                if (selectedModelName != null) {
                    selectedModelName.setText("Identify a " + selectedModel.getDisplayName());
                }
                if (selectedModel.isLegacy()) {
                    identifyModelWarning.setText("This is a legacy model, may not perform up to the mark.");
                    identifyModelWarning.setVisibility(View.VISIBLE);
                } else if (selectedModel.isExperimental()) {
                    identifyModelWarning.setText("This is an experimental model, may not perform up to the mark.");
                    identifyModelWarning.setVisibility(View.VISIBLE);
                } else {
                    identifyModelWarning.setVisibility(View.GONE);
                }
            });
            
            callback.onModelSelected(model);
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception during model chip selection", ex);
        }
    }

    private void deselectAllChipsExcept(InsectModel selected) {
        for (int i = 0; i < modelSelectorContainer.getChildCount(); i++) {
            View child = modelSelectorContainer.getChildAt(i);
            if (child instanceof MaterialCardView) {
                MaterialCardView card = (MaterialCardView) child;
                InsectModel cardModel = (InsectModel) card.getTag();
                if (cardModel != null) {
                    boolean isSelected = cardModel.getModelName().equals(selected.getModelName());
                    int defaultColor = com.google.android.material.color.MaterialColors.getColor(card, com.google.android.material.R.attr.colorOutlineVariant);
                    card.setStrokeColor(isSelected ? ContextCompat.getColor(activity, R.color.primaryGreen) : defaultColor);
                    card.setStrokeWidth(isSelected ? 4 : 1);
                }
            }
        }
    }

    public void selectChipForModel(InsectModel model) {
        deselectAllChipsExcept(model);
    }

    public void setChipGroupsEnabled(boolean enabled) {
        for (int i = 0; i < modelSelectorContainer.getChildCount(); i++) {
            modelSelectorContainer.getChildAt(i).setEnabled(enabled);
        }
    }
}
