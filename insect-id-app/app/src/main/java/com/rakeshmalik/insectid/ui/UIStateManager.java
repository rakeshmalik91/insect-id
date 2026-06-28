package com.rakeshmalik.insectid.ui;

import static com.rakeshmalik.insectid.constants.Constants.LOG_TAG;

import android.animation.ObjectAnimator;
import android.animation.ValueAnimator;
import android.graphics.Color;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.badge.BadgeDrawable;
import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.rakeshmalik.insectid.R;

public class UIStateManager {

    private final AppCompatActivity activity;
    private final BottomNavigationView bottomNavigation;
    private final View tabIdentify;
    private final View tabModels;
    private final View tabSettings;

    private ObjectAnimator downloadIconAnimator;
    private ObjectAnimator identifyIconAnimator;
    private boolean uiLocked = false;
    
    // Dependencies to disable/enable when locked
    private Button buttonPickImage;
    private Button btnDownloadAll;
    private Button btnCancelDownload;
    private ChipGroupEnabler chipGroupEnabler;

    public interface ChipGroupEnabler {
        void setChipGroupsEnabled(boolean enabled);
    }

    public UIStateManager(AppCompatActivity activity, 
                          BottomNavigationView bottomNavigation, 
                          View tabIdentify, 
                          View tabModels, 
                          View tabSettings) {
        this.activity = activity;
        this.bottomNavigation = bottomNavigation;
        this.tabIdentify = tabIdentify;
        this.tabModels = tabModels;
        this.tabSettings = tabSettings;
    }

    public void setLockableViews(Button buttonPickImage, Button btnDownloadAll, Button btnCancelDownload, ChipGroupEnabler chipGroupEnabler) {
        this.buttonPickImage = buttonPickImage;
        this.btnDownloadAll = btnDownloadAll;
        this.btnCancelDownload = btnCancelDownload;
        this.chipGroupEnabler = chipGroupEnabler;
    }

    public boolean isUiLocked() {
        return uiLocked;
    }

    public void switchToIdentifyTab() {
        activity.runOnUiThread(() -> bottomNavigation.setSelectedItemId(R.id.navigation_identify));
    }

    public void switchToModelsTab() {
        activity.runOnUiThread(() -> bottomNavigation.setSelectedItemId(R.id.navigation_models));
    }

    public void startDownloadIconAnimation() {
        try {
            BadgeDrawable badge = bottomNavigation.getOrCreateBadge(R.id.navigation_models);
            badge.setVisible(true);
            badge.setBackgroundColor(Color.parseColor("#4CAF50"));
            badge.clearNumber();

            View modelsTab = bottomNavigation.findViewById(R.id.navigation_models);
            if (modelsTab != null && downloadIconAnimator == null) {
                downloadIconAnimator = ObjectAnimator.ofFloat(modelsTab, "alpha", 1f, 0.4f);
                downloadIconAnimator.setDuration(600);
                downloadIconAnimator.setRepeatCount(ValueAnimator.INFINITE);
                downloadIconAnimator.setRepeatMode(ValueAnimator.REVERSE);
                downloadIconAnimator.start();
            }
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception animating download icon", ex);
        }
    }

    public void stopDownloadIconAnimation() {
        try {
            bottomNavigation.removeBadge(R.id.navigation_models);
            if (downloadIconAnimator != null) {
                downloadIconAnimator.cancel();
                downloadIconAnimator = null;
            }
            View modelsTab = bottomNavigation.findViewById(R.id.navigation_models);
            if (modelsTab != null) {
                modelsTab.setAlpha(1f);
            }
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception stopping download icon animation", ex);
        }
    }

    public void startIdentifyIconAnimation() {
        try {
            BadgeDrawable badge = bottomNavigation.getOrCreateBadge(R.id.navigation_identify);
            badge.setVisible(true);
            badge.setBackgroundColor(Color.parseColor("#FF9800"));
            badge.clearNumber();

            View identifyTab = bottomNavigation.findViewById(R.id.navigation_identify);
            if (identifyTab != null && identifyIconAnimator == null) {
                identifyIconAnimator = ObjectAnimator.ofFloat(identifyTab, "alpha", 1f, 0.4f);
                identifyIconAnimator.setDuration(600);
                identifyIconAnimator.setRepeatCount(ValueAnimator.INFINITE);
                identifyIconAnimator.setRepeatMode(ValueAnimator.REVERSE);
                identifyIconAnimator.start();
            }
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception animating identify icon", ex);
        }
    }

    public void stopIdentifyIconAnimation() {
        try {
            bottomNavigation.removeBadge(R.id.navigation_identify);
            if (identifyIconAnimator != null) {
                identifyIconAnimator.cancel();
                identifyIconAnimator = null;
            }
            View identifyTab = bottomNavigation.findViewById(R.id.navigation_identify);
            if (identifyTab != null) {
                identifyTab.setAlpha(1f);
            }
        } catch (Exception ex) {
            Log.e(LOG_TAG, "Exception stopping identify icon animation", ex);
        }
    }

    public synchronized void lockUI() {
        uiLocked = true;
        activity.runOnUiThread(() -> {
            if (chipGroupEnabler != null) {
                chipGroupEnabler.setChipGroupsEnabled(false);
            }
            if (buttonPickImage != null) buttonPickImage.setEnabled(false);
            if (btnDownloadAll != null) btnDownloadAll.setEnabled(false);
            if (btnCancelDownload != null) btnCancelDownload.setEnabled(true);
            startIdentifyIconAnimation();
        });
    }

    public synchronized void unlockUI() {
        uiLocked = false;
        activity.runOnUiThread(() -> {
            if (chipGroupEnabler != null) {
                chipGroupEnabler.setChipGroupsEnabled(true);
            }
            if (buttonPickImage != null) buttonPickImage.setEnabled(true);
            if (btnDownloadAll != null) btnDownloadAll.setEnabled(true);
            stopIdentifyIconAnimation();
        });
    }
}
