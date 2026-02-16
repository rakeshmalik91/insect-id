package com.rakeshmalik.insectid.pojo;

import static com.rakeshmalik.insectid.constants.Constants.ROOT_CLASSIFIER;

import androidx.annotation.NonNull;

import org.json.JSONObject;

import java.util.Objects;
import com.rakeshmalik.insectid.constants.Constants;

public class InsectModel implements Comparable<InsectModel> {

    private final String modelName;
    private final String displayName;
    private final boolean legacy;
    private final boolean prototype;
    private final boolean enabled;

    public InsectModel(String modelName, String displayName, boolean legacy, boolean prototype, boolean enabled) {
        this.modelName = modelName;
        this.displayName = displayName;
        this.legacy = legacy;
        this.prototype = prototype;
        this.enabled = enabled;
    }

    public static InsectModel fromJson(String modelName, JSONObject json) {
        String displayName = json.optString(Constants.FIELD_NAME, modelName);
        boolean legacy = json.optBoolean(Constants.FIELD_LEGACY, false);
        boolean prototype = json.optBoolean(Constants.FIELD_PROTOTYPE, false);
        boolean enabled = json.optBoolean(Constants.FIELD_ENABLED, true);
        return new InsectModel(modelName, displayName, legacy, prototype, enabled);
    }

    public String getModelName() {
        return modelName;
    }

    public String getDisplayName() {
        return displayName + (legacy ? " (legacy)" : "") + (prototype ? " (prototype)" : "");
    }
    
    public String getRawDisplayName() {
        return displayName;
    }

    public boolean isLegacy() {
        return legacy;
    }

    public boolean isPrototype() {
        return prototype;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public boolean isRootClassifier() {
        return ROOT_CLASSIFIER.equals(modelName);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        InsectModel that = (InsectModel) o;
        return Objects.equals(modelName, that.modelName);
    }

    @Override
    public int hashCode() {
        return Objects.hash(modelName);
    }

    @NonNull
    @Override
    public String toString() {
        return getDisplayName();
    }

    @Override
    public int compareTo(InsectModel o) {
        // Sort non-legacy first, then by name
        if (this.legacy && !o.legacy) return 1;
        if (!this.legacy && o.legacy) return -1;
        return this.displayName.compareToIgnoreCase(o.displayName);
    }
}
