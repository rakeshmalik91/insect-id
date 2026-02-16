package com.rakeshmalik.insectid.pojo;

import static com.rakeshmalik.insectid.constants.Constants.ROOT_CLASSIFIER;

import androidx.annotation.NonNull;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class InsectModel implements Comparable<InsectModel> {

    private final String modelName;
    private final String displayName;
    private final String description;
    private final boolean legacy;
    private final boolean prototype;
    private final boolean enabled;
    private final String classesUrl;
    private final String classDetailsUrl;
    private final String modelUrl;
    private final String imagesUrl;
    private final int version;
    private final long size;
    private final double minAcceptedLogit;
    private final double minAcceptedSoftmax;
    private final double minAcceptedSoftmaxToOverrideRootClassifier;
    private final List<String> classes;
    private final List<String> acceptedClasses;
    private final ModelStats stats;
    private final boolean isRoot;

    public InsectModel(String modelName, String displayName, String description, boolean legacy, boolean prototype, boolean enabled,
                       String classesUrl, String classDetailsUrl, String modelUrl, String imagesUrl,
                       int version, long size, double minAcceptedLogit, double minAcceptedSoftmax,
                       double minAcceptedSoftmaxToOverrideRootClassifier, List<String> classes,
                       List<String> acceptedClasses, ModelStats stats, boolean isRoot) {
        this.modelName = modelName;
        this.displayName = displayName;
        this.description = description;
        this.legacy = legacy;
        this.prototype = prototype;
        this.enabled = enabled;
        this.classesUrl = classesUrl;
        this.classDetailsUrl = classDetailsUrl;
        this.modelUrl = modelUrl;
        this.imagesUrl = imagesUrl;
        this.version = version;
        this.size = size;
        this.minAcceptedLogit = minAcceptedLogit;
        this.minAcceptedSoftmax = minAcceptedSoftmax;
        this.minAcceptedSoftmaxToOverrideRootClassifier = minAcceptedSoftmaxToOverrideRootClassifier;
        this.classes = classes;
        this.acceptedClasses = acceptedClasses;
        this.stats = stats;
        this.isRoot = isRoot;
    }

    public static InsectModel fromJson(String modelName, JSONObject json) {
        String displayName = json.optString("name", modelName);
        String description = json.optString("description", null);
        boolean legacy = json.optBoolean("legacy", false);
        boolean prototype = json.optBoolean("prototype", false);
        boolean enabled = json.optBoolean("enabled", true);
        String classesUrl = json.optString("classes_url", null);
        String classDetailsUrl = json.optString("class_details_url", null);
        String modelUrl = json.optString("model_url", null);
        String imagesUrl = json.optString("images_url", null);
        int version = json.optInt("version", 0);
        long size = json.optLong("size", 0);
        double minAcceptedLogit = json.optDouble("min_accepted_logit");
        double minAcceptedSoftmax = json.optDouble("min_accepted_softmax");
        double minAcceptedSoftmaxToOverrideRootClassifier = json.optDouble("min_accepted_softmax_to_override_root_classifier");
        
        List<String> classes = toList(json.optJSONArray("classes"));
        List<String> acceptedClasses = toList(json.optJSONArray("accepted_classes"));
        ModelStats stats = ModelStats.fromJson(json.optJSONObject("stats"));
        boolean isRoot = json.optBoolean("is_root", false);

        return new InsectModel(modelName, displayName, description, legacy, prototype, enabled, classesUrl, classDetailsUrl,
                modelUrl, imagesUrl, version, size, minAcceptedLogit, minAcceptedSoftmax,
                minAcceptedSoftmaxToOverrideRootClassifier, classes, acceptedClasses, stats, isRoot);
    }
    
    private static List<String> toList(JSONArray array) {
        List<String> list = new ArrayList<>();
        if (array != null) {
            for (int i = 0; i < array.length(); i++) {
                list.add(array.optString(i));
            }
        }
        return list;
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
    
    public String getDescription() {
        return description;
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
        return isRoot || ROOT_CLASSIFIER.equals(modelName);
    }
    
    public String getClassesUrl() {
        return classesUrl;
    }

    public String getClassDetailsUrl() {
        return classDetailsUrl;
    }

    public String getModelUrl() {
        return modelUrl;
    }

    public String getImagesUrl() {
        return imagesUrl;
    }

    public int getVersion() {
        return version;
    }

    public long getSize() {
        return size;
    }

    public double getMinAcceptedLogit() {
        return minAcceptedLogit;
    }

    public double getMinAcceptedSoftmax() {
        return minAcceptedSoftmax;
    }

    public double getMinAcceptedSoftmaxToOverrideRootClassifier() {
        return minAcceptedSoftmaxToOverrideRootClassifier;
    }

    public List<String> getClasses() {
        return classes;
    }

    public List<String> getAcceptedClasses() {
        return acceptedClasses;
    }

    public ModelStats getStats() {
        return stats;
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

    public static class ModelStats {
        private final long classCount;
        private final long speciesCount;
        private final long sppClassCount;
        private final long earlyStageClassCount;
        private final long dataCount;
        private final String modelArch;
        private final List<String> dataSources;
        private final String lastUpdatedDate;
        private final String accuracy;
        private final String accuracyTop3;

        public ModelStats(long classCount, long speciesCount, long sppClassCount, long earlyStageClassCount,
                          long dataCount, String modelArch, List<String> dataSources, String lastUpdatedDate,
                          String accuracy, String accuracyTop3) {
            this.classCount = classCount;
            this.speciesCount = speciesCount;
            this.sppClassCount = sppClassCount;
            this.earlyStageClassCount = earlyStageClassCount;
            this.dataCount = dataCount;
            this.modelArch = modelArch;
            this.dataSources = dataSources;
            this.lastUpdatedDate = lastUpdatedDate;
            this.accuracy = accuracy;
            this.accuracyTop3 = accuracyTop3;
        }

        public static ModelStats fromJson(JSONObject json) {
            if (json == null) return null;
            long classCount = json.optLong("class_count", 0);
            long speciesCount = json.optLong("species_count", 0);
            long sppClassCount = json.optLong("spp_class_count", 0);
            long earlyStageClassCount = json.optLong("early_stage_class_count", 0);
            long dataCount = json.optLong("data_count", 0);
            String modelArch = json.optString("model_arch", null);
            List<String> dataSources = toList(json.optJSONArray("data_sources"));
            String lastUpdatedDate = json.optString("last_updated_date", null);
            String accuracy = json.optString("accuracy", null);
            String accuracyTop3 = json.optString("accuracy_top3", null);

            return new ModelStats(classCount, speciesCount, sppClassCount, earlyStageClassCount, dataCount,
                    modelArch, dataSources, lastUpdatedDate, accuracy, accuracyTop3);
        }

        public long getClassCount() {
            return classCount;
        }

        public long getSpeciesCount() {
            return speciesCount;
        }

        public long getSppClassCount() {
            return sppClassCount;
        }

        public long getEarlyStageClassCount() {
            return earlyStageClassCount;
        }

        public long getDataCount() {
            return dataCount;
        }

        public String getModelArch() {
            return modelArch;
        }

        public List<String> getDataSources() {
            return dataSources;
        }

        public String getLastUpdatedDate() {
            return lastUpdatedDate;
        }

        public String getAccuracy() {
            return accuracy;
        }

        public String getAccuracyTop3() {
            return accuracyTop3;
        }
    }

}
