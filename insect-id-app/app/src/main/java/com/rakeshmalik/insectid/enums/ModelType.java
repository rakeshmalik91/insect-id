package com.rakeshmalik.insectid.enums;

public enum ModelType {
    LEPIDOPTERA_V2ALPHA("Butterfly/Moth", "lepidoptera.v2alpha", false),
    ODONATA("Dragonfly/Damselfly", "odonata", false),
    CICADA("Cicada", "cicada", false),
    LEPIDOPTERA("Butterfly/Moth", "lepidoptera", true),
    BUTTERFLY("Butterfly", "butterfly", true),
    MOTH("Moth", "moth", true);

    private final String displayName;
    public final String modelName;
    public final boolean legacy;

    public String getModelDisplayName() {
        return displayName + (legacy ? " (legacy)" : "");
    }

    public String getIdentificationTypeDisplayName() {
        return displayName;
    }

    ModelType(String displayName, String modelName, boolean legacy) {
        this.displayName = displayName;
        this.modelName = modelName;
        this.legacy = legacy;
    }
}
