package com.rakeshmalik.insectid.enums;

public enum ModelType {
    LEPIDOPTERA_V2ALPHA("Butterfly/Moth", "lepidoptera.v2alpha", false),
    ODONATA("Dragonfly/Damselfly", "odonata", false),
    CICADA("Cicada", "cicada", false),
    LEPIDOPTERA("Butterfly/Moth (legacy model)", "lepidoptera", true),
    BUTTERFLY("Butterfly (legacy model)", "butterfly", true),
    MOTH("Moth (legacy model)", "moth", true);

    public final String displayName;
    public final String modelName;
    public final boolean legacy;

    ModelType(String displayName, String modelName, boolean legacy) {
        this.displayName = displayName;
        this.modelName = modelName;
        this.legacy = legacy;
    }
}
