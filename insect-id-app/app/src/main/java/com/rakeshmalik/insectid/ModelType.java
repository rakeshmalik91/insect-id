package com.rakeshmalik.insectid;

public enum ModelType {
    LEPIDOPTERA("Butterfly/Moth", "lepidoptera"),
    ODONATA("Dragonfly/Damselfly", "odonata"),
    BUTTERFLY("Butterfly", "butterfly"),
    MOTH("Moth", "moth"),
    CICADA("Cicada (inaccurate)", "cicada");

    public final String displayName;
    public final String modelName;

    ModelType(String displayName, String modelName) {
        this.displayName = displayName;
        this.modelName = modelName;
    }
}
