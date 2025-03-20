package com.rakeshmalik.insectid.enums;

public enum ModelType {
    LEPIDOPTERA("Butterfly/Moth", "lepidoptera"),
    ODONATA("Dragonfly/Damselfly", "odonata"),
    CICADA("Cicada", "cicada"),
    BUTTERFLY("Butterfly", "butterfly"),
    MOTH("Moth", "moth");

    public final String displayName;
    public final String modelName;

    ModelType(String displayName, String modelName) {
        this.displayName = displayName;
        this.modelName = modelName;
    }
}
