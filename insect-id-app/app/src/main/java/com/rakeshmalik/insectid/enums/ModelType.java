package com.rakeshmalik.insectid.enums;

public enum ModelType {
    LEPIDOPTERA("Butterfly/Moth", "lepidoptera"),
    LEPIDOPTERA_V0("Butterfly/Moth (v0)", "lepidoptera.v0"),
    BUTTERFLY("Butterfly", "butterfly"),
    MOTH("Moth", "moth"),
    ODONATA("Dragonfly/Damselfly", "odonata"),
    CICADA("Cicada", "cicada");

    public final String displayName;
    public final String modelName;

    ModelType(String displayName, String modelName) {
        this.displayName = displayName;
        this.modelName = modelName;
    }
}
