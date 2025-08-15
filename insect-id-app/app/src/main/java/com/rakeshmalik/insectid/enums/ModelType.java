package com.rakeshmalik.insectid.enums;

public enum ModelType {
//    LEPIDOPTERA("Butterfly/Moth", "lepidoptera"),
    LEPIDOPTERA_V2ALPHA("Butterfly/Moth", "lepidoptera.v2alpha"),
//    BUTTERFLY("Butterfly", "butterfly"),
//    MOTH("Moth", "moth"),
    ODONATA("Dragonfly/Damselfly", "odonata"),
    CICADA("Cicada", "cicada");

    public final String displayName;
    public final String modelName;

    ModelType(String displayName, String modelName) {
        this.displayName = displayName;
        this.modelName = modelName;
    }
}
