package com.rakeshmalik.insectid;

import java.util.Map;

public class Constants {

    public static final int MAX_PREDICTIONS = 10;
    public static final String MODEL_FILE_NAME_FMT = "m.checkpoint.%s.pt";
    public static final String CLASSES_FILE_NAME_FMT = "classes.%s.json";
    public static final String CLASS_DETAILS_FILE_NAME_FMT = "class_details.%s.json";

    // class suffix -> display name
    public static final Map<String, String> CLASS_SUFFIXES = Map.of(
            "-early", " (Early Stage)",
            "-larva", " (Larva)",
            "-pupa", " (Pupa)"
    );

    public static final String NAME = "name";
    public static final String LOG_TAG = "insect-id";
    public static final String PREF = "insect-id";

    public static final String PREF_FILE_DOWNLOADED = PREF + "::file-downloaded";
    public static final String PREF_MODEL_VERSION = PREF + "::version";
    public static final String PREF_METADATA = PREF + "::metadata";
    public static final String PREF_ASSET_TEMP_PATH = PREF + "::asset-temp-path";

    public static final String METADATA_URL = "https://raw.githubusercontent.com/rakeshmalik91/insect-id/refs/heads/main/insect-id-app/metadata.json";

    public static final String FIELD_CLASSES_URL = "classes_url";
    public static final String FIELD_CLASS_DETAILS_URL = "class_details_url";
    public static final String FIELD_MODEL_URL = "model_url";
    public static final String FIELD_VERSION = "version";
    public static final String FIELD_MIN_ACCEPTED_LOGIT = "min_accepted_logit";
    public static final String FIELD_MIN_ACCEPTED_SOFTMAX = "min_accepted_softmax";
    public static final String FIELD_CLASSES = "classes";
    public static final String FIELD_ACCEPTED_CLASSES = "accepted_classes";
    public static final String FIELD_ASSET_PATH = "asset_path";

    public static final String ROOT_CLASSIFIER = "root-classifier";
    public static final String ROOT_CLASS_OTHER_INSECT = "other-insect";

}
