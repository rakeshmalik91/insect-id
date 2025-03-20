package com.rakeshmalik.insectid.constants;

public class Constants {

    public static final int MAX_PREDICTIONS = 10;
    public static final String MODEL_FILE_NAME_FMT = "m.checkpoint.%s.pt";
    public static final String IMAGES_ARCHIVE_FILE_NAME_FMT = "images.%s.zip";
    public static final String CLASSES_FILE_NAME_FMT = "classes.%s.json";
    public static final String CLASS_DETAILS_FILE_NAME_FMT = "class_details.%s.json";

    public static final String EARLY_STAGE_CLASS_SUFFIX = "-early";
    public static final String EARLY_STAGE_DISPLAY_SUFFIX = " (Early Stage)";

    public static final String NAME = "name";
    public static final String IMAGES = "images";
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
    public static final String FIELD_IMAGES_URL = "images_url";
    public static final String FIELD_VERSION = "version";
    public static final String FIELD_MIN_ACCEPTED_LOGIT = "min_accepted_logit";
    public static final String FIELD_MIN_ACCEPTED_SOFTMAX = "min_accepted_softmax";
    public static final String FIELD_MIN_ACCEPTED_SOFTMAX_TO_OVERRIDE_ROOT_CLASSIFIER = "min_accepted_softmax_to_override_root_classifier";
    public static final String FIELD_CLASSES = "classes";
    public static final String FIELD_ACCEPTED_CLASSES = "accepted_classes";
    public static final String FIELD_STATS = "stats";

    public static final String ROOT_CLASSIFIER = "root-classifier";
    public static final String ROOT_CLASS_OTHER = "other";

    public static final int MAX_IMAGES_IN_PREDICTION = 6;

    public static final int PREVIEW_IMAGE_TIMEOUT = 10000;
    public static final int MODEL_LOAD_TIMEOUT = 60000;

    public static final String HTML_NO_IMAGE_AVAILABLE = "<font color='#777777'>(No images available)</font><br/><br/>";
    public static final String HTML_LOADING_IMAGES = "<font color='#777777'>(Loading images...)</font><br/><br/>";

    public static final String WAKE_LOCK_NAME = "insect-id::DownloadLock";
    public static final long WAKE_LOCK_TIME = 10 * 60 * 1000L; /*10 minutes*/

}
