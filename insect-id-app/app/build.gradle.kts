plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.rakeshmalik.insectid"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.rakeshmalik.insectid"
        minSdk = 24
        targetSdk = 35
        versionCode = 6
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    packaging {
        jniLibs {
            // conflicts between opencv and pytorch
            pickFirsts.add("lib/**/libc++_shared.so")
        }
    }
}

dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
    implementation(libs.pytorch.android)
    implementation(libs.pytorch.android.torchvision)
    implementation(libs.gson)
    implementation(libs.ucrop)
    implementation(libs.okhttp)
    implementation(project(":opencv"))
    implementation(libs.relinker)
}
