plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.rakeshmalik.insectid"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.rakeshmalik.insectid"
        minSdk = 24
        targetSdk = 37
        versionCode = 17
        versionName = "v1.0.1 alpha"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64")
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
            useLegacyPackaging = true
        }
    }
    lint {
        baseline = file("lint-baseline.xml")
        abortOnError = true
    }
}


// TODO
// APK app-debug.apk is not compatible with 16 KB devices. Some libraries have LOAD segments not aligned at 16 KB boundaries:
// lib/arm64-v8a/libfbjni.so
// lib/arm64-v8a/libpytorch_jni.so
// lib/arm64-v8a/libpytorch_vision_jni.so
// Starting November 1st, 2025, all new apps and updates to existing apps submitted to Google Play and targeting Android 15+ devices must support 16 KB page sizes. For more information about compatibility with 16 KB devices, visit developer.android.com/16kb-page-size.
// https://github.com/pytorch/pytorch/issues/154449

dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
    // 16KB-aligned PyTorch local AARs
    implementation(files("libs/pytorch_android-patched.aar"))
    implementation(files("libs/pytorch_android_torchvision-patched.aar"))
    implementation("com.facebook.fbjni:fbjni-java-only:0.7.0")
    implementation("com.facebook.soloader:nativeloader:0.12.1")
    implementation(libs.gson)
    implementation(libs.ucrop)
    implementation(libs.okhttp)
    implementation(project(":opencv"))
    implementation(libs.relinker)
}
