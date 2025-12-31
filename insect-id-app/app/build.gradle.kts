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
        versionCode = 7
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

    implementation(libs.gson)
    implementation(libs.ucrop)
    implementation(libs.okhttp)
    implementation(libs.relinker)

// APK app-debug.apk is not compatible with 16 KB devices. Some libraries have LOAD segments not aligned at 16 KB boundaries:
// lib/arm64-v8a/libfbjni.so
// lib/arm64-v8a/libpytorch_jni.so
// lib/arm64-v8a/libpytorch_vision_jni.so
// Starting November 1st, 2025, all new apps and updates to existing apps submitted to Google Play and targeting Android 15+ devices must support 16 KB page sizes. For more information about compatibility with 16 KB devices, visit developer.android.com/16kb-page-size.
// https://github.com/pytorch/pytorch/issues/154449
//    implementation(libs.pytorch.android)
//    implementation(libs.pytorch.android.torchvision)
    implementation("com.facebook.fbjni:fbjni-java-only:0.2.2")
    implementation("com.facebook.soloader:nativeloader:0.10.5")
//    implementation("com.facebook.soloader:nativeloader:0.12.1")
    implementation(files("$rootDir/libs/pytorch_android-release.aar"))
    implementation(files("$rootDir/libs/pytorch_android_torchvision-release.aar"))

    implementation(project(":opencv"))
}
