# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile

# Keep BouncyCastle classes used by OkHttp
-keep class org.bouncycastle.** { *; }
-dontwarn org.bouncycastle.**

# Keep Conscrypt classes used by OkHttp
-keep class org.conscrypt.** { *; }
-dontwarn org.conscrypt.**

# Keep OpenJSSE classes used by OkHttp
-keep class org.openjsse.** { *; }
-dontwarn org.openjsse.**

# Keep OkHttp internals (avoids accidental stripping)
-keep class okhttp3.** { *; }
-dontwarn okhttp3.**

# Keep SSLSocket-related classes (TLS configuration)
-keep class javax.net.ssl.** { *; }
-dontwarn javax.net.ssl.**

-keep class * extends org.pytorch.Module { *; }
-keep class org.pytorch.** { *; }
-dontwarn org.pytorch.**

-keep class com.facebook.jni.** { *; }
-keep class com.facebook.soloader.** { *; }

-keepclassmembers class * {
    native <methods>;
}
