# 16KB Page Alignment Patcher for Android 15+

Starting in Android 15, apps submitted to the Play Store with native libraries (`.so` files) must have their `PT_LOAD` segments physically aligned to a 16KB (16384 bytes) boundary. 

Many standard third-party libraries (like older versions of PyTorch and Torchvision) are aligned to the legacy 4KB boundary and will cause `UnsatisfiedLinkError` or Play Store rejection.

## What this script does
This script takes a pre-compiled Android Archive (`.aar`), unpacks it, recursively finds all `.so` files (across all architectures like `arm64-v8a`, `x86_64`, etc.), and uses `lief` to re-align their `LOAD` segments to 16384 bytes. Finally, it packages everything back into an `.aar`.

## Prerequisites
You need Python and the `lief` package installed.

```bash
pip install lief
```

## Usage
Run the script passing the original `.aar` file and the desired output path:

```bash
python patch_16kb.py original_library.aar patched_library.aar
```

## Integrating into Gradle
If you use patched local `.aar` files in your project, simply place them in your `app/libs` directory and update your `build.gradle` (or `build.gradle.kts`):

```gradle
dependencies {
    implementation(files("libs/patched_library.aar"))
}
```
