@echo off

:: wsl bash -c "mkdir tools"
:: wsl bash -c "wget https://github.com/google/bundletool/releases/download/1.15.6/bundletool-all-1.15.6.jar -O /mnt/d/Projects/insect-id/insect-id-app/tools/bundletool.jar"

set ANDROID_HOME=%USERPROFILE%\AppData\Local\Android\Sdk
set PATH=%PATH%;%ANDROID_HOME%\platform-tools

echo "Building AAB..."
call gradlew :app:clean
call gradlew :app:bundleRelease
set AAB_PATH=app\build\outputs\bundle\release\app-release.aab

echo "Verifying AAB contents for libpytorch..."
set AAB_WSL_PATH=/mnt/d/Projects/insect-id/insect-id-app/app/build/outputs/bundle/release/app-release.aab
wsl bash -c "unzip -l %AAB_WSL_PATH% | grep libpytorch"

echo "Build APKs..."
del app-release.apks
:: adb uninstall com.rakeshmalik.insectid
java -jar tools\bundletool.jar build-apks ^
  --bundle=%AAB_PATH% ^
  --output=app-release.apks ^
  --ks=%USERPROFILE%\.android\debug.keystore ^
  --ks-key-alias=androiddebugkey ^
  --ks-pass=pass:android ^
  --key-pass=pass:android ^
  --mode=universal

echo "Installing APKs on connected devices..."
adb devices
java -jar tools\bundletool.jar install-apks --apks=app-release.apks --adb="%ANDROID_HOME%\platform-tools\adb.exe"
adb shell cmd package resolve-activity --brief com.rakeshmalik.insectid
adb shell am start -n com.rakeshmalik.insectid/.MainActivity
