:: mkdir tools
:: wget https://github.com/google/bundletool/releases/download/1.15.6/bundletool-all-1.15.6.jar -O tools/bundletool.jar

set ANDROID_HOME=%USERPROFILE%\AppData\Local\Android\Sdk
set PATH=%PATH%;%ANDROID_HOME%\platform-tools

:: call gradlew :app:clean
:: call gradlew :app:bundleRelease

:: adb uninstall com.rakeshmalik.insectid

del app-release.apks
java -jar tools\bundletool.jar build-apks ^
  --bundle=app\release\app-release.aab ^
  --output=app-release.apks ^
  --ks=%USERPROFILE%\.android\debug.keystore ^
  --ks-key-alias=androiddebugkey ^
  --ks-pass=pass:android ^
  --key-pass=pass:android ^
  --mode=universal

adb devices

java -jar tools\bundletool.jar install-apks --apks=app-release.apks --adb="%ANDROID_HOME%\platform-tools\adb.exe"

adb shell cmd package resolve-activity --brief com.rakeshmalik.insectid

adb shell am start -n com.rakeshmalik.insectid/.MainActivity
