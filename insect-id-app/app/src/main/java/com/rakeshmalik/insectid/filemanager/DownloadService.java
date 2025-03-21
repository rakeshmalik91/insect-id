//package com.rakeshmalik.insectid.filemanager;
//
//import android.app.Notification;
//import android.app.NotificationChannel;
//import android.app.NotificationManager;
//import android.app.Service;
//import android.content.Intent;
//import android.os.Build;
//import android.os.PowerManager;
//import android.os.IBinder;
//
//import androidx.core.app.NotificationCompat;
//
//import com.rakeshmalik.insectid.constants.Constants;
//
//public class DownloadService extends Service {
//
//    public static final String CHANNEL_ID = "DownloadChannel";
//
//    private PowerManager.WakeLock wakeLock;
//
//    @Override
//    public void onCreate() {
//        super.onCreate();
//        createNotificationChannel();
//
//        PowerManager powerManager = (PowerManager) getSystemService(POWER_SERVICE);
//        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, Constants.WAKE_LOCK_NAME);
//        wakeLock.acquire(Constants.WAKE_LOCK_TIME);
//
//        startForeground(1, getNotification());
//    }
//
//    @Override
//    public int onStartCommand(Intent intent, int flags, int startId) {
//        // TODO here
//
////        stopSelf();
//        return START_NOT_STICKY;
//    }
//
//    @Override
//    public void onDestroy() {
//        super.onDestroy();
//        if (wakeLock != null && wakeLock.isHeld()) {
//            wakeLock.release();
//        }
//    }
//
//    @Override
//    public IBinder onBind(Intent intent) {
//        return null;
//    }
//
//    private void createNotificationChannel() {
//        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
//            NotificationChannel channel = new NotificationChannel(
//                    CHANNEL_ID,
//                    "Download Service",
//                    NotificationManager.IMPORTANCE_LOW
//            );
//            NotificationManager manager = getSystemService(NotificationManager.class);
//            if (manager != null) {
//                manager.createNotificationChannel(channel);
//            }
//        }
//    }
//
//    private Notification getNotification() {
//        return new NotificationCompat.Builder(this, CHANNEL_ID)
//                .setContentTitle("Downloading Model")
//                .setContentText("Your model is being downloaded...")
//                .setSmallIcon(android.R.drawable.stat_sys_download)
//                .build();
//    }
//}
