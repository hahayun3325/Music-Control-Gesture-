package com.example.finalhand1.musiccontrol;

import static java.lang.Math.log;
import static java.lang.Math.max;
import static java.lang.Math.min;

import android.annotation.SuppressLint;
import android.app.Notification;
import android.content.ComponentName;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.SurfaceTexture;
import android.graphics.drawable.Icon;
import android.hardware.SensorEvent;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.provider.MediaStore;
import android.service.notification.StatusBarNotification;
import android.text.Layout;
import android.util.Log;
import android.util.Size;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.coordinatorlayout.widget.CoordinatorLayout;
import androidx.core.app.NotificationManagerCompat;
import androidx.preference.PreferenceManager;

import com.example.finalhand1.BrightnessAdapter;
import com.example.finalhand1.R;
import com.example.finalhand1.RecHands;

import static com.example.finalhand1.RecHands.*;
import static com.example.finalhand1.musiccontrol.NotifyHelper.*;
import static com.example.finalhand1.musiccontrol.NotifyService.*;

import com.example.finalhand1.mainmenuUI.MainMenu;
import com.example.finalhand1.mainmenuUI.MenuAdapter;
import com.google.android.material.snackbar.Snackbar;
import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.CameraXPreviewHelper;
import com.google.mediapipe.components.ExternalTextureConverter;
import com.google.mediapipe.components.FrameProcessor;
import com.google.mediapipe.components.PermissionHelper;
import com.google.mediapipe.formats.proto.LandmarkProto;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.AndroidPacketCreator;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.glutil.EglManager;

import org.checkerframework.checker.units.qual.C;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.sql.Time;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.Timer;
import java.util.concurrent.TimeUnit;


public class MusicControl extends AppCompatActivity implements SharedPreferences.OnSharedPreferenceChangeListener, NotifyListener {

    // 资源文件和流输出名
    private static final String BINARY_GRAPH_NAME = "hand_tracking_mobile_gpu.binarypb";
    private static final String INPUT_VIDEO_STREAM_NAME = "input_video";
    private static final String OUTPUT_VIDEO_STREAM_NAME = "output_video";
    private static final String OUTPUT_HAND_PRESENCE_STREAM_NAME = "hand_presence";
    private static final String OUTPUT_LANDMARKS_STREAM_NAME = "hand_landmarks";
    private static final String INPUT_NUM_HANDS_SIDE_PACKET_NAME = "num_hands";
    private static final int NUM_HANDS = 1;

    private SurfaceTexture previewFrameTexture;
    private SurfaceView previewDisplayView;
    private EglManager eglManager;
    private FrameProcessor processor;
    private ExternalTextureConverter converter;
    private CameraXPreviewHelper cameraHelper;

    private int next_song = 0;
    private int previous_song = 0;
    private int play_music = 0;
    private int pause_music = 0;

    long timestampbefor = -1;
    long timespend = 0;
    boolean flag = true;

    int k = 10;   //the number frame to do a act
    int numm = 0; //the total frame hand detection
    Float preVex = Float.valueOf(-1); // the prior hand x position
    Float handDistance = Float.valueOf(0);
    String currentView = "image_room";
    int songphotochangeflag = 0;

    private static final int numThreads = 1;
    // 所使用的摄像头
    private static final boolean USE_FRONT_CAMERA = true;

    // 因为OpenGL表示图像时假设图像原点在左下角，而MediaPipe通常假设图像原点在左上角，所以要翻转
    private static final boolean FLIP_FRAMES_VERTICALLY = true;

    /**
     * 以下是tflite模型使用到的参数
     */
//    private Classifier classifier;//tflite 模型分类器
    List<float[]> point_history = new ArrayList<>();//tflite 指尖动态手势识别存储手掌滑动的关键点
    List<Integer> finger_gesture_histry = new ArrayList<Integer>();//记录动态手势的历史识别结果
    private int history_length = 20;//tflite 按照模型训练的参数记录指尖的序列长度
    private int finger_gesture_id = 2;//默认情况下识别的动态手势是static
    private int point_history_len = 0;
    private int most_common_fg_id = 2;//记录历史识别序列中出现次数最多的手势编号
    float[] pre_processed_point_history_list = new float[history_length * 2];// tflite记录经过预处理之后的点坐标列表 作为一个一维向量也是直接送入模型输入层
    private int myInternal = 3;// 动态手势的预处理中 代表每隔五个手势进行一次采样，指尖坐标收集停止的标志是收集结果累积到myInternak*history时候

    // 加载动态库
    static {
        System.loadLibrary("mediapipe_jni");
        System.loadLibrary("opencv_java3");
    }



    private void send_message_to_bundle(String message1) {
        Bundle bundle = new Bundle();
        bundle.putString("action", message1);
        Message message = new Message();
        message.setData(bundle);
        toastHandler.sendMessage(message);
    }

    int flag_activity = 0;
    int finish_app = 0;
    private int array_to_count_activity[] = new int[20];

    HashMap string_to_num_map = new HashMap();

    private float prev_changeVol_x=0;// 记录拇指调节音量过程中前一个状态收手指对应的位置
    private int prevEqnowTimes=0; // 记录当前拇指指尖位置和前一个时刻相同的次数
    private boolean smallVolChange=false;// 判断音量细致调节与否的标志
    private int currvol_onchange=0;// 记录系统当前音量
    //用手势进行切歌
    private void HandleTheHands(List<LandmarkProto.NormalizedLandmarkList> multiHandLandmarks) {
        Runtime runtime = Runtime.getRuntime();
        String title = RecHands.handGestureDiscriminator_Pro(multiHandLandmarks);
//        Log.d(myTAG, "musicontrol当前输出的手势"+title);
        int upper_limit = 25;

        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);

        String previous_song_option = prefs.getString("play_previous_song", "0");
        if ("0".equals(previous_song_option)) {
            string_to_num_map.put("handToLeft", 1);

        } else if ("1".equals(previous_song_option)) {
            string_to_num_map.put("one", 1);
        }

        String next_song_option = prefs.getString("play_next_song", "0");
        if ("0".equals(next_song_option)) {
            string_to_num_map.put("handToRight", 2);

        } else if ("1".equals(next_song_option)) {
            string_to_num_map.put("TWO", 2);
        }

        String stop_song_option = prefs.getString("stop_the_song", "0");
        if ("0".equals(stop_song_option)) {
            string_to_num_map.put("handStopMusic", 3);

        } else if ("1".equals(stop_song_option)) {
            string_to_num_map.put("fist", 3);
        }

        String play_song_option = prefs.getString("play_the_song", "0");
        if ("0".equals(play_song_option)) {
            string_to_num_map.put("OK", 4);

        } else if ("1".equals(play_song_option)) {
            string_to_num_map.put("THREE", 4);
        }
        // 执行退出手势
        string_to_num_map.put("FINISHSIGN", 5);

        string_to_num_map.put("TWO", 6);
//        Log.d(myTAG, "musicControl:当前手势 "+title);
        /*
        音量调节部分 使用拇指的移动实现音量的调节
         */
        if (title.equals("one")) {
            LandmarkProto.NormalizedLandmarkList landmarks = multiHandLandmarks.get(multiHandLandmarks.size() - 1);
            Float xx = landmarks.getLandmark(5).getX();
            Float y8 = landmarks.getLandmark(8).getY();
            Float y5 = landmarks.getLandmark(5).getY();
            Float y12 = landmarks.getLandmark(12).getY();
            Float y9 = landmarks.getLandmark(9).getY();
            Float x8=landmarks.getLandmark(8).getX();
            // 将坐标按照屏幕的长度进行映射
            xx*=image_width;
            x8*=image_width;
            Float changeVol_x=(xx+x8)/2; // 计算拇指横坐标的平均值
            // 获得设备音量调节权限
            Context context = getBaseContext();
            AudioManager mAudioManager = (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
//            // 对于记录的前一个状态的拇指位置进行更新
//            if (prevEqnowTimes==0){
//                prev_changeVol_x=0;
//            }
            if(!smallVolChange){
                // : 9/8/2022  音量微调节部分被注释掉
                // 经过实验，横坐标的平均值范围(0,600)
//            Log.d(myTAG, "当前拇指的情况:"+changeVol_x+" 前一个状态坐标情况"+prev_changeVol_x);
                if(Math.abs(changeVol_x-prev_changeVol_x)<=5){
                    // 当拇指当前的坐标和前一个状态对应的坐标变化在一定范围内
                    prevEqnowTimes++;// 记录当前拇指位置和前一个状态拇指位置相同(在一定范围内)的次数
                    Log.d(myTAG, "当前相同位置的次数: "+prevEqnowTimes);
//                    if (prevEqnowTimes==20){
//                        // 进行音量调节
//                        Log.d(myTAG, "进入到音量微调节阶段");
//                        send_text_to_bundle("进入到音量微调节 调节范围是原来的20%");// 使用系统回调显示弹窗情况
//                        smallVolChange=true;
//                        currvol_onchange=mAudioManager.getStreamVolume(AudioManager.STREAM_MUSIC);// 记录系统当前音量，在此基础上进行相关调节
//                        prevEqnowTimes=0;
//                        prev_changeVol_x=0;
//                    }
                }
                else {
                        prev_changeVol_x = changeVol_x;// 记录拇指横坐标位置信息
//                    Log.d(myTAG, "前一个状态的坐标: " + prev_changeVol_x);
                        prevEqnowTimes = 0;
                        // 进行音量粗调节
                        int setVol = 0;
                        int progressVol=0;
                        if ("Xiaomi".equals(android.os.Build.MANUFACTURER)) {
                            // 实验得到小木系统音量范围(0,150)
                            setVol = (int) (changeVol_x * 150 / 600);// 当前设置的音量情况
                            progressVol=(int) (setVol*100/150);
                        } else if ("OPPO".equals(android.os.Build.MANUFACTURER)) {
                            // 实验得到小木系统音量范围(0,16)
                            setVol = (int) (changeVol_x * 16 / 600);// 当前设置的音量情况
                            progressVol=(int) (setVol*100/16);
                        } else {
                            // 对于其他类型的手机
                            setVol = (int) (changeVol_x * 100 / 600);// 当前设置的音量情况
                            progressVol=(int) (setVol*100/100);
                        }
//                Log.d(myTAG, "手机型号: "+android.os.Build.MANUFACTURER);
                        change_volume_progress(progressVol);// 修改音量进度条 将坐标标准化至100范围内
                        int currVol = mAudioManager.getStreamVolume(AudioManager.STREAM_MUSIC); // 获得当前系统对应的音量情况
//                    Log.d(myTAG, "当前系统音量情况: " + currVol);
                        mAudioManager.setStreamVolume(AudioManager.STREAM_MUSIC, //音量类型
                                setVol,//设置音量为手指之间的距离
                                AudioManager.FLAG_PLAY_SOUND
                                        | AudioManager.FLAG_SHOW_UI);
                    }
            }else if(smallVolChange){
            // : 9/8/2022  微调节部分被注释掉
//                // 进入到音量细调节中 在缩小0.5倍范围内在原本音量附近进行调节
//                prevEqnowTimes=0;
////                currvol_onchange=75;
//                Log.d(myTAG, "当前坐标位置: "+changeVol_x);
//                // 进行音量调节
//                int currVol = mAudioManager.getStreamVolume(AudioManager.STREAM_MUSIC); // 获得当前系统对应的音量情况
//                int setVol=0;
//                int progressVol=0;//用来展示在进度条上的音量情况
//                if("Xiaomi".equals(android.os.Build.MANUFACTURER)){
//                    // 实验得到小木系统音量范围(0,150)
//                    setVol = (int) ((changeVol_x-300)*150 / (600*5))+currvol_onchange;// 当前设置的音量情况
//                    progressVol=currVol*100/150;
//                }else if("OPPO".equals(android.os.Build.MANUFACTURER)){
//                    // 实验得到小木系统音量范围(0,16)
//                    setVol = (int) ((changeVol_x-300)*16 / (600*2)+currvol_onchange);// 当前设置的音量情况
//                    progressVol=currVol*100/16;
//                }else{
//                    // 对于其他类型的手机
//                    setVol = (int) ((changeVol_x-300)*100 / 6000)+currvol_onchange;// 当前设置的音量情况
//                    progressVol=setVol*100/100;
//                }
//                Log.d(myTAG, "设置音量 "+setVol);
////                Log.d(myTAG, "手机型号: "+android.os.Build.MANUFACTURER);
//                change_volume_progress(progressVol);// 修改音量进度条 将坐标标准化至100范围内
////                Log.d(myTAG, "当前系统音量情况: " + currVol);
//                mAudioManager.setStreamVolume(AudioManager.STREAM_MUSIC, //音量类型
//                        setVol,//设置音量为手指之间的距离
//                        AudioManager.FLAG_PLAY_SOUND
//                                | AudioManager.FLAG_SHOW_UI);
//                // 其中 退出音量细调节，退出one手势
            }

//            if (y8 < y5 && y12 > y9) {
//                // 满足拇指伸展并且中指闭合的条件
//                int Intxx;
//                // 获得设备音量调节权限
//                Context context = getBaseContext();
//                AudioManager mAudioManager = (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
//                // 进行音量调节
//                if ("Xiaomi".equals(android.os.Build.MANUFACTURER)) {
//                    // 对于小米系统
//                    Intxx = max((int) (xx * 20) - 3, 0);
//                    //调节进度条
//                    change_volume_progress(Intxx * 10);
//                    mAudioManager.setStreamVolume(AudioManager.STREAM_MUSIC, //音量类型
//                            Intxx * 10,//设置音量为手指之间的距离
//                            AudioManager.FLAG_PLAY_SOUND
//                                    | AudioManager.FLAG_SHOW_UI);
//                } else {
////                    Log.d(myTAG, "Inx:"+xx);
//                    if (xx <= 0.6) {
//                        Intxx = max((int) (xx * 15), 0);
//                        change_volume_progress((int) (xx * 100));
//                    } else {
//                        Intxx = max((int) (xx * 20), 0);
//                        change_volume_progress((int) (xx * 100 + 10));
//                    }
////                    Intxx=0;
////                    if(xx<=0.8&&xx>=0.2) {
////                        Intxx = max((int) (xx / 0.6 * 20), 0);
////                        change_volume_progress((int) (xx * 100 + 10));
////                    }
////                    Log.d(myTAG, "当前手指坐标"+xx);
//                    mAudioManager.setStreamVolume(AudioManager.STREAM_MUSIC, //音量类型
//                            Intxx,//设置音量为手指之间的距离
//                            AudioManager.FLAG_PLAY_SOUND
//                                    | AudioManager.FLAG_SHOW_UI);
//                }
//            }
        } else if (title.equals("THREE")) {
//            Log.d(myTAG, "当前收集序列的长度: " + point_history.size());
            smallVolChange=false;
        }
        /*
        使用动态手势实现乐曲切换（上下一曲）
         */
        else if (title.equals("TWO")) {
            smallVolChange=false;
            LandmarkProto.NormalizedLandmarkList landmarks = multiHandLandmarks.get(multiHandLandmarks.size() - 1);
            //获得中指指尖的坐标
            Float x8 = landmarks.getLandmark(12).getX();
            Float y8 = landmarks.getLandmark(12).getY();
            //获得手腕基点坐标
            Float base_x = landmarks.getLandmark(0).getX();
            Float base_y = landmarks.getLandmark(0).getY();

            /**
             * 将mediapipe获得的坐标进行还原（按照图像大小）
             */
            x8 = (float) min((int) (x8 * image_width), image_width - 1);
            y8 = (float) min((int) (y8 * image_width), image_width - 1);

            //tflite 满足手势要求，记录点位情况
            float[] temp_arr1 = {x8, y8};
            point_history.add(temp_arr1);//记录指尖

            //指尖动态手势识别的分类
            if (point_history.size() >= history_length * myInternal) {
                Log.d(myTAG, "当前序列point_history的长度为：" + point_history.size());
                pre_processed_point_history_list = pre_process_point_history_landmark(point_history);//对于历史点位进行预处理 达到的结果是一个一维向量
                point_history_len = pre_processed_point_history_list.length; //
                Log.d(myTAG, "当前预处理坐标点序列中点位的数量：" + point_history_len);// todo 确定size函数返回的是当前list中元素的个数
//                Log.d(myTAG, "预处理之后的结果: " + pre_processed_point_history_list.toString());
                //输入模型的标志 tflist 当记录的点位的数量满足模型训练时候输入的要求
                if (point_history_len == (history_length) * 2) {//通过模型分类器(函数)确定动态手势
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list);
                    Log.d(myTAG, "当前的识别结果: " + finger_gesture_id);
                }
                finger_gesture_histry.add(finger_gesture_id);
                if (finger_gesture_histry.size() >= 1) {
                    Log.d(myTAG, "当前待选择的答案" + finger_gesture_histry.toString());
                    most_common_fg_id = getMost(finger_gesture_histry);//获得历史识别结果中出现次数最多的
                    finger_gesture_histry.clear();// todo 8.6 由于出现误识别的情况，误识别结果的次数可能会累计，这这样不能通过计算结果序列中出现次数作为判断标准
                } else {
                    most_common_fg_id = 3;//总识别次数没有达到累计的25次的时候，不进行手势的响应
                }
                //识别完成一次 对记录的数据进行一次清空
                pre_processed_point_history_list = new float[history_length * 2];//记录最终结果的
                point_history.clear();

                //对于分类结果的处理
                if (most_common_fg_id == 0) {
                    //左滑
                    Log.d(myTAG, "当前识别的手势是:左滑或逆时针");
                    //上一曲
                    doAction(1);
                } else if (most_common_fg_id == 1) {
                    //右滑
                    Log.d(myTAG, "当前识别的手势是:右滑或顺时针");
                    //下一曲
                    doAction(2);
                } else if (most_common_fg_id == 2) {
                    //静止
                    Log.d(myTAG, "当前识别的手势是:静止");
                    try {
//                        TimeUnit.MILLISECONDS.sleep(10000);//暂停10s
                    } catch (Exception e) {
                        Log.d(myTAG, "HandleTheHands: " + e.toString());
                    }//                                Toast.makeText(this,"当前手势静止",Toast.LENGTH_SHORT).show();
                }
                point_history.clear();
                finger_gesture_histry.clear();
//                try {
//                    TimeUnit.MILLISECONDS.sleep(1000);//暂停1s
//                } catch (Exception e) {
//                    Log.d(myTAG, "HandleTheHands: " + e.toString());
//                }
            }
        } else {
            smallVolChange=false;
            float[] temp_arr = {0, 0};
            point_history.add(temp_arr);//当不属于当前手势的时候，记录点的坐标是（0，0）
            if (string_to_num_map.containsKey(title)) {
                int want_the_num = (Integer) string_to_num_map.get(title);
                if (flag_activity == -1 || flag_activity == want_the_num) {
                    array_to_count_activity[want_the_num]++;
                    change_the_bar(array_to_count_activity[want_the_num]);
                } else {
                    flag_activity = want_the_num;
                    array_to_count_activity[want_the_num] = 0;
                    change_the_bar(0);
                }
                for (int i = 0; i < array_to_count_activity.length; i++) {
                    if (i == want_the_num)
                        continue;
                    array_to_count_activity[i] = 0;
                }
                if (array_to_count_activity[want_the_num] == upper_limit) {

                    doAction(want_the_num);
                }

            } else {
                for (int i = 0; i < array_to_count_activity.length; i++) {
                    array_to_count_activity[i] = 0;
                }
                change_the_bar(0);
                point_history.clear();//没有任何手势的时候 进行清空
            }
        }
    }

    //tflite

    /**
     * 模型的预测函数
     *
     * @param pre_processed_point_history_list
     * @return anst
     */
    private int point_history_classifier(float[] pre_processed_point_history_list) {
        int[] ddims = {1, history_length * 2};
        //设置输入shape的大小
        TensorBuffer inputTensorbuffer = TensorBuffer.createFixedSize(ddims, DataType.FLOAT32);
        //将输入转换为TensorBuffer形式
        inputTensorbuffer.loadArray(pre_processed_point_history_list);
//        try {
//            //创建分类器
//            classifier = new Classifier(this, Classifier.Device.CPU, numThreads);
//        } catch (Exception e) {
//            Log.v(myTAG, "classifier is error: " + e.getMessage());
//        }
        String anst = "default";
//        if (classifier != null) {
//            final List<Classifier.Recognition> results = classifier.recognizeGesture(inputTensorbuffer);
//            anst = results.get(0).getTitle();//获得置信度为top1的分类结果
//            Log.d(myTAG, "分类器中输出的分类结果为: " + anst);
//        }
        if (anst.equals("yeahLeft")) { //更换模型时候需要修改
            return 0;
        } else if (anst.equals("yeahRight")) { //更换模型时候需要修改
            return 1;
        } else {
            return 2;
        }
    }

    /**
     * 对记录点位的序列进行归一化处理 并且转换为一维向量输出
     * 关键点位的预处理
     * @param point_history
     * @return
     */
    // TODO: 8/4/2022  
    private float[] pre_process_point_history_landmark(List<float[]> point_history) {
        float[] output = new float[history_length * 2];//记录最终归一化后的处理结果 是一维向量的形式
        int index_temp_land = 0;//计数 判断当前达到模型要求的点位数量与否
        int alllen = point_history.size();//当前记录点位的数量
        Float mx = Float.valueOf(-1000);//记录一维序列output中的最大值
        float base_x = 0, base_y = 0;
        // myInternal表示动态手势采样间隔
        for (int i = 0; i < history_length * myInternal; i += myInternal) {
            float[] temp_arr = point_history.get(i);
            if (i == 0) {
                //记录基准点的坐标 也就是说 指尖的动态手势对应的相对坐标是相对于第一个坐标点
                base_x = temp_arr[0];
                base_y = temp_arr[1];
            }
                //对得到的坐标转换为相对值 并且进行归一化
                //点位的记录 转换为一维向量
                output[index_temp_land++] = (temp_arr[0] - base_x) / image_width;
                output[index_temp_land++] = (temp_arr[1] - base_y) / image_height;
        }
        Log.d(myTAG, "转换为一维向量之后向量的长度: " + index_temp_land);
//        for(int i=0;i<history_length*2;i+=2){
//            if(i==0){
//                //记录基准点的坐标 也就是说 指尖的动态手势对应的相对坐标是相对于第一个坐标点
//                base_x=output[0];
//                base_y=output[1];
//            }
//            output[i]=(output[i]-base_x)/image_width;
//            output[i+1]=(output[i+1]-base_y)/image_height;
////            mx=Math.max(mx,Math.max(Math.abs(output[i]),Math.abs(output[i+1])));//选择整个列表的最大值
//        }
//        for(int i=0;i<history_length*2;i++)
//        {
//            output[i]/=mx;//归一化
//        }
        return output;
    }

    // tflite

    /**
     * 寻找列表中出现次数最多的元素
     * todo 可以优化整个查找过程的时间复杂度
     *
     * @param myList
     * @return 出现次数最多的元素对应的数字
     */
    public int getMost(List<Integer> myList) {//存疑 测试当前传参是否存在问题
        //比较出现次数
        int count = 0;
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();//前面记录元素的名字 后面记录元素的出现次数
        for (int j = 0; j < myList.size(); j++) {
            count = 0;
            for (int k = 0; k < myList.size(); k++) {
                if (myList.get(j) == myList.get(k)) {
                    count++;
                }
            }
            //存入数据
            map.put(myList.get(j), count);
        }

        Set<Integer> set = map.keySet();
        Iterator<Integer> it = set.iterator();
        int max = 0;
        int value = 0;
        while (it.hasNext()) {
            int key = it.next();
            if (map.get(key) > max) {//找到出现次数最多的
                max = map.get(key);
                value = key;
            }
        }
        return value;
    }

    private List<MainMenu> fruitList = new ArrayList<>();
    Handler toastHandler;
    //弹出图像的处理
    Handler photoHandler;
    // 文字弹窗的处理
    Handler popTextHandler;

    //手势识别的进度条
    ProgressBar progressBar;

    private void send_photo_to_bundle(String message1) //使用字符串控制活动
    {
        Bundle bundle = new Bundle();
        bundle.putString("photo", message1);
        Message message = new Message();
        message.setData(bundle);
        photoHandler.sendMessage(message);
    }

    /**
     * 将需要弹窗的文字传递到回调中
     * @param message1
     */
    private void send_text_to_bundle(String message1) //使用字符串控制活动
    {
        Bundle bundle = new Bundle();
        bundle.putString("text", message1);
        Message message = new Message();
        message.setData(bundle);
        popTextHandler.sendMessage(message);
    }

    //照片的修改
    ImageView handimg;

    private void changeSongsphoto() {
        handimg = findViewById(R.id.music_photo);
        int photo_id;
        if (songphotochangeflag == 0) {
            photo_id = R.drawable.song1;
            songphotochangeflag = 1;
        } else {
            photo_id = R.drawable.song2;
            songphotochangeflag = 0;
        }
        //设置ImageView的图片
        handimg.setImageDrawable(getResources().getDrawable((photo_id)));
    }

    /**
     * 执行文字弹窗的回调的实现函数
     * @param action
     */
    private void textPopToast(String action){
        Toast.makeText(this,action,Toast.LENGTH_SHORT).show();
//        Snackbar.make(getCurrentFocus(),"进入到音量微调节阶段",Snackbar.LENGTH_LONG).show();
    }

    /*
    关于歌曲相关信息显示的控件
        一个imgae一个textview
     */
    private ImageView myMusicImage;
    private TextView myMusicInfo;

    /*
    调用窗口亮度调节的类，实现光线传感器对应的亮度自动调节
     */
    private BrightnessAdapter myAPP_BrightnessAdapter;
    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_music_control);

        //对于主界面中的列表进行初始化
        initFruits();//对列表中存在的MainMenu进行初始化
        MenuAdapter adapter = new MenuAdapter(this, R.layout.main_menu_ui_item, fruitList);//其中fruitList列表中的所有元素都是MainMenu
        ListView listView = (ListView) findViewById(R.id.list_view_music);
        listView.setAdapter(adapter);
        //对于listview中点击事件添加
        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            //onItemClick通过 position参数判断出用户点击的是哪一个子项
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                switch (position) {
                    case 0://注意 listview中每一个列表单元从0开始数数
//                        Toast.makeText(MusicControl.this,"第"+position+"个item", Toast.LENGTH_SHORT).show();//目前通过弹窗测试当前用户点按的是哪一个item
                    {
                        doAction(1);//上一首歌
                        break;
                    }
                    case 1:
                        doAction(2);//下一首歌
                        break;
                    case 2:
                        doAction(3);//暂停
                        break;
                    case 3:
                        doAction(4);//播放
                        break;
                    case 4: {
                        //音量调高
                        pressVolume(1);
                        break;
                    }
                    case 5: {
                        //音量调低
                        pressVolume(0);
                        break;
                    }
                    case 6:
                        doAction(5);//退出
                        break;
                }
            }
        });

        /*
        使用光线传感器实现亮度的调节
        8.20 成功
         */
        myAPP_BrightnessAdapter =new BrightnessAdapter(getApplicationContext(),getWindow());
        myAPP_BrightnessAdapter.regist();
        next_song = 0;
        previous_song = 0;

        /*
        音乐信息控件初始化
         */
        myMusicImage = (ImageView) findViewById(R.id.music_photo);
        myMusicInfo = (TextView) findViewById(R.id.music_textView);
        //音乐信息通知栏的接收器
        NotifyHelper.getInstance().setNotifyListener(this);
        /*
        通知栏监听相关权限的获得
         */
        if (!isNLServiceEnabled()) {
            //动态权限授予
            Intent intent = new Intent("android.settings.ACTION_NOTIFICATION_LISTENER_SETTINGS");
            startActivityForResult(intent, REQUEST_CODE);//通过将激活码传递 实现按钮功能的响应
        } else {
//            Log.d(myTAG, "通知服务已开启");
            Toast.makeText(this, "通知服务已发布", Toast.LENGTH_SHORT).show();
            //在已经授权的情况下，需要对监听器服务进行激活
            toggleNotificationListenerService();
        }

        //
        previewDisplayView = new SurfaceView(this);
        setupPreviewDisplayView();

        // 获取权限
        PermissionHelper.checkAndRequestCameraPermissions(this);

        // 初始化assets管理器，以便MediaPipe应用资源
        AndroidAssetUtil.initializeNativeAssetManager(this);

        eglManager = new EglManager(null);

        // 通过加载获取一个帧处理器
        processor = new FrameProcessor(
                this,
                eglManager.getNativeContext(),
                BINARY_GRAPH_NAME,
                INPUT_VIDEO_STREAM_NAME,
                OUTPUT_VIDEO_STREAM_NAME);
        processor.getVideoSurfaceOutput().setFlipY(FLIP_FRAMES_VERTICALLY);

        //音量进度条初始化以及实现拖动音量调节
        Context context = getBaseContext();
        SeekBar myseekbar = (SeekBar) findViewById(R.id.volume_progress);
        myseekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {//设置Seekbar的事件
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {//当Seekbar的进度发生变化时
//                Log.d(myTag, "onStartTrackingTouch: 当前Seekbar条进度被修改");
//                int Intxx=myseekbar.getScrollX();
//                Log.d(myTag, "onCreate: "+Intxx);
//                AudioManager mAudioManager = (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
//                mAudioManager.setStreamVolume(AudioManager.STREAM_MUSIC, //音量类型
//                        Intxx,//设置音量为手指之间的距离
//                        AudioManager.FLAG_PLAY_SOUND
//                                | AudioManager.FLAG_SHOW_UI);
                if (b) {//当修改来自于用户而不是API的时候
//                    Log.d(myTag, "Seekbar当前点进度情况: "+Intxx);
                    if ("Xiaomi".equals(android.os.Build.MANUFACTURER)) {
                        int Intxx = myseekbar.getProgress();
                        AudioManager mAudioManager = (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
                        mAudioManager.setStreamVolume(AudioManager.STREAM_MUSIC, //音量类型
                                (int) (Intxx / 0.75),//设置音量为手指之间的距离
                                AudioManager.FLAG_PLAY_SOUND
                                        | AudioManager.FLAG_SHOW_UI);
                    } else {
                        int Intxx = myseekbar.getProgress();
                        AudioManager mAudioManager = (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
                        mAudioManager.setStreamVolume(AudioManager.STREAM_MUSIC, //音量类型
                                (int) (Intxx * 17 * 1.0 / 100),//设置音量为手指之间的距离
                                AudioManager.FLAG_PLAY_SOUND
                                        | AudioManager.FLAG_SHOW_UI);
                    }
                }
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
//                Log.d(myTag, "onStartTrackingTouch: 当前Seekbar条被触碰");
//                int Intxx=myseekbar.getProgress();
//                Log.d(myTag, "onCreate: "+Intxx);
//                AudioManager mAudioManager = (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
//                mAudioManager.setStreamVolume(AudioManager.STREAM_MUSIC, //音量类型
//                        Intxx,//设置音量为手指之间的距离
//                        AudioManager.FLAG_PLAY_SOUND
//                                | AudioManager.FLAG_SHOW_UI);
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
//                Log.d(myTag, "onStartTrackingTouch: 当前Seekbar条被放开");
            }
        });

        AndroidPacketCreator packetCreator = processor.getPacketCreator();
        Map<String, Packet> inputSidePackets = new HashMap<>();
        inputSidePackets.put(INPUT_NUM_HANDS_SIDE_PACKET_NAME, packetCreator.createInt32(NUM_HANDS));
        processor.setInputSidePackets(inputSidePackets);

        //音乐控制手势对应的弹窗
        toastHandler = new Handler(new Handler.Callback() {
            @Override
            public boolean handleMessage(Message msg) {
                Bundle bundle = msg.getData();
                //获取当前动作
                String action = (String) bundle.get("action");
                //执行动作
                if ("next_song".equals(action)) {
                    pop_toast(R.drawable.next_song);
                } else if ("previous_song".equals(action)) {
                    pop_toast(R.drawable.previous_song);
                } else if ("puase_song".equals(action)) {
                    pop_toast(R.drawable.puase_song);
                } else {
                    pop_toast(R.drawable.play_song);
                }
                return false;
            }
        });
        //音量调节对应的相关弹窗
        photoHandler = new Handler(new Handler.Callback() {
            @Override//对于进行手势的识别
            public boolean handleMessage(Message msg) {
                Bundle bundle = msg.getData();
                //获取当前动作
                String action = (String) bundle.get("photo");
                //执行动作 照片的弹窗
                changeSongsphoto();
                return false;
            }
        });
        // 弹窗对应的回调
        //音量调节对应的相关弹窗
        popTextHandler = new Handler(new Handler.Callback() {
            @Override//对于进行手势的识别
            public boolean handleMessage(Message msg) {
                Bundle bundle = msg.getData();
                //获取当前动作
                String action = (String) bundle.get("text");
                //执行动作 照片的弹窗
                textPopToast(action);
                return false;
            }
        });

        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        prefs.registerOnSharedPreferenceChangeListener(this); // 注册

        Log.d("indexof", "test");

        //调节进度条
        progressBar = (ProgressBar) findViewById(R.id.progress_bar_music);
        change_the_bar(0);
        //新增一个识别手指，并进行操作的模块---手势识别模块的调用
        processor.addPacketCallback(
                OUTPUT_LANDMARKS_STREAM_NAME,
                (packet) -> {
                    List<LandmarkProto.NormalizedLandmarkList> multiHandLandmarks =
                            PacketGetter.getProtoVector(packet, LandmarkProto.NormalizedLandmarkList.parser());
                    if(isHandInRecZone(multiHandLandmarks)){
                        // : 9/8/2022 当手掌在识别区域内，才对手势进行响应
                        HandleTheHands(multiHandLandmarks);
                    }
//                    find_music_info();//目前只实现了对于本地音乐信息的获取
//                    HandleTheVolume(multiHandLandmarks);
//                    if (juTheback(packet, multiHandLandmarks)) {
//                        finish();
//                    }
                }
        );

    }

    /**
     * 对于手掌是否在识别范围内的判断
     *  当前识别范围设置的是listview所在的constraintview 是因为getTop等函数只能获得相对于控件母view的位置情况
     * @param multiHandLandmarks
     * @return 判断结果
     */
    private boolean isHandInRecZone(List<LandmarkProto.NormalizedLandmarkList> multiHandLandmarks){
        // : 9/8/2022 实现手掌识别范围的限制
        LandmarkProto.NormalizedLandmarkList landmarks = multiHandLandmarks.get(multiHandLandmarks.size() - 1);
        Float y2 = landmarks.getLandmark(9).getY();// 获得中指根部的纵坐标情况
        Float yy=landmarks.getLandmark(0).getY();
        // 当前视频在屏幕上的大小 按照相对于父容器的情况进行计算
        TextView vedio_load_size=findViewById(R.id.no_camera_access_view2);
        int vedio_bottom=vedio_load_size.getBottom();
        int vedio_top=vedio_load_size.getTop();
        int vedio_height=vedio_bottom-vedio_top;
        //获得识别区域控件在屏幕上的坐标
        ConstraintLayout rec_zone=findViewById(R.id.listAndRecArea);
        int rec_zone_bottom=rec_zone.getBottom();
        int rec_zone_top=rec_zone.getTop();
//        Log.d(myTAG, "rec_Zone_top="+rec_zone_top+" rec_zone_bottom"+rec_zone_bottom);
        if(!(yy*vedio_height>rec_zone_top&&yy*vedio_height<rec_zone_bottom&&y2*vedio_height>rec_zone_top&&y2*vedio_height<rec_zone_bottom)){
//            Log.d(myTAG, "当前的手掌不在识别区域内");
//            send_text_to_bundle("当前的手掌不在识别区域内");
            return false;
        }
        else{
            return true;
        }
    }
    //获得当前播放音乐情况的信息
    //:线程调用不成功 当前报错android.database.CursorIndexOutOfBoundsException: Index 0 requested, with a size of 0在try块对应的查询语句中 那么大概率是当前使用的方法不能够拿到正在播放的媒体的信息
    // 目前本地音乐信息的获取成功
    private void find_music_info() {
        ContentResolver cr = getApplication().getContentResolver();
        if (cr == null) {
//            Log.d(myTag, "find_music_info: ContentResolver存在问题");
            return;
        }
        Cursor c = null;//扫描文件库
        c = cr.query(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI,
                null, null, null, MediaStore.Audio.Media.DEFAULT_SORT_ORDER);
        if (c == null) {
//            Log.d(myTag, "find_music_info: Cursor为空值");
            return;
        }
        try {
            c.moveToFirst();
//            if (c.moveToFirst()) {
            do {
//                    Log.d(myTag, "find_music_info: 开始歌曲信息查找");
                //歌曲名
                @SuppressLint("Range") String title = c.getString(c.getColumnIndex(MediaStore.Audio.Media.TITLE));

                //歌手
                @SuppressLint("Range") String singer = c.getString(c.getColumnIndex(MediaStore.Audio.Media.ARTIST));

                //专辑
                @SuppressLint("Range") String album = c.getString(c.getColumnIndex(MediaStore.Audio.Media.ALBUM));
//                    Log.d(myTag, "find_music_info:当前播放音乐信息：\n歌曲名:"+title+"\n歌手:"+singer+"\n专辑:"+album);
//
//                //长度
//                long size = c.getLong(c
//                        .getColumnIndex(MediaStore.Audio.Media.SIZE));
//
//                //时长
//                int duration = c.getInt(c
//                        .getColumnIndex(MediaStore.Audio.Media.DURATION));
//
//                //路径
//                String url = c.getString(c
//                        .getColumnIndex(MediaStore.Audio.Media.DATA));
//
//                //显示的文件名
//                String _display_name = c.getString(c
//                        .getColumnIndex(MediaStore.Audio.Media.DISPLAY_NAME));
//
//                //类型
//                String mime_type = c.getString(c
//                        .getColumnIndex(MediaStore.Audio.Media.MIME_TYPE));
            } while (c.moveToNext());
//            }
        } catch (Exception e) {
//            Log.d(myTAG, "find_music_info: "+e.toString());
        }
    }

    //修改进度条
    private void change_the_bar(int change_num) {
        progressBar.setProgress(max(0, min(change_num * 4, 100)));
    }

    //修改音量调节条 音量进度条调节
    private void change_volume_progress(int intxx) {
        SeekBar myseekbar = (SeekBar) findViewById(R.id.volume_progress);
        myseekbar.setProgress(max(0, min(intxx, 100)));
    }

    //弹窗提示
    private void pop_toast(int photo_id) {
        Toast customToast = new Toast(getApplicationContext());
        //获得view的布局
        View customView = LayoutInflater.from(getApplicationContext()).inflate(R.layout.custom_toast, null);
        ImageView img = (ImageView) customView.findViewById(R.id.toast_image);
        //设置ImageView的图片
        img.setBackgroundResource(photo_id);
        //设置toast的View,Duration,Gravity最后显示
        customToast.setView(customView);
        customToast.setDuration(Toast.LENGTH_SHORT);
        // 弹窗位置的设置 y方向上向下移动300单位 实现弹窗照片避开手势识别进度条
        customToast.setGravity(Gravity.CENTER, 0, 300);
        customToast.show();
    }

    //初始化菜单
    private void initFruits() {
        fruitList.clear();

        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);


//设置上一首的图标
        String previous_song_option = prefs.getString("play_previous_song", "0");
        if ("0".equals(previous_song_option)) {
            MainMenu next_song_item = new MainMenu("上一首", R.drawable.handtoleft);
            fruitList.add(next_song_item);

        } else if ("1".equals(previous_song_option)) {
            MainMenu next_song_item = new MainMenu("上一首", R.drawable.one);
            fruitList.add(next_song_item);
        }
//设置下一首的图标
        String next_song_option = prefs.getString("play_next_song", "0");
        if ("0".equals(next_song_option)) {
            MainMenu next_song_item = new MainMenu("下一首", R.drawable.handtoright);
            fruitList.add(next_song_item);

        } else if ("1".equals(next_song_option)) {
            MainMenu next_song_item = new MainMenu("下一首", R.drawable.two);
            fruitList.add(next_song_item);
        }

        //设置暂停的图标
        String stop_song_option = prefs.getString("stop_the_song", "0");
        if ("0".equals(stop_song_option)) {
            MainMenu next_song_item = new MainMenu("暂停", R.drawable.handstopmusic);
            fruitList.add(next_song_item);

        } else if ("1".equals(stop_song_option)) {
            MainMenu next_song_item = new MainMenu("暂停", R.drawable.fist);
            fruitList.add(next_song_item);
        }

//设置播放的图标
        String play_song_option = prefs.getString("play_the_song", "0");
        if ("0".equals(play_song_option)) {
            MainMenu next_song_item = new MainMenu("播放", R.drawable.ok);
            fruitList.add(next_song_item);

        } else if ("1".equals(play_song_option)) {
            MainMenu next_song_item = new MainMenu("播放", R.drawable.three);
            fruitList.add(next_song_item);
        }

        String volume_up_option = prefs.getString("volume_up", "0");
        if ("0".equals(volume_up_option)) {
            MainMenu next_song_item = new MainMenu("音量调大", R.drawable.handmoveright);
            fruitList.add(next_song_item);

        } else if ("1".equals(volume_up_option)) {
            MainMenu next_song_item = new MainMenu("音量调大", R.drawable.handmoveright);
            fruitList.add(next_song_item);
        }
        String volume_down_option = prefs.getString("volume_down", "0");
        if ("0".equals(volume_down_option)) {
            MainMenu next_song_item = new MainMenu("音量调小", R.drawable.handmoveleft);
            fruitList.add(next_song_item);

        } else if ("1".equals(volume_down_option)) {
            MainMenu next_song_item = new MainMenu("音量调小", R.drawable.handmoveleft);
            fruitList.add(next_song_item);
        }


        MainMenu finish_app = new MainMenu("退出", R.drawable.finish_sign);
        fruitList.add(finish_app);
    }


    protected void play(int which_mp3) {
        MediaPlayer mpMediaPlayer = null;
        mpMediaPlayer = MediaPlayer.create(this, which_mp3);
        try {
            mpMediaPlayer.prepare();
        } catch (IllegalStateException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        mpMediaPlayer.start();
    }

    private void doAction(Integer title) {
        Runtime runtime = Runtime.getRuntime();
        if (title == 2) {
            play(R.raw.next_song);
            //下一首歌
            send_message_to_bundle("next_song");
            //通过歌曲切换 更换音乐封面（使用图片测试时期）
//            send_photo_to_bundle("");
            try {
                runtime.exec("input keyevent " + KeyEvent.KEYCODE_MEDIA_NEXT);
            } catch (Exception e) { // TODO Auto-generated catch block
                e.printStackTrace();
            }
        } else if (title == 1) {
            play(R.raw.previous_song);
            //上一首歌
            send_message_to_bundle("previous_song");
            //通过歌曲切换 更换音乐封面（使用图片测试时期）
//            send_photo_to_bundle("");
            try {
                runtime.exec("input keyevent " + KeyEvent.KEYCODE_MEDIA_PREVIOUS);
            } catch (Exception e) { // TODO Auto-generated catch block
                e.printStackTrace();
            }
        } else if (title == 3) {
            play(R.raw.pause_music);
            //暂停
            send_message_to_bundle("puase_song");
            try {
                runtime.exec("input keyevent " + KeyEvent.KEYCODE_MEDIA_PAUSE);
            } catch (Exception e) { // TODO Auto-generated catch block
                e.printStackTrace();
            }
        } else if (title == 4) {
            play(R.raw.play_music);
            play(R.raw.play_music);
            //播放
            send_message_to_bundle("playplayplay");
            try {
                runtime.exec("input keyevent " + KeyEvent.KEYCODE_MEDIA_PLAY);
            } catch (Exception e) { // TODO Auto-generated catch block
                e.printStackTrace();
            }
        } else if (title == 5) {
//            play(R.raw.pause_music);
//            // 退出
//            send_message_to_bundle("puase_song");
//            try {
//                runtime.exec("input keyevent " + KeyEvent.KEYCODE_MEDIA_PAUSE);
//            } catch (Exception e) { // TODO Auto-generated catch block
//                e.printStackTrace();
//            }
            finish();
            // TODO: 8/31/2022
        }
        try {
            Thread.sleep(500);
        } catch (Exception e) {
            e.printStackTrace();
        }
//        Long upp = Long.valueOf("1500000000");
//
//        for (long i = 0; i <= upp; i++) {
//            long aa = i;
//
//        }
    }

    //根据手的横坐标实时改变音量
    private void HandleTheVolume(List<LandmarkProto.NormalizedLandmarkList> multiHandLandmarks) {
        LandmarkProto.NormalizedLandmarkList landmarks = multiHandLandmarks.get(multiHandLandmarks.size() - 1);
        Float xx = landmarks.getLandmark(5).getX();
        Float y8 = landmarks.getLandmark(8).getY();
        Float y5 = landmarks.getLandmark(5).getY();

        Float y12 = landmarks.getLandmark(12).getY();
        Float y9 = landmarks.getLandmark(9).getY();
        if (y8 < y5 && y12 > y9) {
            int Intxx = Math.max((int) (xx * 20) - 3, 0);
            Context context = getBaseContext();
            AudioManager mAudioManager = (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
            if ("Xiaomi".equals(android.os.Build.MANUFACTURER)) {
                mAudioManager.setStreamVolume(AudioManager.STREAM_MUSIC, //音量类型
                        Intxx * 10,//设置音量为手指之间的距离
                        AudioManager.FLAG_PLAY_SOUND
                                | AudioManager.FLAG_SHOW_UI);
            } else {
                mAudioManager.setStreamVolume(AudioManager.STREAM_MUSIC, //音量类型
                        Intxx,//设置音量为手指之间的距离
                        AudioManager.FLAG_PLAY_SOUND
                                | AudioManager.FLAG_SHOW_UI);
            }
        }
    }

    private void pressVolume(int flag) {
        Context context = getBaseContext();
        AudioManager mAudioManager = (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
        int currvolumem = mAudioManager.getStreamVolume(AudioManager.STREAM_MUSIC);//获得的是当前媒体播放的音量
//        Log.d(myTAG, "当前系统的音量: "+currvolumem);
//        mAudioManager.getDevices();//获得当前设备的型号
        if ("Xiaomi".equals(android.os.Build.MANUFACTURER)) {
            // 实验测试 小米系统音量范围(0,150)
            if (flag == 1)//音量调高
                currvolumem += 5;
            else if (flag == 0)//音量降低
                currvolumem -= 5;
            mAudioManager.setStreamVolume(AudioManager.STREAM_MUSIC, //音量类型
                    (currvolumem),//设置音量为手指之间的距离
                    AudioManager.FLAG_PLAY_SOUND
                            | AudioManager.FLAG_SHOW_UI);
        } else {
            /*
            注意 oppo的音量调节范围在[0,16]之间
             */
            if (flag == 1)//音量调高
            {
                currvolumem += 1;
            } else if (flag == 0)//音量降低
            {
                currvolumem -= 1;
            }
//            Log.d(myTAG, "oppo当前音量:"+(int)currvolumem);
            change_volume_progress((int) (currvolumem * 100.0 / 17));
            mAudioManager.setStreamVolume(AudioManager.STREAM_MUSIC, //音量类型
                    ((int) currvolumem),//设置音量为手指之间的距离
                    AudioManager.FLAG_PLAY_SOUND
                            | AudioManager.FLAG_SHOW_UI);
        }
    }

    public void onSharedPreferenceChanged(SharedPreferences prefs, String key) {
        // 事件处理器. 根据数据的变化,对显示和行为作改变
//        fruitList.remove(fruitList.indexOf())
//        MainMenu play_song_item = new MainMenu("播放", R.drawable.ok);
//        Log.d("indexof", String.valueOf(fruitList.indexOf(play_song_item)));


    }

    protected void onResume() {

        super.onResume();

        initFruits();
        MenuAdapter adapter = new MenuAdapter(this, R.layout.main_menu_ui_item, fruitList);
        ListView listView = (ListView) findViewById(R.id.list_view_music);
        listView.setAdapter(adapter);

        next_song = 0;
        converter = new ExternalTextureConverter(eglManager.getContext());
        converter.setFlipY(FLIP_FRAMES_VERTICALLY);
        converter.setConsumer(processor);
        if (PermissionHelper.cameraPermissionsGranted(this)) {
            startCamera();
        }
    }

    @Override
    protected void onPause() {

        super.onPause();
        converter.close();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {

        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    // 计算最佳的预览大小
    protected Size computeViewSize(int width, int height) {
        return new Size(width, height);
    }

    protected void onPreviewDisplaySurfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        // 设置预览大小
        Size viewSize = computeViewSize(width, height);
        Size displaySize = cameraHelper.computeDisplaySizeFromViewSize(viewSize);
        // 根据是否旋转调整预览图像大小
        boolean isCameraRotated = cameraHelper.isCameraRotated();
        converter.setSurfaceTextureAndAttachToGLContext(
                previewFrameTexture,
                isCameraRotated ? displaySize.getHeight() : displaySize.getWidth(),
                isCameraRotated ? displaySize.getWidth() : displaySize.getHeight());
    }


    private void setupPreviewDisplayView() {
        previewDisplayView.setVisibility(View.GONE);
        ViewGroup viewGroup = findViewById(R.id.preview_display_layout_music);
        viewGroup.addView(previewDisplayView);

        previewDisplayView
                .getHolder()
                .addCallback(
                        new SurfaceHolder.Callback() {
                            @Override
                            public void surfaceCreated(SurfaceHolder holder) {
                                processor.getVideoSurfaceOutput().setSurface(holder.getSurface());
                            }

                            @Override
                            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                                onPreviewDisplaySurfaceChanged(holder, format, width, height);
                            }

                            @Override
                            public void surfaceDestroyed(SurfaceHolder holder) {
                                processor.getVideoSurfaceOutput().setSurface(null);
                            }
                        });
    }

    // 相机启动后事件
    protected void onCameraStarted(SurfaceTexture surfaceTexture) {
        // 显示预览
        previewFrameTexture = surfaceTexture;
        previewDisplayView.setVisibility(View.VISIBLE);
    }

    // 设置相机大小
    protected Size cameraTargetResolution() {
        return null;
    }

    // 启动相机
    public void startCamera() {
        cameraHelper = new CameraXPreviewHelper();
        cameraHelper.setOnCameraStartedListener(this::onCameraStarted);
        CameraHelper.CameraFacing cameraFacing =
                USE_FRONT_CAMERA ? CameraHelper.CameraFacing.FRONT : CameraHelper.CameraFacing.BACK;
        cameraHelper.startCamera(this, cameraFacing, null, cameraTargetResolution());
    }

    // 解析关键点
    private static String getLandmarksDebugString(LandmarkProto.NormalizedLandmarkList landmarks) {
        int landmarkIndex = 0;
        StringBuilder landmarksString = new StringBuilder();
        for (LandmarkProto.NormalizedLandmark landmark : landmarks.getLandmarkList()) {
            landmarksString.append("\t\tLandmark[").append(landmarkIndex).append("]: (").append(landmark.getX()).append(", ").append(landmark.getY()).append(", ").append(landmark.getZ()).append(")\n");
            ++landmarkIndex;
        }
        return landmarksString.toString();
    }

    /**
     * 是否启用通知监听服务
     *
     * @return
     */
    public boolean isNLServiceEnabled() {
        Set<String> packageNames = NotificationManagerCompat.getEnabledListenerPackages(this);
        if (packageNames.contains(getPackageName())) {
            return true;
        }
        return false;
    }

    /**
     * 切换通知监听器服务 需要将监听器服务进行激活
     */
    public void toggleNotificationListenerService() {
        PackageManager pm = getPackageManager();
        pm.setComponentEnabledSetting(new ComponentName(getApplicationContext(), NotifyService.class),
                PackageManager.COMPONENT_ENABLED_STATE_DISABLED, PackageManager.DONT_KILL_APP);

        pm.setComponentEnabledSetting(new ComponentName(getApplicationContext(), NotifyService.class),
                PackageManager.COMPONENT_ENABLED_STATE_ENABLED, PackageManager.DONT_KILL_APP);
    }

    private static final int REQUEST_CODE = 9527;

    @Override
    public void onReceiveMessage(int type) {
        switch (type) {
            case N_MESSAGE:
                myMusicInfo.setText("收到短信消息");
                break;
            case N_CALL:
                myMusicInfo.setText("收到来电消息");
                break;
            case N_WX:
                myMusicInfo.setText("收到微信消息");
                break;
            case N_QQ:
                myMusicInfo.setText("收到QQ消息");
                break;
            case N_YUN:
                myMusicInfo.setText("收到网易云消息");
                break;
            default:
                break;
        }
    }

    @Override
    public void onRemovedMessage(int type) {
        switch (type) {
            case N_MESSAGE:
                myMusicInfo.setText("移除短信消息");
                break;
            case N_CALL:
                myMusicInfo.setText("移除来电消息");
                break;
            case N_WX:
                myMusicInfo.setText("移除微信消息");
                break;
            case N_QQ:
                myMusicInfo.setText("移除QQ消息");
                break;
            case N_YUN:
                myMusicInfo.setText("移除网易云消息");
                break;
            case N_QQMUSIC:
                myMusicInfo.setText("移除QQ音乐消息");
                break;
            default:
                break;
        }
    }

    /**
     * 收到通知 获得通知包中的信息
     *
     * @param sbn 状态栏通知
     */
    @RequiresApi(api = Build.VERSION_CODES.M)//对于版本有一定要求
    @Override
    public void onReceiveMessage(StatusBarNotification sbn) {
//        Log.d(myTAG, "package Name:"+sbn.getPackageName());
//        Bundle extras=sbn.getNotification().extras;
//        String title=extras.getString(Notification.EXTRA_TITLE);
//        String content = extras.getString(Notification.EXTRA_TEXT); //通知内容
//        String context1=extras.getString(Notification.CATEGORY_EVENT);
//        Log.d(myTAG, context1);
//        Log.d(myTAG, "onReceiveMessage:\ntitle:"+title+"\ncontent:"+content);
        if (sbn.getPackageName().equals(YUN) || sbn.getPackageName().equals(qqmusic)) {//网易云文件消息的处理
            //利用extra获得Notification中更为详细的信息
            Bundle extras = sbn.getNotification().extras;
            String title = extras.getString(Notification.EXTRA_TITLE);
            String content = extras.getString(Notification.EXTRA_TEXT); //通知内容
            int smallIconId = extras.getInt(Notification.EXTRA_SMALL_ICON); //通知小图标id 获得结果是int类型
            Icon largeIcon = extras.getParcelable(Notification.EXTRA_LARGE_ICON); //通知的大图标，注意和获取小图标的区别 得到的是一个bitmap类型的
//            PendingIntent pendingIntent = sbn.getNotification().contentIntent; //获取通知的PendingIntent
//            Bitmap smallIcon=getSmallIcon(getApplicationContext(),sbn.getPackageName(),smallIconId);
//            myimg.setImageBitmap(smallIcon);
            myMusicImage.setImageIcon(largeIcon);
            //消息时间
            String time = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.CHINESE).format(new Date(sbn.getPostTime()));
            myMusicInfo.setText(String.format(Locale.getDefault(),
                    "%s\n%s\n%s\n",
                    title, content, time));
//            Log.d(TAG, "largeIcon"+largeIcon.toString());
        } else if (sbn.getPackageName().equals(QQ)) {
            Bundle extras = sbn.getNotification().extras;
            String title = extras.getString(Notification.EXTRA_TITLE);//消息来源
            String content = extras.getString(Notification.EXTRA_TEXT); //通知内容
            String toastText = "QQ来信\n消息来源:" + title + "\n消息内容:" + content;
            Toast.makeText(this, toastText, Toast.LENGTH_LONG).show();
        } else if (sbn.getPackageName().equals(WX)) {
            Bundle extras = sbn.getNotification().extras;
            String title = extras.getString(Notification.EXTRA_TITLE);//消息来源
            String content = extras.getString(Notification.EXTRA_TEXT); //通知内容
            String toastText = "微信来信\n消息来源:" + title + "\n消息内容:" + content;
            Toast.makeText(this, toastText, Toast.LENGTH_LONG).show();
        } else if (sbn.getPackageName().equals(DD)) {
            Bundle extras = sbn.getNotification().extras;
            String title = extras.getString(Notification.EXTRA_TITLE);//消息来源
            String content = extras.getString(Notification.EXTRA_TEXT); //通知内容
            String toastText = "钉钉来信\n消息来源:" + title + "\n消息内容:" + content;
            Toast.makeText(this, toastText, Toast.LENGTH_LONG).show();
        } else {//其余消息类型的处理
            Bundle extras = sbn.getNotification().extras;
            String title = extras.getString(Notification.EXTRA_TITLE);
            String content = extras.getString(Notification.EXTRA_TEXT); //通知内容
            Log.d(myTAG, "onReceiveMessage:\ntitle:" + title + "\ncontent:" + content);
        }
    }

    /**
     * 当通知栏移除通知时
     *
     * @param sbn
     */
    @Override
    public void onRemovedMessage(StatusBarNotification sbn) {
        myMusicInfo.setText("通知移除");
    }
}