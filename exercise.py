from gtts import gTTS
from pygame import mixer
import cv2
import mediapipe as mp
import numpy as np
import math
import tempfile
import time
import schedule

cap = cv2.VideoCapture(0)
myFace = mp.solutions.face_mesh
face = myFace.FaceMesh(refine_landmarks=True)

# ---需要用到的位置---
LEFT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LIPS = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,42,183,78]
RIGHT_IRIS = [474,475,476,477]
LEFT_IRIS = [469,470,471,472]
RightEye_R_P = [33]     #右邊眼睛最右邊位置
RightEye_L_P = [133]    #右邊眼睛最左邊位置
LeftEye_R_P = [362]     #左邊眼睛最右邊位置
LeftEye_L_P = [263]     #左邊眼睛最左邊位置

# ---變數---
c = 1
TIME_RATE = 60
blink_times = 0
CEF_COUNTER1 = 0
CEF_COUNTER2 = 0
frame_counter = 0
CLOSE_EYES_FRAME = 3
yawn_times = 0
CLOSE_YAWN_FRAME = 3
time_list = []
eye_position_time = 0

# ---function---
def gamma_trans(img, gamma):  # gamma correction
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def speak(sentence):
    with tempfile.NamedTemporaryFile(delete=True) as f:
        tts = gTTS(text=sentence,lang="zh")
        tts.save("{}.mp3".format(f.name))
        mixer.init()
        mixer.music.load("{}.mp3".format(f.name))
        mixer.music.play()

def blink_speak(blink_times):
    if blink_times > 18:
        speak("眨眼頻率過高，請打起精神")
        blink_times = 0
    return blink_times

def yawn_speak(yawn_times):
    if yawn_times > 2:
        speak("頻繁打哈欠，請打起精神")
        yawn_times = 0
    return yawn_times

def euclidean_distance(pt1,pt2):    #歐式距離
    x1, y1 = pt1.ravel()
    x2, y2 = pt2.ravel()
    distance = math.sqrt((x2 - x1) ** 2 + (y2-y1) ** 2)
    return distance

def eye_position(iris_center,right_point,left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist / total_distance
    iris_pos = ""
    if ratio <= 0.34:
        iris_pos = "right"
        eye_switch = True
    elif ratio >0.34 and ratio <= 0.8:
        iris_pos = "center"
        eye_switch = False
    else:
        iris_pos = "left"
        eye_switch = True
    return iris_pos, ratio, eye_switch

def blink(right_eye,left_eye):
    rh_right = right_eye[0]
    rh_left = right_eye[8]
    rv_top = right_eye[12]
    rv_bottom = right_eye[4]

    lh_right = left_eye[0]
    lh_left = left_eye[8]
    lv_top = left_eye[12]
    lv_bottom = left_eye[4]

    rh_distance = euclidean_distance(rh_right, rh_left)
    rv_distance = euclidean_distance(rv_top, rv_bottom)
    lh_distance = euclidean_distance(lh_right, lh_left)
    lv_distance = euclidean_distance(lv_top, lv_bottom)

    reRatio = rh_distance/rv_distance
    leRatio = lh_distance/lv_distance

    blink_Ratio = (reRatio+leRatio)/2
    return blink_Ratio

def yawn(lips):
    lip_left_pos = lips[0]
    lip_right_pos = lips[10]
    lip_bottom_pos = lips[5]
    lip_top_pos = lips[25]

    lip_h_distance = euclidean_distance(lip_left_pos, lip_right_pos)
    lip_v_distance = euclidean_distance(lip_top_pos, lip_bottom_pos)
    lip_ratio = lip_h_distance/lip_v_distance
    return lip_ratio

# ---main_code---
while True:
    # ---前置作業---
    ret, frame = cap.read()
    FPS = cap.get(5)
    frame = cv2.flip(frame, 1)
    Height, Width = frame.shape[:2]
    if ret == True:
        frameRate = int(FPS) * TIME_RATE
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    #bgr圖片轉rgb圖片
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #bgr圖片轉灰階圖片
        result = face.process(rgb_img)
        mean = np.mean(gray_img)
        gamma_val = math.log10(0.5) / math.log10(mean / 255)
        image_gamma_correct = gamma_trans(frame, gamma_val)
        try:
            mesh_points = np.array([np.multiply([p.x, p.y], [Width, Height]).astype(int) for p in result.multi_face_landmarks[0].landmark])

            # ---眼睛位置判斷---
            (left_eye_x, left_eye_y), left_eye_r = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])   #minEnclosingCircle函式(可以取得該所有點所圍成最小的圓並回船圓心跟半徑)
            (right_eye_x, right_eye_y), right_eye_r = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            left_eye_center = np.array([left_eye_x, left_eye_y], dtype=np.int32)
            right_eye_center = np.array([right_eye_x, right_eye_y], dtype=np.int32)
            eye_pos, ratio, eye_switch = eye_position(right_eye_center, mesh_points[LeftEye_L_P], mesh_points[LeftEye_R_P][0])
            if eye_switch == True:
                start = time.time()
                time_list.append(start)
                eye_position_time = time_list[-1] - time_list[0]
                if eye_position_time > 1:
                    speak("請專心看前方")
                    time.sleep(2)
            else:
                time_list = []

            # ---眨眼判斷---
            blink_ratio = blink(mesh_points[RIGHT_EYE], mesh_points[LEFT_EYE])
            # ---打哈欠判斷---
            lip_ratio = yawn(mesh_points[LIPS])

            # ---眨眼條件判斷---
            if blink_ratio > 4.0:
                CEF_COUNTER1 += 1
            else:
                if CEF_COUNTER1 > CLOSE_EYES_FRAME:
                    blink_times += 1
                    CEF_COUNTER1 = 0

            # ---打哈欠條件判斷---
            if lip_ratio < 0.9:
                CEF_COUNTER2 += 1
            else:
                if CEF_COUNTER2 > CLOSE_YAWN_FRAME:
                    yawn_times += 1
                    CEF_COUNTER2 = 0

            # ---每五秒重置打哈欠及眨眼次數---
            if (c % frameRate == 0):
                blink_times = 0
                yawn_times = 0
            c += 1
            schedule.every(5).seconds.do(blink_speak,blink_times)
            schedule.every(5).seconds.do(yawn_speak, yawn_times)
            schedule.run_pending()
            schedule.clear()
            blink_times = blink_speak(blink_times)
            yawn_times = yawn_speak(yawn_times)



            # ---draw on frame---
            # cv2.circle(image_gamma_correct, left_eye_center, int(left_eye_r), (255, 0, 0), 1, cv2.LINE_AA)
            # cv2.circle(image_gamma_correct, right_eye_center, int(right_eye_r), (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image_gamma_correct, str(eye_pos), (10, 40), 5, 2, (0, 0, 0), 2)
            cv2.putText(image_gamma_correct, f"yawntimes{yawn_times}", (10, 110), 5, 2, (0, 0, 0), 2)
            cv2.putText(image_gamma_correct, f"blinktimes{blink_times}", (10, 80), 5, 2, (0, 0, 0), 2)
            cv2.putText(image_gamma_correct, f"FPS{frameRate}", (10, 150), 5, 2, (0, 0, 0), 2)
            cv2.putText(image_gamma_correct, str(eye_position_time), (10, 170), 5, 2, (0, 0, 0), 2)


            # ---show frame---
            cv2.imshow("img", image_gamma_correct)
        except:
            speak("未偵測到人臉")
            time.sleep(3)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
