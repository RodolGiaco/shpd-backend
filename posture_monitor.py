import cv2
import time
import math as m
import mediapipe as mp  # RPI3 FIX
import argparse
import json  # JSON FIX
import os    # JSON FIX
from datetime import datetime
from api.database import SessionLocal
from api.models import MetricaPostural
import logging
import redis

r = redis.Redis(host='redis', port=6379, decode_responses=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
class PostureMonitor:
    def __init__(self,  session_id: str):
        self.mp_drawing = mp.solutions.drawing_utils
        self.session_id = session_id
        self.mp_pose = mp.solutions.pose  # RPI3 FIX
        self.args = self.parse_arguments()

        if os.path.exists("calibration.json"):
            with open("calibration.json", "r") as f:
                data = json.load(f)
                self.args.offset_threshold = data.get("offset_threshold", self.args.offset_threshold)
                self.args.neck_angle_threshold = data.get("neck_angle_threshold", self.args.neck_angle_threshold)
                self.args.torso_angle_threshold = data.get("torso_angle_threshold", self.args.torso_angle_threshold)
                self.args.time_threshold = data.get("time_threshold", self.args.time_threshold)
            print("\U0001F4E5 Umbrales cargados desde calibration.json:")  # JSON FIX
            print(f"Offset: {self.args.offset_threshold}, Neck: {self.args.neck_angle_threshold}, Torso: {self.args.torso_angle_threshold}, Tiempo: {self.args.time_threshold}")  # JSON FIX
        else:
            print("⚠️  calibration.json no encontrado. Usando valores por defecto o línea de comandos.")  # JSON FIX

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.good_frames = 0
        self.bad_frames = 0
        self.all_frames = 0
        self.flag_alert = True
        self.flag_transition = True
        

    def findDistance(self, x1, y1, x2, y2):
        dist = m.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return dist

    def findAngle(self, x1, y1, x2, y2):
        theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
        degree = int(180/m.pi) * theta
        return degree

    def sendWarning(self):
        pass

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Posture Monitor with MediaPipe')
        parser.add_argument('--video', type=str, default=0, help='Path to the input video file. If not provided, the webcam will be used.')
        parser.add_argument('--offset-threshold', type=float, default=100, help='Threshold value for shoulder alignment.')  # JSON FIX
        parser.add_argument('--neck-angle-threshold', type=float, default=25, help='Threshold value for neck inclination angle.')  # JSON FIX
        parser.add_argument('--torso-angle-threshold', type=float, default=10, help='Threshold value for torso inclination angle.')  # JSON FIX
        parser.add_argument('--time-threshold', type=int, default=10, help='Time threshold for triggering a posture alert.')
        return parser.parse_args()



    def save_data_to_redis(self, datos: dict):
            key = f"metricas:{self.session_id}"
            r.rpush(key, json.dumps(datos))
            r.ltrim(key, -50, -1)  # conserva solo las últimas 50 métricas

    
    def process_frame(self, image):
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = self.pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        lm = keypoints.pose_landmarks
        lmPose = self.mp_pose.PoseLandmark
        fps = 15
        buffer_key = f"shpd-data:{self.session_id}"
        if lm is None:
            delta = 1.0 / fps       
            r.hincrbyfloat(buffer_key, "tiempo_parado", round(delta,1))
            return image

        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        r_ear_x = int(lm.landmark[lmPose.RIGHT_EAR].x * w)
        r_ear_y = int(lm.landmark[lmPose.RIGHT_EAR].y * h)
        r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
        r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)

        neck_inclination = self.findAngle(r_shldr_x, r_shldr_y, r_ear_x, r_ear_y)
        torso_inclination = self.findAngle(r_hip_x, r_hip_y, r_shldr_x, r_shldr_y)

        angle_text_string_neck = 'Neck inclination: ' + str(int(neck_inclination))
        angle_text_string_torso = 'Torso inclination: ' + str(int(torso_inclination))
        
        if neck_inclination < self.args.neck_angle_threshold and torso_inclination < self.args.torso_angle_threshold:
            self.bad_frames = 0
            self.good_frames += 1
            self.all_frames += 1
            color = (127, 233, 100)
                
            # # HINCRBY crea el campo si no existe y suma 1
            r.hincrby(buffer_key, "good_frames", 1)
            self.flag_transition = True
            self.flag_alert = True
            
        else:
            self.good_frames = 0
            self.bad_frames += 1
            self.all_frames += 1
            color = (50, 50, 255)
            
            # HINCRBY crea el campo si no existe y suma 1
            r.hincrby(buffer_key, "bad_frames", 1)
       
            if self.flag_transition:
                r.hincrby(buffer_key, "transiciones_malas", 1)
                self.flag_transition = False
        
        cv2.putText(image, angle_text_string_neck, (10, 30), self.font, 0.6, color, 2)
        cv2.putText(image, angle_text_string_torso, (10, 60), self.font, 0.6, color, 2)
        cv2.putText(image, str(int(neck_inclination)), (r_shldr_x + 10, r_shldr_y), self.font, 0.9, color, 2)
        cv2.putText(image, str(int(torso_inclination)), (r_hip_x + 10, r_hip_y), self.font, 0.9, color, 2)

        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, (255, 255, 255), 2)
        cv2.circle(image, (r_ear_x, r_ear_y), 7, (255, 255, 255), 2)
        cv2.circle(image, (r_shldr_x, r_shldr_y - 100), 7, (255, 255, 255), 2)
        cv2.circle(image, (r_hip_x, r_hip_y), 7, (0, 255, 255), -1)
        cv2.circle(image, (r_hip_x, r_hip_y - 100), 7, (0, 255, 255), -1)

        cv2.line(image, (r_shldr_x, r_shldr_y), (r_ear_x, r_ear_y), color, 2)
        cv2.line(image, (r_shldr_x, r_shldr_y), (r_shldr_x, r_shldr_y - 100), color, 2)
        cv2.line(image, (r_hip_x, r_hip_y), (r_shldr_x, r_shldr_y), color, 2)
        cv2.line(image, (r_hip_x, r_hip_y), (r_hip_x, r_hip_y - 100), color, 2)

        good_time = (1 / fps) * self.good_frames
        bad_time = (1 / fps) * self.bad_frames
        r.hincrbyfloat(buffer_key, "tiempo_sentado", round((1.0 / fps) , 1))
        
        if good_time > 0:
            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
            cv2.putText(image, time_string_good, (10, h - 20), self.font, 0.9, (127, 255, 0), 2)
        else:
            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
            cv2.putText(image, time_string_bad, (10, h - 20), self.font, 0.9, (50, 50, 255), 2)

        if bad_time > self.args.time_threshold:
            if self.flag_alert:
                self.sendWarning()
                r.hincrby(buffer_key, "alert_count", 1)
                self.flag_alert = False

        try:
            accum = r.hgetall(buffer_key)
            good_acc = int(accum.get("good_frames", 0))
            bad_acc  = int(accum.get("bad_frames",  0))
            alert_count  = int(accum.get("alert_count",  0))
            transiciones_malas  = int(accum.get("transiciones_malas",  0))
            tiempo_sentado  = float(accum.get("tiempo_sentado",  0))
            tiempo_parado  = float(accum.get("tiempo_parado",  0))

            total    = good_acc + bad_acc or 1
            percentage_good = round(good_acc / total * 100, 1)
            percentage_bad  = round(bad_acc  / total * 100, 1)
            datos = {
                "actual": "menton_en_mano",
                "porcentaje_correcta": percentage_good,
                "porcentaje_incorrecta": percentage_bad,
                "transiciones_malas": transiciones_malas,
                "tiempo_sentado": tiempo_sentado,
                "tiempo_parado": tiempo_parado,
                "alertas_enviadas": alert_count,
            }
        except Exception:
            logger.exception("Failed metrics")
        self.save_data_to_redis(datos)

        return image

    def run(self):
        cap = cv2.VideoCapture(self.args.video) if self.args.video else cv2.VideoCapture(0)

        while True:
            success, image = cap.read()
            if not success:
                print("Null.Frames")
                break

            image = self.process_frame(image)
            cv2.imshow('MediaPipe Pose', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()