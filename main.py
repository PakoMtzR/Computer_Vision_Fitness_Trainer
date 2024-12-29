import cv2
import mediapipe as mp
import numpy as np

class ExerciseDetector:
    def __init__(self):
        # Inicializzamos componentes de mediapipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5)
        
        # Inicializamos el contador de repeticiones
        self.reps = 0
        self.is_down = False

        # Mapeo de ejercicios a funciones
        self.exercise_functions = {
            "SQUAT": self.detect_squats,
            "PUSH-UP": self.detect_push_up
        }
        # Generamos la lista de ejercicios
        self.list_exercises = list(self.exercise_functions.keys())
        # Asignamos el ejercicio por default
        self.current_exercise = self.list_exercises[0]

    # Esta funcion perimite calcular el angulo entre tres puntos
    @staticmethod
    def calculate_angle(a,b,c):
        a = np.array(a) 
        b = np.array(b)  
        c = np.array(c)  

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360.0 - angle

        return int(angle)
    
    def detect_squats(self, landmarks, width, height):
        # Extraemos las coordenadas de interes para las sentadillas
        left_hip = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * width),
                    int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * height))
        left_knee = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * width),
                     int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * height))
        left_ankle = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width),
                      int(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height))
        right_hip = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * width),
                     int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * height))
        right_knee = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x * width),
                      int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y * height))
        right_ankle = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * width),
                       int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * height))
        
        # Logica para el conteo de sentadillas
        angle_left = self.calculate_angle(left_hip, left_knee, left_ankle)
        angle_right = self.calculate_angle(right_hip, right_knee, right_ankle)

        if self.is_down and any(angle > 100 for angle in (angle_left, angle_right)):
            self.is_down = False
            self.reps += 1
        elif not self.is_down and any(angle < 40 for angle in (angle_left, angle_right)):
            self.is_down = True
    
    def detect_push_up(self, landmarks, width, height):
        # Extraemos las coordenadas de interes para las lagartijas
        left_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width),
                         int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height))
        left_elbow = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width),
                      int(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height))
        left_wrist = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x * width),
                      int(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y * height))
        right_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width),
                          int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height))
        right_elbow = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * width),
                       int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * height))
        right_wrist = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * width),
                       int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * height))

        # Logica para el conteo de sentadillas
        angle_left = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        angle_right = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        print(angle_left)
        if self.is_down and any(angle > 160 for angle in (angle_left, angle_right)):
            self.is_down = False
            self.reps += 1
        elif not self.is_down and any(angle < 90 for angle in (angle_left, angle_right)):
            self.is_down = True
            
    def draw(self):
        # cv2.line(img, point1, point2, color, thinkless)
        # Dibujamos circulos en los puntos de articulacion
        # cv2.circle(image, center_coordinates, radius, color, thickness)
        pass

    def write_info(self, frame):
        # Obtenemos dimensiones de la imagen
        height, width, _ = frame.shape

        # Dibujamos recuadros
        cv2.rectangle(frame, (0,0), (280,70), (28,10,0), -1)
        cv2.rectangle(frame, (0,height-30), (width,height), (28,10,0), -1)
        # Escribimos texto
        cv2.putText(frame, f"Exercise: {self.current_exercise}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (221,179,144), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Reps: {self.reps}", (10,60), cv2.FONT_HERSHEY_SIMPLEX , 1, (221,179,144), 2, cv2.LINE_AA)
        cv2.putText(frame, "[1]SQUAT [2]PUSH-UP [3]BICEPS", (10, height-10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)

    def process_frame(self, frame):
        # Obtenemos dimensiones de la imagen
        height, width, _ = frame.shape

        # Cambiamos el espacio de color de BGR a RBG porque mediapipe solo trabaja con RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesamos la imagen
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            self.exercise_functions[self.current_exercise](landmarks, width, height)
            self.write_info(frame)
        
         # Dibuja puntos clave en la imagen
        self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return frame
    
    def change_exercise(self, new_exercise_index):
        self.current_exercise = self.list_exercises[new_exercise_index]
        self.reps = 0


class VideoProcessor:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.detector = ExerciseDetector()

    @staticmethod
    def resize_image(image, new_height=0, new_width=0):
        # Obtenemos las dimensiones de la imagen
        image_height, image_width, _ = image.shape

        # Calculamos el nuevo ancho o alto de la imagen manteniendo su proporcion
        if new_height != 0 and new_width == 0:
            new_width = int((image_height/image_width)*new_height)
        elif new_height == 0 and new_width != 0:
            new_height = int((image_width/image_height)*new_width)
        else:
            return None
        
        # Nuevo tamaño de la imagen
        new_size = (new_height, new_width)

        # Retornamos la nueva imagen redimensionada
        return cv2.resize(image, new_size, cv2.INTER_AREA)
    
    def run(self):
        while True:
            # Capturamos una imagen
            ret, frame = self.cap.read()
            if not ret:
                break

            # Redimensionamos la imagen
            frame = self.resize_image(frame, new_width=300)
            if self.source == 0:
                frame = cv2.flip(frame, 1)

            # Procesamos la imagen
            frame = self.detector.process_frame(frame)

            # Visualizacion
            cv2.imshow("Computer Vision Trainer", frame)

            # Menu de teclas
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Salir con ESC
                break   
            elif key == ord('1'):
                self.detector.change_exercise(0)
            elif key == ord('2'):
                self.detector.change_exercise(1)
            
        # Liberamos todos los recursos
        self.cap.release()
        cv2.destroyAllWindows()


# Iniciamos el procesamiento del video
if __name__ == "__main__":
    app = VideoProcessor(source="videos/push_up.mp4")
    # app = VideoProcessor(source=0)
    # app = VideoProcessor(source="videos/examples/blanche.ts")
    app.run()