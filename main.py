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
            "PUSH-UP": self.detect_push_up,
            "PULL-UP": self.detect_pull_up,
            "BICEPS": self.detect_biceps,
            "CRUNCHES": self.detect_crunches
        }
        # Generamos la lista de ejercicios
        self.list_exercises = list(self.exercise_functions.keys())
        # Asignamos el ejercicio por default
        self.current_exercise = self.list_exercises[0]

    # Esta funcion perimite calcular el angulos de una lista de tres puntos en R2
    @staticmethod
    def calculate_angle(positions):
        # Desempaquetar las posiciones de la tupla
        a, b, c = positions
        
        # Convertir las posiciones en arrays de NumPy
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Calcular el ángulo
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        # Asegurarse de que el ángulo esté entre 0 y 180 grados
        if angle > 180.0:
            angle = 360.0 - angle

        return int(angle)
    
    def get_angles(self, frame, landmarks, width, height, *indexes_list):
        angles = []
        for indexes in indexes_list:
            not_visible_landmarks = [landmarks[i].visibility < 0.6 for i in indexes]
            if any(not_visible_landmarks):
                continue
            # Extraemos las coordenadas de interes
            points = [
                [int(point.x * width), int(point.y * height)]
                for point in [landmarks[i] for i in indexes]
            ]
            # Dibujamos los puntos y las lineas de union
            for i, point in enumerate(points):
                if i != 0:
                    cv2.line(frame, points[i-1], points[i], (255,255,255), 2)
                cv2.circle(frame, point, 5, (255,0,0), 3)

            # Calculamos el angulo que forma cada lista de posiciones
            # y los guardamos en la lista de angulos
            angles.append(self.calculate_angle(points))
        # print(angles)
        return angles
    
    def detect_squats(self, frame, landmarks, width, height):
        # Extraemos las angulos de interes para las sentadillas
        angles = self.get_angles(frame, landmarks, width, height, [23, 25, 27], [24, 26, 28])
        # Logica para el conteo de sentadillas
        if self.is_down and any(angle > 150 for angle in angles):
            self.is_down = False
            self.reps += 1
        elif not self.is_down and any(angle < 90 for angle in angles):
            self.is_down = True
    
    def detect_push_up(self, frame, landmarks, width, height):
        # Extraemos las angulos de interes para las lagartijas
        angles = self.get_angles(frame, landmarks, width, height, [11, 13, 15], [12, 14, 16])

        # Logica para el conteo de lajartijas
        if self.is_down and any(angle > 150 for angle in angles):
            self.is_down = False
            self.reps += 1
        elif not self.is_down and any(angle < 90 for angle in angles):
            self.is_down = True
    
    def detect_pull_up(self, frame, landmarks, width, height):
        # Extraemos las angulos de interes para las dominadas
        angles = self.get_angles(frame, landmarks, width, height, [11, 13, 15], [12, 14, 16])

        # Logica para el conteo de lajartijas
        if self.is_down and any(angle > 140 for angle in angles):
            self.is_down = False
            self.reps += 1
        elif not self.is_down and any(angle < 40 for angle in angles):
            self.is_down = True
    
    def detect_biceps(self, frame, landmarks, width, height):
        # Extraemos las angulos de interes para los biceps
        angles = self.get_angles(frame, landmarks, width, height, [11, 13, 15], [12, 14, 16])

        # Logica para el conteo de lajartijas
        if self.is_down and any(angle > 140 for angle in angles):
            self.is_down = False
            self.reps += 1
        elif not self.is_down and any(angle < 35 for angle in angles):
            self.is_down = True

    def detect_crunches(self, frame, landmarks, width, height):
        # Extraemos las angulos de interes para las abdominales
        angles = self.get_angles(frame, landmarks, width, height, [11, 23, 25], [12, 24, 26])

        # Logica para el conteo de lajartijas
        if self.is_down and any(angle > 100 for angle in angles):
            self.is_down = False
            self.reps += 1
        elif not self.is_down and any(angle < 50 for angle in angles):
            self.is_down = True

    def write_info(self, frame):
        # Obtenemos dimensiones de la imagen
        height, width, _ = frame.shape
        # Dibujamos recuadros
        cv2.rectangle(frame, (0,0), (300,70), (28,10,0), -1)
        cv2.rectangle(frame, (0,height-30), (width,height), (28,10,0), -1)
        # Escribimos texto
        cv2.putText(frame, f"Mode: {self.current_exercise}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (221,179,144), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Reps: {self.reps}", (10,60), cv2.FONT_HERSHEY_SIMPLEX , 1, (221,179,144), 2, cv2.LINE_AA)
        cv2.putText(frame, "[1]SQUAT [2]PUSH-UP [3]PULL-UP [4]BICEPS [5]CRUNCH", (10, height-10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)

    def process_frame(self, frame):
        # Obtenemos dimensiones de la imagen
        height, width, _ = frame.shape
        # Cambiamos el espacio de color de BGR a RBG porque mediapipe solo trabaja con RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Procesamos la imagen
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            self.exercise_functions[self.current_exercise](frame, landmarks, width, height)
            self.write_info(frame)
        
        # Dibuja puntos clave en la imagen
        # self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
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
        
        # Retornamos la nueva imagen redimensionada
        return cv2.resize(image, (new_height, new_width), cv2.INTER_AREA)
    
    def run(self):
        while True:
            # Capturamos una imagen
            ret, frame = self.cap.read()
            if not ret:
                break

            # Redimensionamos la imagen
            frame = self.resize_image(frame, new_width=480)
            if self.source == 0:
                frame = cv2.flip(frame, 1)

            # Procesamos la imagen
            frame = self.detector.process_frame(frame)
            # Visualizamos la image procesada
            cv2.imshow("Computer Vision Trainer", frame)

            # Menu de teclas
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Salir con ESC
                break   
            elif key == ord('1'):
                self.detector.change_exercise(0)    # SQUAT
            elif key == ord('2'):
                self.detector.change_exercise(1)    # PUSH-UP
            elif key == ord('3'):
                self.detector.change_exercise(2)    # PULL-UP
            elif key == ord('4'):
                self.detector.change_exercise(3)    # BICEPS
            elif key == ord('5'):
                self.detector.change_exercise(4)    # CRUNCH

        # Liberamos todos los recursos
        self.cap.release()
        cv2.destroyAllWindows()


# Iniciamos el procesamiento del video
if __name__ == "__main__":
    app = VideoProcessor(source="videos/crunches/1.mp4")
    # app = VideoProcessor(source=0)
    app.run()