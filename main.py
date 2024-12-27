import cv2
import mediapipe as mp
import numpy as np

# Inicializzamos componentes de mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5)

# Inicializamos la camara
cap = cv2.VideoCapture("videos/video.mp4")
# cap = cv2.VideoCapture(0)

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

# Esta funcion perimite calcular el angulo entre tres puntos
def calculate_angle(a,b,c):
    a = np.array(a) 
    b = np.array(b)  
    c = np.array(c)  

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return int(angle)

counter = 0
stage = False  # "bajando" o "subiendo"

while True:
    # Capturamos una imagen de la camara
    ret, frame = cap.read()
    if ret == False:
        break
    
    # Redimensionamos la imagen
    frame = resize_image(frame, new_width=400)
    # frame = cv2.flip(frame, 1)

    # Obtenemos las nuevas dimensiones de la imagen
    height, width, _ = frame.shape

    # Covertimos la imagen de BGR a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen
    results = pose.process(frame_rgb)

    # Si existe puntos de interes
    if results.pose_landmarks:
        # Extraemos los puntos clave
        landmarks = results.pose_landmarks.landmark

        hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x*width),
               int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y*height))
        knee = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x*width),
                int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y*height))
        ankle = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x*width),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y*height))

        angle = calculate_angle(hip, knee, ankle)

        # Contar las sentadillas
        if angle > 160:
            stage = True
        if angle < 90 and stage == True:
            stage = False
            counter += 1
            print(f"Sentadillas: {counter}")

        # Visualizacion
        # Dibujamos lineas
        cv2.line(frame, hip, ankle, (255,255,255), 3)

        # Dibujamos circulos en los puntos de articulacion
        # cv2.circle(image, center_coordinates, radius, color, thickness)
        cv2.circle(frame, hip, 6, (255,0,0), 2)
        cv2.circle(frame, knee, 6, (255,0,0), 2)
        cv2.circle(frame, ankle, 6, (255,0,0), 2)

        # Escribimos texto
        cv2.putText(frame, f"Modo: {stage}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Modo: {stage}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        # Mostramos video
        cv2.imshow("Video", frame)

        # Menu de teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            print("sentadillas")
        elif key == ord('2'):
            print("lagartijas")
        elif key == 27:  # Salir
            break

cap.release()
cv2.destroyAllWindows()