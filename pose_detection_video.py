import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video-Streaming
cap = cv2.VideoCapture(0)

# Video
cap = cv2.VideoCapture("videos/video.mp4")

with mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5) as pose:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        # Nuevo alto deseado
        new_height = 480

        # Calcular el nuevo ancho manteniendo la proporción
        aspect_ratio = frame.shape[1] / frame.shape[0]
        new_width = int(new_height * aspect_ratio)

        # Redimensionar la imagen
        new_size = (new_width, new_height)
        frame = cv2.resize(frame, new_size, cv2.INTER_AREA)

        # Obtenemos dimensiones de la imagen
        height, width, _ = frame.shape

        # Cambiamos la imagen de BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb)

        # Dibuja todos los puntos y traza lineas conectoras
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,140,78), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(230,230,230), thickness=2)
                )


        cv2.imshow("Pose video example", frame)

        # Esperamos la tecla ESC para terminar el programa
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()