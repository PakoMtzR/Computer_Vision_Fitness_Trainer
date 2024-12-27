import cv2
import mediapipe as mp
import numpy
from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

image = cv2.imread("imgs/pose_example.jpg")
height, width, _ = image.shape
print(image.shape)

# Cambiamos la imagen de BGR a RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5) as pose:
    results = pose.process(image_rgb)

    print("Landmarks:", results.pose_landmarks)

    if results.pose_landmarks:
        # Obtenemos puntos de las posiciones de interes [0-1] y convertimos a int
        screen_position = [
            [int(point.x * width), int(point.y * height)]
            for point in [results.pose_landmarks.landmark[i] for i in [23, 25, 27]]
        ]

        # Dibujamos lineas
        cv2.line(image, screen_position[0], screen_position[1], (255,255,255), 3)
        cv2.line(image, screen_position[1], screen_position[2], (255,255,255), 3)

        # Dibujamos los puntos
        for point in screen_position:
            cv2.circle(image, (point[0], point[1]), 6, (255,140,78), 2)

    # Dibuja todos los puntos y traza lineas conectoras
    # if results.pose_landmarks:
    #     mp_drawing.draw_landmarks(
    #         image, 
    #         results.pose_landmarks, 
    #         mp_pose.POSE_CONNECTIONS,
    #         mp_drawing.DrawingSpec(color=(255,140,78), thickness=2, circle_radius=3),
    #         mp_drawing.DrawingSpec(color=(230,230,230), thickness=2)
    #         )


    cv2.imshow("Pose example", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()