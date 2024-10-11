import cv2
import numpy as np
import mediapipe as mp
import math
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# LANDMARKS
RIGHT_EYE = [33, 246, 161, 160, 159, 158, 157,
             173, 133, 155, 154, 153, 145, 144, 163, 7]
LEFT_EYE = [362, 466, 388, 387, 386, 385, 384,
            398, 363, 382, 381, 380, 374, 373, 390, 249]
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]  # RIGHT EYE RIGHT MOST LANDMARK
L_H_RIGHT = [133]  # RIGHT EYE LEFT MOST LANDMARK
R_H_LEFT = [362]  # LEFT EYE RIGHT MOST LANDMARK
R_H_RIGHT = [263]  # LEFT EYE LEFT MOST LANDMARK


def eye_aspect_ratio(landmarks, right_indices, left_indices):
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]

    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclidean_distance(rh_right, rh_left)
    rvDistance = euclidean_distance(rv_top, rv_bottom)

    lvDistance = euclidean_distance(lv_top, lv_bottom)
    lhDistance = euclidean_distance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio


def euclidean_distance(p1, p2):
    x1, y1 = p1.ravel()
    x2, y2 = p2.ravel()
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


# def right_iris_position(iris_center, right_point, left_point):
#     center_to_right_dist = euclidean_distance(iris_center, right_point)
#     total_dist = euclidean_distance(right_point, left_point)
#     ratio = center_to_right_dist / total_dist
#     iris_pos_text = ""
#     if ratio <= 0.42:
#         iris_pos_text = "RIGHT"
#     elif ratio > 0.42 and ratio <= 0.57:
#         iris_pos_text = "CENTER"
#     else:
#         iris_pos_text = "LEFT"
#     return iris_pos_text, ratio

def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_dist = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist / total_dist
    iris_pos_text = ""
    if ratio <= 0.42:
        iris_pos_text = "RIGHT"
    elif ratio > 0.42 and ratio <= 0.57:
        iris_pos_text = "CENTER"
    else:
        iris_pos_text = "LEFT"
    return iris_pos_text, ratio


# def left_iris_position(iris_center, right_point, left_point):
#     center_to_right_dist = euclidean_distance(iris_center, right_point)
#     total_dist = euclidean_distance(right_point, left_point)
#     ratio = center_to_right_dist / total_dist
#     iris_pos_text = ""
#     if ratio <= 0.42:
#         iris_pos_text = "RIGHT"
#     elif ratio > 0.42 and ratio <= 0.57:
#         iris_pos_text = "CENTER"
#     else:
#         iris_pos_text = "LEFT"
#     return iris_pos_text, ratio


cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()

    start = time.time()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False

    results = face_mesh.process(image)

    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape

    if results.multi_face_landmarks:
        mesh_points = np.array([
            np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
            for p in results.multi_face_landmarks[0].landmark
        ])

        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(
            mesh_points[RIGHT_IRIS])
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        cv2.circle(image, center_left, int(l_radius),
                   (255, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(image, center_right, int(
            r_radius), (255, 0, 255), 1, cv2.LINE_AA)

        # Right eye most landmarks
        cv2.circle(image, mesh_points[R_H_RIGHT[0]],
                   3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, mesh_points[R_H_LEFT[0]],
                   3, (0, 255, 255), -1, cv2.LINE_AA)

        # Left eye most landmarks
        cv2.circle(image, mesh_points[L_H_RIGHT[0]],
                   3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, mesh_points[L_H_LEFT[0]],
                   3, (0, 255, 255), -1, cv2.LINE_AA)

        right_iris_pos, right_eye_ratio = iris_position(
            center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT[0]])

        left_iris_pos, left_eye_ratio = iris_position(
            center_left, mesh_points[L_H_RIGHT], mesh_points[L_H_LEFT[0]])

        cv2.putText(image, f"right eye pos: {right_iris_pos}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"right eye ratio: {right_eye_ratio:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(image, f"left eye pos: {left_iris_pos}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"left eye ratio: {left_eye_ratio:.2f}", (
            10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Get average of the ratios, and determine the position of the eyes
        avg_ratio = (right_eye_ratio + left_eye_ratio) / 2
        eye_pos = ""
        if avg_ratio <= 0.42:
            eye_pos = "RIGHT"
        elif avg_ratio > 0.42 and avg_ratio <= 0.57:
            eye_pos = "CENTER"
        else:
            eye_pos = "LEFT"
        cv2.putText(image, f"eye pos: {eye_pos}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw RIGHT_IRIS
        for idx in RIGHT_IRIS:
            cv2.circle(image, mesh_points[idx], 3,
                       (255, 0, 0), -1, cv2.LINE_AA)

        # Draw LEFT_IRIS
        for idx in LEFT_IRIS:
            cv2.circle(image, mesh_points[idx], 3,
                       (255, 0, 0), -1, cv2.LINE_AA)

        # Calculate EAR
        ear = eye_aspect_ratio(mesh_points, RIGHT_EYE, LEFT_EYE)
        cv2.putText(image, f"EAR: {ear:.2f}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if ear > 5.5:
            cv2.putText(image, "Eyes Closed", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "Eyes Opened", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    end = time.time()
    print("FPS: ", 1 / (end - start))


cap.release()
cv2.destroyAllWindows()
