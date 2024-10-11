import cv2
import mediapipe as mp
import numpy as np
import time
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

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

    if rvDistance != 0:
        reRatio = rhDistance / rvDistance
    else:
        reRatio = 0

    if lvDistance != 0:
        leRatio = lhDistance / lvDistance
    else:
        leRatio = 0

    ratio = (reRatio+leRatio)/2
    return ratio


def euclidean_distance(p1, p2):
    x1, y1 = p1.ravel()
    x2, y2 = p2.ravel()
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


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
    face_3d = []
    face_2d = []

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

        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Draw the selected landmarks
                    cv2.circle(image, (x, y), drawing_spec.circle_radius,
                               (0, 255, 0), drawing_spec.thickness)
                    # draw the landmark numbers
                    cv2.putText(image, str(idx), (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

                    # Get 2d coordinate
                    face_2d.append([x, y])

                    # get 3d coordinate
                    face_3d.append([x, y, lm.z])

            # Convert to np array
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]], dtype=np.float64)

            # The distance matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotation matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the Y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See the user head tilting
            if x > 10 and y > 10:
                text = "Upper Right"
            elif x < -10 and y > 10:
                text = "Lower Right"
            elif x < -10 and y < -10:
                text = "Lower Left"
            elif x > 10 and y < -10:
                text = "Upper Left"
            elif y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Looking Forward"

            # Display the nose direction
            nose_3d_proj, jacobian = cv2.projectPoints(
                nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # add text
            cv2.putText(image, text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "X: " + str(np.round(x, 2)), (500, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "Y: " + str(np.round(y, 2)), (500, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "Z: " + str(np.round(z, 2)), (500, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        print("FPS: ", fps)

        # mp_drawing.draw_landmarks(image=image,
        #                           landmark_list=face_landmarks,
        #                           connections=mp_face_mesh.FACEMESH_TESSELATION,
        #                           landmark_drawing_spec=drawing_spec,
        #                           connection_drawing_spec=drawing_spec)

    cv2.imshow('Face Mesh', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
