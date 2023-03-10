import mediapipe as mp
import cv2
import numpy as np
import av


# 各部位の座標を取得
def get_keypoints(results, height, width):
    nose_x = int(results.pose_landmarks.landmark[0].x * width)
    nose_y = int(results.pose_landmarks.landmark[0].y * height)
    nose_xy = [nose_x, nose_y]

    left_shoulder_x = int(results.pose_landmarks.landmark[11].x * width)
    left_shoulder_y = int(results.pose_landmarks.landmark[11].y * height)
    left_shoulder_xy = [left_shoulder_x, left_shoulder_y]

    right_shoulder_x = int(results.pose_landmarks.landmark[12].x * width)
    right_shoulder_y = int(results.pose_landmarks.landmark[12].y * height)
    right_shoulder_xy = [right_shoulder_x, right_shoulder_y]

    left_elbow_x = int(results.pose_landmarks.landmark[13].x * width)
    left_elbow_y = int(results.pose_landmarks.landmark[13].y * height)
    left_elbow_xy = [left_elbow_x, left_elbow_y]

    right_elbow_x = int(results.pose_landmarks.landmark[14].x * width)
    right_elbow_y = int(results.pose_landmarks.landmark[14].y * height)
    right_elbow_xy = [right_elbow_x, right_elbow_y]

    left_wrist_x = int(results.pose_landmarks.landmark[15].x * width)
    left_wrist_y = int(results.pose_landmarks.landmark[15].y * height)
    left_wrist_xy = [left_wrist_x, left_wrist_y]

    right_wrist_x = int(results.pose_landmarks.landmark[16].x * width)
    right_wrist_y = int(results.pose_landmarks.landmark[16].y * height)
    right_wrist_xy = [right_wrist_x, right_wrist_y]

    left_hip_x = int(results.pose_landmarks.landmark[23].x * width)
    left_hip_y = int(results.pose_landmarks.landmark[23].y * height)
    left_hip_xy = [left_hip_x, left_hip_y]

    right_hip_x = int(results.pose_landmarks.landmark[24].x * width)
    right_hip_y = int(results.pose_landmarks.landmark[24].y * height)
    right_hip_xy = [right_hip_x, right_hip_y]

    left_knee_x = int(results.pose_landmarks.landmark[25].x * width)
    left_knee_y = int(results.pose_landmarks.landmark[25].y * height)
    left_knee_xy = [left_knee_x, left_knee_y]

    right_knee_x = int(results.pose_landmarks.landmark[26].x * width)
    right_knee_y = int(results.pose_landmarks.landmark[26].y * height)
    right_knee_xy = [right_knee_x, right_knee_y]

    left_ankle_x = int(results.pose_landmarks.landmark[27].x * width)
    left_ankle_y = int(results.pose_landmarks.landmark[27].y * height)
    left_ankle_xy = [left_ankle_x, left_ankle_y]

    right_ankle_x = int(results.pose_landmarks.landmark[28].x * width)
    right_ankle_y = int(results.pose_landmarks.landmark[28].y * height)
    right_ankle_xy = [right_ankle_x, right_ankle_y]

    left_heel_x = int(results.pose_landmarks.landmark[29].x * width)
    left_heel_y = int(results.pose_landmarks.landmark[29].y * height)
    left_heel_xy = [left_heel_x, left_heel_y]

    right_heel_x = int(results.pose_landmarks.landmark[30].x * width)
    right_heel_y = int(results.pose_landmarks.landmark[30].y * height)
    right_heel_xy = [right_heel_x, right_heel_y]

    left_foot_index_x = int(results.pose_landmarks.landmark[31].x * width)
    left_foot_index_y = int(results.pose_landmarks.landmark[31].y * height)
    left_foot_index_xy = [left_foot_index_x, left_foot_index_y]

    right_foot_index_x = int(results.pose_landmarks.landmark[32].x * width)
    right_foot_index_y = int(results.pose_landmarks.landmark[32].y * height)
    right_foot_index_xy = [right_foot_index_x, right_foot_index_y]

    return nose_xy, left_shoulder_xy, right_shoulder_xy, left_elbow_xy, right_elbow_xy, left_wrist_xy, right_wrist_xy,\
           left_hip_xy, right_hip_xy, left_knee_xy, right_knee_xy, left_ankle_xy, right_ankle_xy,\
           left_heel_xy, right_heel_xy, left_foot_index_xy, right_foot_index_xy

# 各部位の座標を描画
def draw_keypoint(image, radius, color, thickness, nose_xy, left_shoulder_xy, right_shoulder_xy,\
                  left_elbow_xy, right_elbow_xy, left_wrist_xy, right_wrist_xy, left_hip_xy, right_hip_xy,\
                  left_knee_xy, right_knee_xy, left_ankle_xy, right_ankle_xy, left_heel_xy, right_heel_xy,\
                  left_foot_index_xy, right_foot_index_xy, height, width):
    # 背景
    size = height, width, 3
    image = np.zeros(size, dtype=np.uint8)
    image.fill(255)

    # 頭
    cv2.circle(image, (nose_xy[0], nose_xy[1]), radius, color, thickness=-1)

    # 胴体
    points = np.array([(left_shoulder_xy), (right_shoulder_xy), (right_hip_xy), (left_hip_xy)])
    cv2.fillConvexPoly(image, points, color=color, lineType=cv2.LINE_8, shift=0)

    # 腕, 脚
    cv2.line(image, (left_shoulder_xy[0], left_shoulder_xy[1]), (left_elbow_xy[0], left_elbow_xy[1]), color, thickness, lineType=cv2.LINE_8, shift=0)
    cv2.line(image, (left_elbow_xy[0], left_elbow_xy[1]), (left_wrist_xy[0], left_wrist_xy[1]), color, thickness, lineType=cv2.LINE_8, shift=0)
    cv2.line(image, (right_shoulder_xy[0], right_shoulder_xy[1]), (right_elbow_xy[0], right_elbow_xy[1]), color, thickness, lineType=cv2.LINE_8, shift=0)
    cv2.line(image, (right_elbow_xy[0], right_elbow_xy[1]), (right_wrist_xy[0], right_wrist_xy[1]), color, thickness, lineType=cv2.LINE_8, shift=0)

    cv2.line(image, (left_hip_xy[0], left_hip_xy[1]), (left_knee_xy[0], left_knee_xy[1]), color, thickness, lineType=cv2.LINE_8, shift=0)
    cv2.line(image, (left_knee_xy[0], left_knee_xy[1]), (left_ankle_xy[0], left_ankle_xy[1]), color, thickness, lineType=cv2.LINE_8, shift=0)    
    cv2.line(image, (left_ankle_xy[0], left_ankle_xy[1]), (left_heel_xy[0], left_heel_xy[1]), color, thickness, lineType=cv2.LINE_8, shift=0)
    cv2.line(image, (left_heel_xy[0], left_heel_xy[1]), (left_foot_index_xy[0], left_foot_index_xy[1]), color, thickness, lineType=cv2.LINE_8, shift=0)
    cv2.line(image, (right_hip_xy[0], right_hip_xy[1]), (right_knee_xy[0], right_knee_xy[1]), color, thickness, lineType=cv2.LINE_8, shift=0)
    cv2.line(image, (right_knee_xy[0], right_knee_xy[1]), (right_ankle_xy[0], right_ankle_xy[1]), color, thickness, lineType=cv2.LINE_8, shift=0)    
    cv2.line(image, (right_ankle_xy[0], right_ankle_xy[1]), (right_heel_xy[0], right_heel_xy[1]), color, thickness, lineType=cv2.LINE_8, shift=0)
    cv2.line(image, (right_heel_xy[0], right_heel_xy[1]), (right_foot_index_xy[0], right_foot_index_xy[1]), color, thickness, lineType=cv2.LINE_8, shift=0)    

    return image


# 棒人間のクラス
class StickFigure_VideoProcessor:
    def __init__(self) -> None:
        self.radius = int(30.0)
        self.thickness = int(15.0)
        self.R = 0
        self.G = 0
        self.B = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # pose のインスタンス
        mp_pose = mp.solutions.pose

        with mp_pose.Pose(
        min_detection_confidence=0.5,
        static_image_mode=False) as pose_detection:
            # 左右逆転
            img = cv2.flip(img, 1)

            # 画角の高さ
            height = img.shape[0]
            # 画角の幅
            width = img.shape[1]
        
            # pose座標の算出
            results = pose_detection.process(img)

            if not results.pose_landmarks:
                    print('not results')

            else:
                nose_xy, left_shoulder_xy, right_shoulder_xy, left_elbow_xy, right_elbow_xy, left_wrist_xy,\
                right_wrist_xy, left_hip_xy, right_hip_xy, left_knee_xy, right_knee_xy, left_ankle_xy,\
                right_ankle_xy, left_heel_xy, right_heel_xy, left_foot_index_xy, right_foot_index_xy\
                = get_keypoints(results, height, width)

                img = draw_keypoint(img, self.radius, (self.B, self.G, self.R), self.thickness, nose_xy, left_shoulder_xy, right_shoulder_xy,\
                                    left_elbow_xy, right_elbow_xy, left_wrist_xy, right_wrist_xy, left_hip_xy, right_hip_xy,\
                                    left_knee_xy, right_knee_xy, left_ankle_xy, right_ankle_xy, left_heel_xy, right_heel_xy,\
                                    left_foot_index_xy, right_foot_index_xy, height, width)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# mediapipe_poseのクラス
class mp_pose_VideoProcessor:
    def __init__(self) -> None:
        self.radius = int(3.0)
        self.thickness = int(2.0)
        self.detection = 0.5

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # pose のインスタンス
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=self.thickness, color=(0, 255, 0))
        mark_drawing_spec = mp_drawing.DrawingSpec(thickness=self.thickness, circle_radius=self.radius, color=(0, 0, 255))

        with mp_pose.Pose(
        min_detection_confidence=self.detection,
        static_image_mode=False) as pose_detection:
            # 左右逆転
            img = cv2.flip(img, 1)

            # 画角の高さ
            height = img.shape[0]
            # 画角の幅
            width = img.shape[1]
        
            # pose座標の算出
            results = pose_detection.process(img)

            if not results.pose_landmarks:
                    print('not results')

            else:
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mark_drawing_spec,
                    connection_drawing_spec=mesh_drawing_spec
                    )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

class Nomal_VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")