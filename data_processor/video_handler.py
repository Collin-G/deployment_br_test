import tempfile
import torch

import cv2
import dlib
import numpy as np
from scipy.spatial import procrustes
import imageio

# Initialize Dlib's face detector and landmark predictor

class VideoHandler():
    def __init__(self, video_path, detector, predictor):
        self.detector = detector
        self.predictor = predictor 
        self.video_path = video_path 
        # Save the video temporarily
     

    def get_video_features(self):
        
        video_features = self.process_video(self.video_path)
        video_features = video_features.reshape(video_features.shape[0], video_features.shape[1], -1)
        return video_features

    

    def get_total_frames_imageio(self, video_file):
        reader = imageio.get_reader(video_file)
        total_frames = reader.count_frames()
        reader.close()  # Clean up resources
        return total_frames

    # Helper function to extract facial landmarks
    def get_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:
            return None  # No face detected

        landmarks = self.predictor(gray, faces[0])  # Assuming 1 face per frame
        points = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32)
        return points

    # Helper function to normalize landmarks using affine transform and scaling
    def affine_transform(self, landmarks):
        src_points = np.float32([landmarks[36], landmarks[45], landmarks[30]])  # Eye corners + nose
        target_points = np.float32([[0, 0], [1, 0], [0.5, 1]])  # Canonical alignment
        M = cv2.getAffineTransform(src_points, target_points)
        transformed = cv2.transform(np.expand_dims(landmarks, axis=0), M)
        return transformed[0]

    def scale_landmarks(self, landmarks):
        interocular_dist = np.linalg.norm(landmarks[36] - landmarks[45])
        return landmarks / interocular_dist

    def center_landmarks(self, landmarks):
        return landmarks - landmarks[30]  # Centering on the nose

    def normalize_landmarks(self, landmarks, reference_landmarks=None):
        aligned = self.affine_transform(landmarks)
        scaled = self.scale_landmarks(aligned)
        centered = self.center_landmarks(scaled)
        if reference_landmarks is not None:
            _, centered, _ = procrustes(reference_landmarks, centered)  # Optional Procrustes alignment
        return centered

    # Process video and create batches of 5 (68, 2) landmarks
    def process_video(self, video_path, batch_size=5):
        cap = cv2.VideoCapture(video_path)
        total_frames = self.get_total_frames_imageio(video_path)
        print("Total frames:", total_frames)

        all_landmarks = []  # Store landmarks across all batches

        # Temporary list to hold landmarks for the current batch
        batch_landmarks = []
        frame_count = 0
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        while True:
            frame_count += 1
            if frame_count%fps == 0:
                continue


            ret, frame = cap.read()
            if not ret:
                break  # End of video

            landmarks = self.get_landmarks(frame)

            if landmarks is not None and landmarks.shape == (68, 2):  # Check shape
                normalized = self.normalize_landmarks(landmarks)
                # Ensure normalized landmarks also have the correct shape
                if normalized.shape == (68, 2):
                    batch_landmarks.append(normalized)

            # If we have filled a batch of 5 valid landmarks
            if len(batch_landmarks) == batch_size:
                all_landmarks.append(np.array(batch_landmarks))  # Add batch to all_landmarks
                batch_landmarks = []  # Reset the batch for the next set of landmarks

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        return np.array(all_landmarks) if len(all_landmarks) > 0 else None  # Return None if no batches

    # Example usage: