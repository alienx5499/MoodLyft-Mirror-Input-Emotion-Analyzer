import os
import time
import cv2
import logging
import numpy as np
from fer import FER
import pyttsx3
import random
import math
import datetime
from typing import Tuple, Dict, List
import platform
from PIL import Image, ImageDraw, ImageFont

########################################
# LOGGING CONFIGURATION
########################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

########################################
# INPUT/OUTPUT DIRECTORIES
########################################
INPUT_IMAGES_DIR = "Input/Images"
INPUT_VIDEOS_DIR = "Input/Videos"
OUTPUT_IMAGES_DIR = "Output/analyzedImages"
OUTPUT_VIDEOS_DIR = "Output/analyzedVideos"

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

########################################
# ENHANCED EMOTION DATA
########################################
COMPLIMENTS = {
    "happy": [
        "Your smile brightens everyone's day!",
        "That genuine happiness is contagious!",
        "You have such a warm, inviting presence!",
        "Your joy creates ripples of positivity!",
        "Keep shining with that beautiful energy!"
    ],
    "neutral": [
        "Your calm presence is so grounding!",
        "There's strength in your composure!",
        "Your steady energy is admirable!",
        "You bring balance wherever you go!",
        "Your mindful presence is appreciated!"
    ],
    "sad": [
        "You're stronger than you know!",
        "Every storm passes - hang in there!",
        "Your resilience is truly inspiring!",
        "Better days are coming - believe it!",
        "You're never alone in this journey!"
    ],
    "angry": [
        "Channel that energy into positive change!",
        "Your passion can move mountains!",
        "Transform that fire into motivation!",
        "Your intensity can spark amazing things!",
        "Use that power to achieve greatness!"
    ],
    "surprise": [
        "Your wonderment is refreshing!",
        "Stay curious and keep exploring!",
        "Life is full of amazing discoveries!",
        "Your enthusiasm is infectious!",
        "Keep embracing new experiences!"
    ],
    "fear": [
        "Courage isn't fearlessness - it's facing fear!",
        "You're braver than you believe!",
        "Every step forward conquers fear!",
        "Your strength shines through uncertainty!",
        "Fear is temporary - your courage is permanent!"
    ],
    "disgust": [
        "Your standards show self-respect!",
        "Trust your instincts - they serve you well!",
        "Your boundaries protect your peace!",
        "Standing firm shows inner strength!",
        "Your authenticity is powerful!"
    ]
}

########################################
# COLOR MAP AND EMOJIS
########################################
COLOR_MAP = {
    "happy": (255, 223, 0),
    "neutral": (200, 200, 200),
    "sad": (147, 112, 219),
    "angry": (220, 20, 60),
    "surprise": (255, 165, 0),
    "fear": (138, 43, 226),
    "disgust": (50, 205, 50)
}

EMOJIS = {
    "happy": "😊",
    "neutral": "😌",
    "sad": "🥺",
    "angry": "😤",
    "surprise": "😲",
    "fear": "😨",
    "disgust": "😖"
}

########################################
# EMOTION DETECTOR
########################################
class EmotionDetector:
    def __init__(self):
        self.detector = FER(mtcnn=True)
        self.engine = pyttsx3.init()
        self.setup_voice()
        self.last_compliment_time = 0
        self.compliment_cooldown = 5

    def setup_voice(self):
        """Configure text-to-speech settings."""
        self.engine.setProperty("rate", 150)
        self.engine.setProperty("volume", 0.8)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Analyze a frame for emotions and annotate it."""
        faces = self.detector.detect_emotions(frame)
        emotion_data = {"faces": faces, "dominant_emotion": None}

        if faces:
            for face in faces:
                box = face["box"]
                emotions = face["emotions"]
                dominant_emotion = max(emotions, key=emotions.get)
                color = COLOR_MAP.get(dominant_emotion, (255, 255, 255))
                label = f"{dominant_emotion.title()} ({emotions[dominant_emotion]:.1%})"
                cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                emotion_data["dominant_emotion"] = dominant_emotion

        return frame, emotion_data

########################################
# PROCESS IMAGES
########################################
def process_images(detector: EmotionDetector):
    for image_name in os.listdir(INPUT_IMAGES_DIR):
        image_path = os.path.join(INPUT_IMAGES_DIR, image_name)
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Failed to load image: {image_name}")
            continue

        processed_frame, _ = detector.process_frame(image)
        output_path = os.path.join(OUTPUT_IMAGES_DIR, f"analyzed_{image_name}")
        cv2.imwrite(output_path, processed_frame)
        logging.info(f"Analyzed image saved: {output_path}")

########################################
# PROCESS VIDEOS
########################################
def process_videos(detector: EmotionDetector):
    for video_name in os.listdir(INPUT_VIDEOS_DIR):
        video_path = os.path.join(INPUT_VIDEOS_DIR, video_name)
        if not video_name.lower().endswith(('.mp4', '.avi', '.mov')):
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning(f"Failed to open video: {video_name}")
            continue

        output_path = os.path.join(OUTPUT_VIDEOS_DIR, f"analyzed_{video_name}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, _ = detector.process_frame(frame)
            out.write(processed_frame)

        cap.release()
        out.release()
        logging.info(f"Analyzed video saved: {output_path}")

########################################
# MAIN FUNCTION
########################################
def main():
    logging.info("Starting MoodLyft-Mirror: Input Emotion Analyzer")
    detector = EmotionDetector()
    process_images(detector)
    process_videos(detector)
    logging.info("Processing complete.")

if __name__ == "__main__":
    main()