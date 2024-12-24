import os
import time
import cv2
import logging
import numpy as np
from fer import FER
import pyttsx3
import random
from typing import Tuple, Dict, List

########################################
# LOGGING CONFIGURATION
########################################
# Configures the logging settings to output informational messages with timestamps.
# Logs are written only to the console.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
    datefmt="%Y-%m-%d %H:%M:%S"
)

########################################
# INPUT/OUTPUT DIRECTORIES
########################################
# Determines the base directory where the script is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Defines the paths for input images, input videos, output analyzed images, and output analyzed videos.
INPUT_IMAGES_DIR = os.path.join(BASE_DIR, "Input", "Images")
INPUT_VIDEOS_DIR = os.path.join(BASE_DIR, "Input", "Videos")
OUTPUT_IMAGES_DIR = os.path.join(BASE_DIR, "Output", "analyzedImages")
OUTPUT_VIDEOS_DIR = os.path.join(BASE_DIR, "Output", "analyzedVideos")

# Creates the input and output directories if they do not exist to ensure that processed files can be saved.
os.makedirs(INPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(INPUT_VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

########################################
# ENHANCED EMOTION DATA
########################################
# A dictionary mapping each detected emotion to a list of complimentary phrases.
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
# COLOR MAP
########################################
# Defines a color map for each emotion to visually differentiate them in the output.
COLOR_MAP = {
    "happy": (255, 223, 0),       # Yellow
    "neutral": (200, 200, 200),   # Gray
    "sad": (147, 112, 219),       # Medium Purple
    "angry": (220, 20, 60),       # Crimson
    "surprise": (255, 165, 0),    # Orange
    "fear": (138, 43, 226),       # Blue Violet
    "disgust": (50, 205, 50)      # Lime Green
}

########################################
# EMOTION DETECTOR CLASS
########################################
class EmotionDetector:
    """
    A class to detect emotions in images and videos using the FER library and provide verbal compliments.
    """
    def __init__(self):
        # Initializes the FER emotion detector with MTCNN for face detection.
        self.detector = FER(mtcnn=True)
        
        # Initializes the text-to-speech engine for verbal compliments.
        self.engine = pyttsx3.init()
        self.setup_voice()
        
        # Tracks the last time a compliment was given per emotion to prevent spamming.
        self.last_compliment_time = {emotion: 0 for emotion in COMPLIMENTS.keys()}
        self.compliment_cooldown = 5  # Cooldown period in seconds.

    def setup_voice(self):
        """
        Configures the text-to-speech engine's properties such as speech rate and volume.
        """
        self.engine.setProperty("rate", 150)    # Sets the speech rate.
        self.engine.setProperty("volume", 0.8)  # Sets the volume level.

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Analyzes a single frame for emotions, annotates detected faces with bounding boxes and labels,
        and overlays compliments.

        Args:
            frame (np.ndarray): The image frame to process.

        Returns:
            Tuple[np.ndarray, Dict]: The annotated frame and a dictionary containing emotion data.
        """
        # Detects emotions in the frame.
        faces = self.detector.detect_emotions(frame)
        emotion_data = {"faces": faces, "dominant_emotion": None}

        if faces:
            for face in faces:
                box = face["box"]  # Bounding box of the detected face.
                emotions = face["emotions"]  # Dictionary of emotion probabilities.
                dominant_emotion = max(emotions, key=emotions.get)  # Identifies the dominant emotion.
                confidence = emotions[dominant_emotion] * 100  # Converts to percentage.
                color = COLOR_MAP.get(dominant_emotion, (255, 255, 255))  # Selects color based on emotion.
                label = f"{dominant_emotion.title()} ({confidence:.1f}%)"  # Creates label text.
                
                # Draws a rectangle around the face.
                cv2.rectangle(
                    frame,
                    (box[0], box[1]),
                    (box[0] + box[2], box[1] + box[3]),
                    color,
                    2
                )
                
                # Puts the emotion label above the bounding box.
                cv2.putText(
                    frame,
                    label,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
                
                # Updates the dominant emotion in the emotion_data dictionary.
                emotion_data["dominant_emotion"] = dominant_emotion

                # Provide a verbal compliment if cooldown has passed for this emotion.
                current_time = time.time()
                if current_time - self.last_compliment_time[dominant_emotion] > self.compliment_cooldown:
                    compliment = self.give_compliment(dominant_emotion)
                    self.last_compliment_time[dominant_emotion] = current_time
                    # Log the emotion and compliment
                    logging.info(f"Detected Emotion: {dominant_emotion.title()} ({confidence:.1f}%) - Compliment: \"{compliment}\"")
                    
                    # Calculate dynamic font scale based on frame size
                    font_scale = self.calculate_font_scale(frame.shape)
                    # Determine position relative to the face
                    compliment_position = (box[0], box[1] + box[3] + 30)  # Below the face
                    # Ensure the position is within frame boundaries
                    compliment_position = self.adjust_position(compliment_position, frame.shape, compliment, font_scale)
                    
                    # Overlay the compliment text on the frame
                    self.overlay_text(
                        frame,
                        compliment,
                        position=compliment_position,
                        font_scale=font_scale,
                        color=(0, 255, 0),
                        thickness=2
                    )

        return frame, emotion_data

    def give_compliment(self, emotion: str) -> str:
        """
        Selects a random compliment based on the detected emotion and uses text-to-speech to deliver it.

        Args:
            emotion (str): The detected dominant emotion.

        Returns:
            str: The compliment that was delivered.
        """
        compliments = COMPLIMENTS.get(emotion, ["You're amazing!"])
        compliment = random.choice(compliments)
        self.engine.say(compliment)
        self.engine.runAndWait()
        return compliment

    def overlay_text(self, frame: np.ndarray, text: str, position: Tuple[int, int], font_scale: float, color: Tuple[int, int, int], thickness: int):
        """
        Overlays text onto the frame at the specified position.

        Args:
            frame (np.ndarray): The image/frame to annotate.
            text (str): The text to overlay.
            position (Tuple[int, int]): The (x, y) position for the text.
            font_scale (float): The scale of the font.
            color (Tuple[int, int, int]): The color of the text in BGR.
            thickness (int): The thickness of the text.
        """
        # Choose a font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Calculate text size to adjust position if needed
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

        # Adjust position if text goes beyond frame boundaries
        x, y = position
        if x + text_width > frame.shape[1]:
            x = frame.shape[1] - text_width - 10  # 10 pixels from the edge
        if y + text_height > frame.shape[0]:
            y = frame.shape[0] - text_height - 10  # 10 pixels from the bottom

        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

    def calculate_font_scale(self, frame_shape: Tuple[int, int, int]) -> float:
        """
        Calculates an appropriate font scale based on the frame size.

        Args:
            frame_shape (Tuple[int, int, int]): The shape of the frame (height, width, channels).

        Returns:
            float: The calculated font scale.
        """
        height, width, _ = frame_shape
        # Base font scale for a 640x480 frame
        base_scale = 1.0
        base_width = 640
        scale = (width / base_width) * base_scale
        # Clamp the scale to prevent it from being too small or too large
        scale = max(0.5, min(scale, 2.0))
        return scale

    def adjust_position(self, position: Tuple[int, int], frame_shape: Tuple[int, int, int], text: str, font_scale: float) -> Tuple[int, int]:
        """
        Adjusts the position of the compliment text to ensure it stays within frame boundaries.

        Args:
            position (Tuple[int, int]): The initial (x, y) position for the text.
            frame_shape (Tuple[int, int, int]): The shape of the frame (height, width, channels).
            text (str): The text to overlay.
            font_scale (float): The scale of the font.

        Returns:
            Tuple[int, int]: The adjusted (x, y) position for the text.
        """
        x, y = position
        height, width, _ = frame_shape
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

        # If the text goes beyond the right edge, move it to the left
        if x + text_width > width:
            x = width - text_width - 10  # 10 pixels padding

        # If the text goes below the bottom edge, move it above the face
        if y + text_height > height:
            y = height - text_height - 10  # 10 pixels from the bottom

        return (x, y)

########################################
# PROCESS IMAGES FUNCTION
########################################
def process_images(detector: EmotionDetector):
    """
    Processes all images in the INPUT_IMAGES_DIR by detecting emotions and saving annotated images.

    Args:
        detector (EmotionDetector): An instance of the EmotionDetector class.
    """
    # Retrieves and sorts image files for consistent ordering.
    image_files = sorted([
        file for file in os.listdir(INPUT_IMAGES_DIR)
        if file.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    for idx, image_name in enumerate(image_files, start=1):
        image_path = os.path.join(INPUT_IMAGES_DIR, image_name)
        
        # Reads the image using OpenCV.
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Failed to load image: {image_name}")
            continue

        # Processes the image frame to detect and annotate emotions.
        processed_frame, emotion_data = detector.process_frame(image)
        
        # Defines the output path for the annotated image.
        output_path = os.path.join(OUTPUT_IMAGES_DIR, f"analyzed_{image_name}")
        
        # Writes the annotated image to the output directory.
        cv2.imwrite(output_path, processed_frame)
        logging.info(f"Image {idx}: {emotion_data.get('dominant_emotion', 'No Emotion Detected')} - Analyzed image saved: {output_path}")

########################################
# PROCESS VIDEOS FUNCTION
########################################
def process_videos(detector: EmotionDetector):
    """
    Processes all videos in the INPUT_VIDEOS_DIR by detecting emotions frame-by-frame and saving annotated videos.

    Args:
        detector (EmotionDetector): An instance of the EmotionDetector class.
    """
    # Retrieves and sorts video files for consistent ordering.
    video_files = sorted([
        file for file in os.listdir(INPUT_VIDEOS_DIR)
        if file.lower().endswith(('.mp4', '.avi', '.mov'))
    ])
    
    for video_name in video_files:
        video_path = os.path.join(INPUT_VIDEOS_DIR, video_name)
        
        # Attempts to open the video file.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning(f"Failed to open video: {video_name}")
            continue

        # Defines the output path for the annotated video.
        output_path = os.path.join(OUTPUT_VIDEOS_DIR, f"analyzed_{video_name}")
        
        # Retrieves video properties to maintain consistency in the output video.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Defines the codec.
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initializes the VideoWriter object to write the annotated video.
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        logging.info(f"Processing video: {video_name}")
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if no frames are returned.

            # Processes the frame to detect and annotate emotions.
            processed_frame, emotion_data = detector.process_frame(frame)
            
            # Writes the annotated frame to the output video.
            out.write(processed_frame)
            frame_count += 1
            
            # Optional: Log every 100 frames to avoid excessive logging
            if frame_count % 100 == 0:
                dominant_emotion = emotion_data.get("dominant_emotion", "No Emotion")
                logging.info(f"Video {video_name}: Processed {frame_count} frames - Current Emotion: {dominant_emotion}")
        
        # Releases the VideoCapture and VideoWriter objects.
        cap.release()
        out.release()
        logging.info(f"Processed {frame_count} frames for video: {video_name}")
        logging.info(f"Analyzed video saved: {output_path}")

########################################
# MAIN FUNCTION
########################################
def main():
    """
    The main function initializes the EmotionDetector and processes all images and videos for emotion analysis.
    """
    logging.info("Starting MoodLyft-Mirror: Input Emotion Analyzer")
    
    # Initializes the EmotionDetector instance.
    detector = EmotionDetector()
    
    # Processes all images in the input directory.
    process_images(detector)
    
    # Processes all videos in the input directory.
    process_videos(detector)
    
    logging.info("Processing complete.")

########################################
# ENTRY POINT
########################################
if __name__ == "__main__":
    main()