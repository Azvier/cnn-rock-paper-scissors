import os
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configure environment and logging
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'TF_ENABLE_ONEDNN_OPTS': '0',
    'TF_CPP_MIN_VLOG_LEVEL': '3'
})
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
import tkinter as tk
from tkinter import filedialog
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model


class RockPaperScissorsPredictor:
    """Rock Paper Scissors classifier using pre-trained CNN models."""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        os.chdir(self.script_dir)
        
        self.class_names = ['paper', 'rock', 'scissors']
        self.models = self._load_models()
    
    def _load_models(self) -> Dict[str, Model]:
        """Load pre-trained models and validate their existence."""
        model_paths = {
            'VGG16': self.script_dir / '../models/best_models/best_VGG16_fine_tuned_model.keras',
            'ResNet50': self.script_dir / '../models/best_models/best_ResNet50_fine_tuned_model.keras'
        }
        
        models = {}
        for name, path in model_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{name} model not found at {path}")
            models[name] = load_model(path)
            
        return models
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Preprocess image for model prediction."""
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0
    
    def predict_image(self, image_path: str) -> None:
        """Predict image class using all loaded models."""
        img_array = self.preprocess_image(image_path)
        
        for model_name, model in self.models.items():
            predictions = model.predict(img_array, verbose=0)
            predicted_class = self.class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100
            print(f"{model_name}: {predicted_class} ({confidence:.2f}%)")


class CameraHandler:
    """Handle camera operations for image capture."""
    
    @staticmethod
    def list_available_cameras() -> List[str]:
        """List all available camera devices."""
        try:
            from pygrabber.dshow_graph import FilterGraph
            graph = FilterGraph()
            return graph.get_input_devices()
        except ImportError:
            raise ImportError("pygrabber not installed. Install with: pip install pygrabber")
    
    @staticmethod
    def select_camera() -> int:
        """Prompt user to select camera from available devices."""
        devices = CameraHandler.list_available_cameras()
        if not devices:
            raise RuntimeError("No cameras detected")
        
        print("\nAvailable cameras:")
        for idx, name in enumerate(devices):
            print(f"Camera {idx}: {name}")
        
        while True:
            try:
                cam_index = int(input("Enter camera index [0]: ") or 0)
                if 0 <= cam_index < len(devices):
                    return cam_index
                print("Invalid camera index. Please choose from the list above.")
            except ValueError:
                print("Please enter a valid integer.")
    
    @staticmethod
    def capture_image_from_camera(cam_index: int) -> str:
        """Capture image using selected camera."""
        try:
            from PIL import Image, ImageTk
        except ImportError:
            raise ImportError("PIL not installed. Install with: pip install Pillow")
        
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError("Could not access the webcam")
        
        captured_image_path = "captured_image.png"
        
        def show_camera():
            ret, frame = cap.read()
            if ret:
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                camera_label.imgtk = imgtk
                camera_label.configure(image=imgtk)
            camera_label.after(10, show_camera)
        
        def capture_image():
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(captured_image_path, frame)
                print(f"Image captured: {captured_image_path}")
            cap.release()
            root.destroy()
            cv2.destroyAllWindows()
        
        root = tk.Tk()
        root.title("Camera Capture")
        camera_label = tk.Label(root)
        camera_label.pack()
        
        capture_btn = tk.Button(root, text="Capture", command=capture_image)
        capture_btn.pack()
        
        show_camera()
        root.mainloop()
        
        return captured_image_path


class UserInterface:
    """Handle user interactions and input selection."""
    
    @staticmethod
    def get_user_choice() -> str:
        """Prompt user to choose between upload or camera capture."""
        print("\nChoose an option:")
        print("1. Upload an image")
        print("2. Take a picture (requires webcam)")
        return input("Enter 1 or 2: ").strip()
    
    @staticmethod
    def select_image_file() -> Optional[str]:
        """Open file dialog to select image."""
        root = tk.Tk()
        root.withdraw()
        
        image_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )
        return image_path if image_path else None


def main():
    """Main application entry point."""
    try:
        predictor = RockPaperScissorsPredictor()
        ui = UserInterface()
        
        choice = ui.get_user_choice()
        
        if choice == "1":
            image_path = ui.select_image_file()
            if image_path:
                print(f"Selected: {image_path}")
                predictor.predict_image(image_path)
            else:
                print("No image selected.")
                
        elif choice == "2":
            try:
                camera_handler = CameraHandler()
                cam_index = camera_handler.select_camera()
                image_path = camera_handler.capture_image_from_camera(cam_index)
                predictor.predict_image(image_path)
            except (ImportError, RuntimeError) as e:
                print(f"Camera error: {e}")
                sys.exit(1)
        else:
            print("Invalid choice.")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()