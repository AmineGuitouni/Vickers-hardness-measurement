import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import torch
import numpy as np
from typing import Dict, Union
from utils.imagePoint import has_close_points, image_corners
from utils.image_utils import crop_circle_with_predictions, detect_circle_diameter

class ImageDetectionService:
    def __init__(self, model_path: str = './checkpoints/test-50.pth', image_size: int = 256):
        self.image_size = image_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.model = torch.load(model_path)
            self.model.eval()
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model from {model_path}")

    def preprocess_image(self, image_data: bytes) -> tuple[np.ndarray, Image.Image]:
        """Preprocess the image data."""
        try:
            image = Image.open(BytesIO(image_data))
            image_cropped, x_offset, y_offset, crop_width, crop_height = crop_circle_with_predictions(np.array(image))
            image_resized = cv2.resize(image_cropped, (self.image_size, self.image_size))
            image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            return image_gray, image, x_offset, y_offset, crop_width, crop_height
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")

    def get_default_response(self) -> Dict[str, Union[float, int]]:
        """Return default response when error occurs."""
        return {
            "x1": 0.25, "y1": 0.25,
            "x2": 0.75, "y2": 0.25,
            "x3": 0.75, "y3": 0.75,
            "x4": 0.25, "y4": 0.75,
            "pixelDistance": 1.0
        }

    async def process_image(self, image_data: bytes) -> Dict[str, Union[float, int]]:
        """Process image and return corner coordinates and pixel distance."""
        try:
            # Preprocess image
            image_gray, original_image, x_offset, y_offset, crop_width, crop_height = self.preprocess_image(image_data)
            
            original_w, original_h = original_image.size
            # Prepare tensor
            image_tensor = torch.from_numpy(image_gray / 255).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                mask_pred = self.model(image_tensor)

            # Get corners and circle detection
            corners = image_corners(mask_pred=mask_pred, threshold=0.95)
            circle = detect_circle_diameter(np.array(original_image))
            print(corners)
            # Calculate pixel distance
            pixel_distance = (circle['diameter'] / 14) / 50 if circle else 1.0
            

            offseted_corners = []
            for corner in corners:
                original_cropped_x = int(corner[0] * (crop_width / self.image_size))
                original_cropped_y = int(corner[1] * (crop_height / self.image_size))
                
                offseted_corners.append([original_cropped_x + x_offset, original_cropped_y + y_offset])

            print(offseted_corners)
            # Return default response if corners not found
            if len(offseted_corners) != 4 :
                print("Warning: Could not detect 4 corners")
                return self.get_default_response()
            
            # Return normalized coordinates and pixel distance
            return {
                "x1": offseted_corners[0][0] / original_w,
                "y1": offseted_corners[0][1] / original_h,
                "x2": offseted_corners[1][0] / original_w,
                "y2": offseted_corners[1][1] / original_h,
                "x3": offseted_corners[2][0] / original_w,
                "y3": offseted_corners[2][1] / original_h,
                "x4": offseted_corners[3][0] / original_w,
                "y4": offseted_corners[3][1] / original_h,
                "pixelDistance": pixel_distance
            }
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return self.get_default_response()

# Initialize FastAPI app and service
app = FastAPI()
try:
    detection_service = ImageDetectionService()
except Exception as e:
    print(f"Failed to initialize ImageDetectionService: {str(e)}")
    raise

@app.post("/image-detection/")
async def get_image_size(image: UploadFile = File(...)) -> Dict[str, Union[float, int]]:
    """
    Process uploaded image and return corner coordinates and pixel distance.
    Returns default values (0 for coordinates, 1 for pixel distance) on error.
    """
    try:
        image_data = await image.read()
        return await detection_service.process_image(image_data)
    except Exception as e:
        print(f"Error in get_image_size endpoint: {str(e)}")
        return detection_service.get_default_response()

@app.get("/")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}