import csv
import os
import cv2
import numpy as np

def detect_circle_diameter(image):
    # Convert to HSV color space for better green detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for green color
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No circle detected")
    
    # Get the largest contour (should be our circle)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit a minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)
    diameter = radius * 2
    
    # Calculate circle metrics
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Display results
    print(f"Circle Measurements:")
    print(f"Diameter: {diameter} pixels")
    print(f"Radius: {radius} pixels")
    print(f"Center: ({center[0]}, {center[1]})")
    print(f"Circularity: {circularity:.3f}")
    
    return {
        'diameter': diameter,
        'radius': radius,
        'center': center,
        'circularity': circularity
    }

def crop_circle(image_path, save_path=None):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")
    
    # Convert to HSV color space for better green detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for green color
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No circle detected")
    
    # Get the largest contour (should be our circle)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit a minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)
    
    # Create a circular mask
    circle_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 255, -1)
    
    # Create masked image (black background)
    masked_image = cv2.bitwise_and(image, image, mask=circle_mask)
    
    # Get the bounding box of the circle
    x_min = max(0, int(x - radius))
    x_max = min(image.shape[1], int(x + radius))
    y_min = max(0, int(y - radius))
    y_max = min(image.shape[0], int(y + radius))
    
    # Crop the image to the bounding box
    cropped_image = masked_image[y_min:y_max, x_min:x_max]
    
    # Save the cropped image if a save path is provided
    if save_path:
        cv2.imwrite(save_path, cropped_image)
        print(f"Saved cropped image to: {save_path}")
    
    return cropped_image

from dataclasses import dataclass
from typing import List

@dataclass
class CropResult:
    image: np.ndarray
    adjusted_coordinates: List[float]
    offset_x: int
    offset_y: int
    scale: float

def crop_circle_with_coordinates(image_path: str, coordinates: List[float], save_path: str = None) -> CropResult:
    """
    Crop the circular region of an image and adjust the corner coordinates accordingly.
    
    Args:
        image_path: Path to the input image
        coordinates: List of normalized coordinates [x1,y1,x2,y2,x3,y3,x4,y4]
        save_path: Optional path to save the cropped image
        
    Returns:
        CropResult object containing the cropped image and adjusted coordinates
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")
    
    original_height, original_width = image.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    pixel_coords = []
    for i in range(0, len(coordinates), 2):
        x = int(coordinates[i] * original_width)
        y = int(coordinates[i + 1] * original_height)
        pixel_coords.extend([x, y])
    
    # Convert to HSV color space for better green detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for green color
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No circle detected")
    
    # Get the largest contour (should be our circle)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit a minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)
    
    # Create a circular mask
    circle_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 255, -1)
    
    # Create masked image (black background)
    masked_image = cv2.bitwise_and(image, image, mask=circle_mask)
    
    # Get the bounding box of the circle
    x_min = max(0, int(x - radius))
    x_max = min(original_width, int(x + radius))
    y_min = max(0, int(y - radius))
    y_max = min(original_height, int(y + radius))
    
    # Crop the image to the bounding box
    cropped_image = masked_image[y_min:y_max, x_min:x_max]
    
    # Calculate new dimensions
    new_width = x_max - x_min
    new_height = y_max - y_min
    
    # Adjust coordinates
    adjusted_coords = []
    for i in range(0, len(pixel_coords), 2):
        # Adjust for cropping offset
        adj_x = (pixel_coords[i] - x_min) / new_width  # Normalize to 0-1
        adj_y = (pixel_coords[i + 1] - y_min) / new_height  # Normalize to 0-1
        
        # Ensure coordinates are within bounds
        adj_x = max(0, min(1, adj_x))
        adj_y = max(0, min(1, adj_y))
        
        adjusted_coords.extend([adj_x, adj_y])
    
    # Save the cropped image if a save path is provided
    if save_path:
        cv2.imwrite(save_path, cropped_image)
        print(f"Saved cropped image to: {save_path}")
    
    return CropResult(
        image=cropped_image,
        adjusted_coordinates=adjusted_coords,
        offset_x=x_min,
        offset_y=y_min,
        scale=min(new_width/original_width, new_height/original_height)
    )

import cv2
import numpy as np
from typing import Tuple

def crop_circle_with_predictions(image: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Crop the circular region of an image and calculate scale-preserving offsets.
    The offsets are calculated to maintain the relative positions of predicted corners
    within the cropped region.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Tuple containing:
        - cropped_image: The cropped circular region
        - x_offset: Scale-preserving x offset for corner predictions
        - y_offset: Scale-preserving y offset for corner predictions
    """
    if image is None:
        raise ValueError("Could not load image")
    
    original_height, original_width = image.shape[:2]
    
    # Convert to HSV color space for better green detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for green color
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No circle detected")
    
    # Get the largest contour (should be our circle)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit a minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)
    
    # Create a circular mask
    circle_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 255, -1)
    
    # Create masked image (black background)
    masked_image = cv2.bitwise_and(image, image, mask=circle_mask)
    
    # Get the bounding box of the circle
    x_min = max(0, int(x - radius))
    x_max = min(original_width, int(x + radius))
    y_min = max(0, int(y - radius))
    y_max = min(original_height, int(y + radius))
    
    # Crop the image to the bounding box
    cropped_image = masked_image[y_min:y_max, x_min:x_max]
    
    return cropped_image, x_min, y_min, cropped_image.shape[0], cropped_image.shape[1]