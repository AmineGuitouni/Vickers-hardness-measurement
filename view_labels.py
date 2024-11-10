import cv2
import csv
import os
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from utils.image_utils import crop_circle, crop_circle_with_coordinates

@dataclass
class Point:
    x: int
    y: int
    label: str

    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)

class ImageAnnotator:
    LINE_COLOR = (0, 0, 255)  # BGR format (Red)
    LINE_THICKNESS = 2
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_COLOR = (0, 0, 255)  # BGR format (Red to match lines)
    FONT_THICKNESS = 1
    LABEL_OFFSET = 10  # Pixels to offset label from corner

    def __init__(self, csv_file: str, image_folder: str, screen_width: int = 1280, screen_height: int = 720):
        self.csv_file = csv_file
        self.image_folder = image_folder
        self.screen_width = screen_width
        self.screen_height = screen_height

    @staticmethod
    def denormalize_point(x: float, y: float, img_width: int, img_height: int) -> Tuple[int, int]:
        """Convert normalized coordinates (0-1) to pixel values."""
        return int(x * img_width), int(y * img_height)

    @staticmethod
    def resize_image(image, max_width: int, max_height: int) -> Tuple[cv2.Mat, float]:
        """Resize image while maintaining aspect ratio."""
        height, width = image.shape[:2]
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)

        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_image = cv2.resize(image, (new_width, new_height))
            return resized_image, scale
        return image, 1.0

    def draw_box_with_labels(self, img: cv2.Mat, points: List[Point]) -> None:
        """Draw bounding box with labeled corners."""
        for i in range(4):
            cv2.line(img, 
                     points[i].to_tuple(), 
                     points[(i + 1) % 4].to_tuple(), 
                     self.LINE_COLOR, 
                     self.LINE_THICKNESS)
            
            # Add corner labels
            x, y = points[i].to_tuple()
            label = points[i].label
            
            # Calculate label position offset based on corner position
            if "Top" in label:
                y_offset = -self.LABEL_OFFSET
            else:
                y_offset = self.LABEL_OFFSET * 2

            if "Left" in label:
                x_offset = -self.LABEL_OFFSET
            else:
                x_offset = self.LABEL_OFFSET

            # Draw label text
            cv2.putText(img,
                       label,
                       (x + x_offset, y + y_offset),
                       self.FONT,
                       self.FONT_SCALE,
                       self.FONT_COLOR,
                       self.FONT_THICKNESS)

    def process_images(self) -> None:
        """Process all images in the CSV file."""
        # Load CSV rows into memory
        with open(self.csv_file, newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)
        
        # Skip the header row
        header = rows[0]
        data_rows = rows[1:]
        updated_rows = [header]  # Start with header row for re-writing

        for row in data_rows:
            image_name = row[0]
            normalized_coords = np.array(list(map(float, row[1:])))  # Read normalized coordinates

            # Load and process image
            img_path = os.path.join(self.image_folder, image_name)
            cropped_data = crop_circle_with_coordinates(img_path, normalized_coords)
            img = cropped_data.image

            normalized_coords = cropped_data.adjusted_coordinates

            if img is None:
                print(f"Error loading image: {image_name}")
                continue

            # Resize image
            img_resized, scale = self.resize_image(img, self.screen_width, self.screen_height)
            img_height, img_width = img_resized.shape[:2]

            # Create Points with labels
            points = [
                Point(*self.denormalize_point(normalized_coords[0], normalized_coords[1], img_width, img_height), "Top Left"),
                Point(*self.denormalize_point(normalized_coords[2], normalized_coords[3], img_width, img_height), "Top Right"),
                Point(*self.denormalize_point(normalized_coords[4], normalized_coords[5], img_width, img_height), "Bottom Right"),
                Point(*self.denormalize_point(normalized_coords[6], normalized_coords[7], img_width, img_height), "Bottom Left")
            ]

            # Draw box and labels
            self.draw_box_with_labels(img_resized, points)

            # Display image
            cv2.imshow("Image with Labeled Box", img_resized)
            key = cv2.waitKey(0)

            if key == ord('q'):  # Press 'q' to quit
                break
            elif key != ord('d'):  # Press 'd' to delete image from CSV
                updated_rows.append(row)  # Keep row if not deleting

        cv2.destroyAllWindows()

        # Rewrite CSV with the updated rows
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(updated_rows)

def main():
    # Configuration
    csv_file = 'bounding_boxes_points_all.csv'
    image_folder = './data/images/all'
    
    # Create and run annotator
    annotator = ImageAnnotator(csv_file, image_folder)
    annotator.process_images()

if __name__ == '__main__':
    main()