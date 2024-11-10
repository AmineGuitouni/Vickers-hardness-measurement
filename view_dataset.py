import cv2
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from data.dataset import ImagesDataSetBb, ImagesDataSetUNet

@dataclass
class Point:
    x: int
    y: int
    label: str

    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)

class DatasetViewer:
    LINE_COLOR = (0, 0, 255)  # BGR format (Red)
    LINE_THICKNESS = 2
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_COLOR = (0, 0, 255)  # BGR format (Red)
    FONT_THICKNESS = 1
    LABEL_OFFSET = 10

    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    def draw_bounding_box(self, img: np.ndarray, points: List[Point]) -> None:
        """Draw bounding box with labeled corners."""
        # Draw lines between consecutive points
        for i in range(4):
            cv2.line(img, 
                     points[i].to_tuple(), 
                     points[(i + 1) % 4].to_tuple(), 
                     self.LINE_COLOR, 
                     self.LINE_THICKNESS)
            
            # Add corner labels
            x, y = points[i].to_tuple()
            label = points[i].label
            
            if "Top" in label:
                y_offset = -self.LABEL_OFFSET
            else:
                y_offset = self.LABEL_OFFSET * 2

            if "Left" in label:
                x_offset = -self.LABEL_OFFSET
            else:
                x_offset = self.LABEL_OFFSET

            cv2.putText(img,
                       label,
                       (x + x_offset, y + y_offset),
                       self.FONT,
                       self.FONT_SCALE,
                       self.FONT_COLOR,
                       self.FONT_THICKNESS)

    def draw_mask(self, img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay mask on image with transparency."""
        mask_colored = np.zeros_like(img)
        mask_colored[mask > 0] = [0, 255, 0]  # Green mask
        return cv2.addWeighted(img, 1, mask_colored, alpha, 0)

    def view_unet_dataset(self):
        """View images and masks from UNet dataset."""
        for batch in self.dataloader:
            image = batch['image']
            mask = batch['mask']
            
            for i in range(len(image)):
                # Convert tensor to numpy array and prepare for display
                img_display = image[i].numpy()
                if len(img_display.shape) == 2:  # Grayscale
                    img_display = cv2.cvtColor((img_display * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                else:  # RGB
                    img_display = (img_display.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)

                mask_display = (mask[i].numpy() * 255).astype(np.uint8)
                
                # Overlay mask on image
                img_with_mask = self.draw_mask(img_display, mask_display)
                
                # Display
                cv2.imshow('Image with Mask', img_with_mask)
                key = cv2.waitKey(0)
                
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return

    def view_bb_dataset(self):
        """View images and bounding boxes from BB dataset."""
        for batch in self.dataloader:
            image = batch['image']
            bb = batch['bb']
            
            for i in range(len(image)):
                # Convert tensor to numpy array and prepare for display
                img_display = image[i].numpy()
                if len(img_display.shape) == 2:  # Grayscale
                    img_display = cv2.cvtColor((img_display * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                else:  # RGB
                    img_display = (img_display.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)

                # Get bounding box coordinates and convert to pixels
                bb_coords = (bb[i].numpy() * img_display.shape[0]).astype(int)
                points = [
                    Point(bb_coords[0], bb_coords[1], "Top Left"),
                    Point(bb_coords[2], bb_coords[3], "Top Right"),
                    Point(bb_coords[4], bb_coords[5], "Bottom Right"),
                    Point(bb_coords[6], bb_coords[7], "Bottom Left")
                ]
                
                # Draw bounding box
                self.draw_bounding_box(img_display, points)
                
                # Display
                cv2.imshow('Image with Bounding Box', img_display)
                key = cv2.waitKey(0)
                
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return

def main():
    # Example usage
    unet_dataset = ImagesDataSetUNet('bounding_boxes_points_all.csv', image_size=256)
    bb_dataset = ImagesDataSetBb('bounding_boxes_points_all.csv', image_size=256)
    
    # View UNet dataset
    # print("Viewing UNet dataset (press 'q' to quit)...")
    # unet_viewer = DatasetViewer(unet_dataset)
    # unet_viewer.view_unet_dataset()
    
    # View BB dataset
    print("Viewing BB dataset (press 'q' to quit)...")
    bb_viewer = DatasetViewer(bb_dataset)
    bb_viewer.view_bb_dataset()

if __name__ == '__main__':
    main()