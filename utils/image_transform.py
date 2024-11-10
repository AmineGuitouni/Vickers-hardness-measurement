import csv
import os
import albumentations as A
import cv2

def denormalize_point(x: float, y: float, img_width: int, img_height: int):
        """Convert normalized coordinates (0-1) to pixel values."""
        return int(x * img_width), int(y * img_height)

def transform(image, raw_keypoints, size=(256, 256)):
    # Define the augmentation pipeline
    transform = A.Compose([
        A.Resize(height=size[0], width=size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=45, p=0.7, border_mode=cv2.BORDER_CONSTANT)
    ], keypoint_params=A.KeypointParams(format='xy'))
   
    img_height, img_width = image.shape[:2]
    x1, y1 = denormalize_point(raw_keypoints[0], raw_keypoints[1], img_width, img_height)
    x2, y2 = denormalize_point(raw_keypoints[2], raw_keypoints[3], img_width, img_height)
    x3, y3 = denormalize_point(raw_keypoints[4], raw_keypoints[5], img_width, img_height)
    x4, y4 = denormalize_point(raw_keypoints[6], raw_keypoints[7], img_width, img_height)

    keypoints = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    # Apply the augmentation
    augmented = transform(image=image, keypoints=keypoints)
    
    return augmented['image'], augmented['keypoints']

def safe_transform(image, raw_keypoints, size=(256, 256)):
    # Define the augmentation pipeline
    transform = A.Compose([
        A.Resize(height=size[0], width=size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ], keypoint_params=A.KeypointParams(format='xy'))
   
    img_height, img_width = image.shape[:2]
    x1, y1 = denormalize_point(raw_keypoints[0], raw_keypoints[1], img_width, img_height)
    x2, y2 = denormalize_point(raw_keypoints[2], raw_keypoints[3], img_width, img_height)
    x3, y3 = denormalize_point(raw_keypoints[4], raw_keypoints[5], img_width, img_height)
    x4, y4 = denormalize_point(raw_keypoints[6], raw_keypoints[7], img_width, img_height)

    keypoints = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    # Apply the augmentation
    augmented = transform(image=image, keypoints=keypoints)
    
    return augmented['image'], augmented['keypoints']

if __name__ == '__main__':
    s = 0
    with open("./bounding_boxes_points.csv", newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row

        for row in reader:
            image_name = row[0]
            normalized_coords = list(map(float, row[1:]))
            if len(normalized_coords) != 8 : print(normalized_coords)
            image = cv2.imread(os.path.join("./images/cropped", image_name))
            
            for i in range(100):
                augmented_image, augmented_keypoints = transform(image, normalized_coords)
                # Draw the augmented keypoints on the image
                #draw_box(augmented_image, augmented_keypoints)
                if len(augmented_keypoints) != 4 : s += 1
                # Display the augmented image
                # cv2.imshow('Augmented Image', augmented_image)
                # cv2.waitKey(0)

    print(s)