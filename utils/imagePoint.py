import cv2
import torch
import numpy as np

def has_close_points(points, min_distance=10):
    """
    Checks if there are any two points within a minimum distance of each other.
    
    Parameters:
        points (np.ndarray): Array of points of shape (4, 2).
        min_distance (float): Minimum allowed distance between any two points.
        
    Returns:
        bool: True if there are points closer than min_distance, False otherwise.
    """
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(points[i] - points[j])
            if distance < min_distance:
                return True
    return False

def find_corners(contour, target_points=4):
    """
    Approximates a contour to find exactly 4 corners.
    Uses iterative adjustment of epsilon until exactly 4 points are found.
    """
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.01  # Start with 1% of perimeter
    max_iterations = 100
    iterations = 0
     
    while iterations < max_iterations:
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) == target_points:
            return approx
        elif len(approx) > target_points:
            epsilon *= 1.1  # Increase epsilon to get fewer points
        else:
            epsilon *= 0.9  # Decrease epsilon to get more points
        iterations += 1
    
    # If we couldn't find exactly 4 points, return the closest approximation
    return cv2.approxPolyDP(contour, 0.02 * perimeter, True)[:16]

def order_points(pts):
    """
    Orders points in clockwise order starting from top-left.
    """
    # Convert points to a more manageable format
    pts = pts.reshape(4, 2)
    
    # Initialize ordered points array
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Top-left will have the smallest sum
    # Bottom-right will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right will have the smallest difference
    # Bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def image_corners(mask_pred, threshold=0.5):    
    mask_pred = torch.sigmoid(mask_pred).squeeze().cpu().numpy() > threshold
    mask_pred = (mask_pred * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Find the largest contour
        cnt = max(contours, key=cv2.contourArea)
        
        # Find exactly 4 corners
        corners = find_corners(cnt)
        print(corners)
        if len(corners[0]) != 4 or has_close_points(corners[0]):
            print("Warning: Could not detect 4 corners")
            print("Using minAreaRect instead")
            min_rect = cv2.minAreaRect(cnt)
            corners = cv2.boxPoints(min_rect)
            corners = np.int0(corners)

        # Order the corners
        ordered_corners = order_points(corners)

        return ordered_corners
    else:
        print("Error: No contours found in the image.")
        return None