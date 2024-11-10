import numpy as np
import cv2

def order_contour_points(contour_points):
    # Calculate the centroid of the points
    centroid = np.mean(contour_points, axis=0)

    # Compute the angles of each point relative to the centroid
    angles = np.arctan2(contour_points[:, 1] - centroid[1], contour_points[:, 0] - centroid[0])

    # Sort the points based on the angles in counterclockwise order
    ordered_points = contour_points[np.argsort(angles)]

    return ordered_points

def mask_contour(contour_points_normalized, img_width, img_height):
    # Convert normalized coordinates to pixel coordinates
    contour_points = np.array([[contour_points_normalized[0], contour_points_normalized[1]],
                               [contour_points_normalized[2], contour_points_normalized[3]],
                               [contour_points_normalized[4], contour_points_normalized[5]],
                               [contour_points_normalized[6], contour_points_normalized[7]]], dtype=np.int32)

    # Order the contour points
    contour_points = order_contour_points(contour_points)

    # Create a blank mask (e.g., 256x256), initially all zeros (black)
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Fill the ordered contour on the mask with white (255)
    cv2.fillPoly(mask, [contour_points], 255)

    return mask