import cv2
import os
import csv

SKIP_BTN = "q"
SAVE_BTN = "s"
RESET_BTN = "r"
BACK_BTN = "b"
ZOOM_IN_BTN = "+"
ZOOM_OUT_BTN = "-"

# Variables to store bounding box points
points = []
zoom_factor = 1.0  # Initialize zoom factor

# Mouse callback function to register points and remove the last one with right-click
def select_points(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:  # Left-click to add a point
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(img_resized, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("Image", img_resized)

def resize_image(image, max_width, max_height, zoom=1.0):
    """Resize image to fit within max_width and max_height while maintaining aspect ratio and applying zoom."""
    height, width = image.shape[:2]
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h) * zoom  # Apply zoom factor

    if scale != 1.0:  # Resize based on scale (either for zoom or fitting to screen)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image, scale
    return image, 1.0

def label_images(image_folder, output_file, screen_width, screen_height):
    global img_resized, points, zoom_factor

    # Open the CSV file to store bounding box points
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])

        # Loop through all images in the folder
        for filename in os.listdir(image_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                # Load the image
                img_path = os.path.join(image_folder, filename)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Error loading {filename}")
                    continue

                # Reset points and zoom factor for each image
                points = []
                zoom_factor = 1.0

                # Resize image to fit within screen dimensions
                img_resized, scale_factor = resize_image(img, screen_width, screen_height, zoom_factor)
                resize_image_height, resize_image_width = img_resized.shape[:2]
                
                # Create a window and set a mouse callback function
                cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

                cv2.setMouseCallback("Image", select_points)
                cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


                # Display the image
                while True:
                    cv2.imshow("Image", img_resized)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord(RESET_BTN):  # Reset the points
                        points = []
                        print("Reset points")
                        # Redraw the image without the points
                        img_resized, scale_factor = resize_image(img, screen_width, screen_height, zoom_factor)
                        cv2.imshow("Image", img_resized)

                    if key == ord(BACK_BTN) and points:  # Undo the last point
                        points.pop()  # Remove the last point
                        print("Removed the last point")
                        img_resized, scale_factor = resize_image(img, screen_width, screen_height, zoom_factor)

                        # Redraw the remaining points
                        for (x, y) in points:
                            cv2.circle(img_resized, (x, y), 3, (0, 0, 255), -1)

                        cv2.imshow("Image", img_resized)

                    if key == ord(SAVE_BTN) and len(points) == 4:  # Save after selecting 4 points
                        # Scale the points back to original size
                        scaled_points = [(x / resize_image_width, y / resize_image_height) for (x, y) in points]
                        writer.writerow([filename, *sum(scaled_points, ())])  # Flatten list of points
                        print(f"Saved points for {filename}: {scaled_points}")
                        break

                    elif key == ord(SKIP_BTN):  # Skip the image
                        break

                    if key == ord(ZOOM_IN_BTN):  # Zoom in
                        zoom_factor += 0.1  # Increase zoom factor
                        img_resized, scale_factor = resize_image(img, screen_width, screen_height, zoom_factor)
                        print(f"Zoomed in: {zoom_factor}")
                        cv2.imshow("Image", img_resized)

                    if key == ord(ZOOM_OUT_BTN):  # Zoom out
                        zoom_factor = max(0.1, zoom_factor - 0.1)  # Decrease zoom factor, but don't go below 0.1
                        img_resized, scale_factor = resize_image(img, screen_width, screen_height, zoom_factor)
                        print(f"Zoomed out: {zoom_factor}")
                        cv2.imshow("Image", img_resized)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Directory containing the images to be labeled
    image_folder = './data/images/'

    # Output CSV file to save bounding box points
    output_file = 'bounding_boxes_points.csv'

    # Get the screen resolution (manually set this to match your screen)
    screen_width, screen_height = 1920, 1080

    # Run the labeling function
    label_images(image_folder, output_file, screen_width, screen_height)