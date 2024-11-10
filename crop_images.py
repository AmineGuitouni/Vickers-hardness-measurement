import cv2
import os

def CropImage(images_folder, image_name, SAVE_PATH):
    # Read image
    img = cv2.imread(os.path.join(images_folder, image_name))

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to create a mask
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the circular green region
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Crop the image based on the bounding rectangle of the circular area
        cropped_img = img[y:y+h, x:x+w]

        # Save the cropped image
        cv2.imwrite(os.path.join(SAVE_PATH, "c_" + image_name), cropped_img)

    else:
        print("No contour found!")

if __name__ == "__main__":
    save_path = "./data/images/cropped"
    Images_path = "./data/images/all"

    for filename in os.listdir(Images_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            CropImage(Images_path, filename, save_path)