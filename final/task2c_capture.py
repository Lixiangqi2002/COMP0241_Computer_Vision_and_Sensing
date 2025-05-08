import cv2
import os
from datetime import datetime

def initialize_cameras(left_device='/dev/video2', right_device='/dev/video4'):
    """
    Initialize the left and right cameras.

    Args:
        left_device (str): Left camera device number.
        right_device (str): Right camera device number.

    Returns:
        tuple: VideoCapture objects for the left and right cameras.
    """
    left_camera = cv2.VideoCapture(left_device)
    right_camera = cv2.VideoCapture(right_device)

    if not left_camera.isOpened():
        raise RuntimeError(f"Unable to open left camera: {left_device}")
    if not right_camera.isOpened():
        raise RuntimeError(f"Unable to open right camera: {right_device}")

    return left_camera, right_camera

def capture_images(left_camera, right_camera, save_dir, index):
    """
    Capture frames from the left and right cameras and save them to the specified directory.

    Args:
        left_camera (cv2.VideoCapture): Left camera object.
        right_camera (cv2.VideoCapture): Right camera object.
        save_dir (str): Directory to save the images.
        index (int): Index to track the photo sequence number.
    """
    os.makedirs(save_dir, exist_ok=True)

    ret_left, frame_left = left_camera.read()
    ret_right, frame_right = right_camera.read()

    if not ret_left or not ret_right:
        print("Unable to read camera frames")
        return

    # Generate filenames using the index

    # Save left photo
    left_image_path = os.path.join(save_dir, f"left_{index}.jpg")
    cv2.imwrite(left_image_path, frame_left)

    # Save right photo
    right_image_path = os.path.join(save_dir, f"right_{index}.jpg")
    cv2.imwrite(right_image_path, frame_right)

    print(f"Photos saved: {left_image_path} and {right_image_path}")


def display_cameras(left_camera, right_camera):
    """
    Display real-time footage from the left and right cameras and listen for key events.

    Args:
        left_camera (cv2.VideoCapture): Left camera object.
        right_camera (cv2.VideoCapture): Right camera object.
    """
    print("Press 'c' to capture left and right photos, press 'q' to exit the program")

    count = 0
    while True:
        # Read frames from the left and right cameras
        ret_left, frame_left = left_camera.read()
        ret_right, frame_right = right_camera.read()

        if not ret_left or not ret_right:
            print("Unable to read camera frames")
            break

        # Display images from the left and right cameras
        cv2.imshow("Left Camera", frame_left)
        cv2.imshow("Right Camera", frame_right)

        # Detect key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Press 'q' key to exit
            break
        elif key == ord('c'):  # Press 'c' key to capture photos
            capture_images(left_camera, right_camera, "Dataset/task2/", count)
            count += 1

    # Release camera resources and close windows
    left_camera.release()
    right_camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Initialize cameras
        left_camera, right_camera = initialize_cameras()

        # Display real-time footage from the cameras
        display_cameras(left_camera, right_camera)
    except Exception as e:
        print(f"An error occurred: {e}")
