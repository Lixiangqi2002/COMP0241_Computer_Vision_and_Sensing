from moviepy.video.io.VideoFileClip import VideoFileClip
import os
import cv2
import numpy as np


# Camera calibration parameters
camera_matrix = np.array([[891.63053566, 0., 689.13358247],
                          [0., 887.21568238, 386.16992913],
                          [0., 0., 1.]])

dist_coeffs = np.array([[-0.02120685, 0.12578736, 0.01656369, 0.00091947, -0.31341862]])



def clip_video(input_path, output_path, start_time, end_time):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"No file found at {input_path}")

    clip = VideoFileClip(input_path)

    short_clip = clip.subclip(start_time, end_time)

    short_clip.write_videofile(output_path, codec="libx264")


def extract_frames(video_path, dst_folder, EXTRACT_FREQUENCY, index=1):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("can not open the video")
        exit(1)

    count = 1
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break

        if count % EXTRACT_FREQUENCY == 0:
            save_path = "{}{:>05d}.jpg".format(dst_folder, index)
            print(f"save_path: {save_path}")
            # save_path = "{}/{}_{:>05d}.jpg".format(dst_folder, video_path.split('/')[-1][0:-4], index)
            cv2.imwrite(save_path, frame)
            frames.append(frame)
            index += 1
        count += 1
    print(f"index: {index}, count: {count}")
    video.release()
    cv2.destroyAllWindows()
    return


def clip_video(input_path, output_path, start_time, end_time):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"No file found at {input_path}")
    # 加载视频文件
    video = VideoFileClip(input_path)

    # 剪辑视频
    clipped_video = video.subclip(start_time, end_time)

    # 保存剪辑后的视频
    clipped_video.write_videofile(output_path, codec="libx264")


# Calibration
def undistort(camera_matrix, dist_coeffs, img):
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0, (w, h))

    # Undistort the image
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, new_camera_matrix)
    # undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
    # Crop the image (optional, based on ROI)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y + h, x:x + w]

    # Display the original and undistorted images in a figure
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # ax[0].set_title("Original Image")
    # ax[0].axis("off")
    # ax[1].imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
    # ax[1].set_title("Undistorted Image")
    # ax[1].axis("off")
    # plt.show()

    return undistorted_img


def read_and_undistort_images(folder_path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, valid_extensions=("jpg", "png")):
    image_list = []

    for filename in os.listdir(folder_path):

        if filename.endswith(valid_extensions):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"can not read {image_path}")
                continue


            undistorted_img = undistort(camera_matrix, dist_coeffs, image)
            image_list.append(undistorted_img)

    return image_list


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"can not read {image_path}")
    return image


def rename_files(src_folder, dst_folder, prefix, suffix):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for count, filename in enumerate(os.listdir(src_folder), start=1):
        src_path = os.path.join(src_folder, filename)
        if os.path.isfile(src_path):
            new_filename = f"{prefix}{count:05d}{suffix}{os.path.splitext(filename)[1]}"
            dst_path = os.path.join(dst_folder, new_filename)
            os.rename(src_path, dst_path)
            print(f"Renamed {src_path} to {dst_path}")

from matplotlib.animation import FuncAnimation
def animate_points(points, interval=500):
    fig, ax = plt.subplots()
    sc = ax.scatter([], [])

    def update(frame):
        sc.set_offsets(points[:frame])
        return sc,

    
    points_array = np.array(points)

    ani = FuncAnimation(fig, update, frames=len(points_array)+1, interval=interval, blit=True)
    plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt


def remove_noise_and_display(image, min_area=500):
    # image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_ERODE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    cleaned_image = np.zeros_like(binary_image)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_image[labels == i] = 255

    return cleaned_image

