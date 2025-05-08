# use given dataset images and masks
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from aux_functions import (
    clip_video,
    extract_frames,
    read_and_undistort_images,
    undistort
)
from task1 import extract_AO_Kmeans, extract_AO_k_means_ellipse
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.fft import fft, fftfreq


# Camera calibration parameters
camera_matrix = np.array([[891.63053566, 0., 689.13358247],
                          [0., 887.21568238, 386.16992913],
                          [0., 0., 1.]])

dist_coeffs = np.array([[-0.02120685, 0.12578736, 0.01656369, 0.00091947, -0.31341862]])


def detect_centriod(predict_mask):
    """
    Detect the centroid of the object
    
    Args:
        predict_mask (np.array): predicted mask
        
    Returns:    
        float: cx
        float: cy
        list: largest_contour
    """
    contours, _ = cv2.findContours(predict_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)

    moments = cv2.moments(largest_contour)
    if moments["m00"] != 0:
        cx = (moments["m10"] / moments["m00"])  # x 
        cy = (moments["m01"] / moments["m00"])  # y 
        # print(f"({cx:.2f}, {cy:.2f})")

    return cx, cy, largest_contour


def analyse_trajectory(centroids, ref_img, ref_contour, save_path=None,name=None, show_image=True, save_image=True):
    ''' 
    Analyse the trajectory of the centroids 
    
    Args:
        centroids (list): list of centroid coordinates
        ref_img (np.array): reference image
        ref_contour (list): reference contour
        save_path (str): path to save the image
        name (str): name of the dataset
        show_image (bool): show the image
        save_image (bool): save the image
    
    '''
    x_coords, y_coords = zip(*centroids)

    mean_x, mean_y = np.mean(x_coords), np.mean(y_coords)
    max_x, min_x = int(np.max(x_coords)), int(np.min(x_coords))
    max_y, min_y = int(np.max(y_coords)), int(np.min(y_coords))
    range_x, range_y = np.max(x_coords) - np.min(x_coords), np.max(y_coords) - np.min(y_coords)
    std_x, std_y = np.std(x_coords), np.std(y_coords)

    print("############################################")
    print(f"Average: ({mean_x:.2f}, {mean_y:.2f})")
    print(f"Swing Range: x={range_x:.2f}, y={range_y:.2f}")
    print(f"Standard Deviation: x={std_x:.2f}, y={std_y:.2f}")
    print("############################################")

    # Store average, swing range and standard deviation to a text file
    with open(f"{save_path}/trajectory_analysis.txt", "a") as f:
        f.write("Trajectory Analysis ({name}\n".format(name=name))
        f.write("Average: ({:.2f}, {:.2f})\n".format(mean_x, mean_y))
        f.write("Swing Range: x={:.2f}, y={:.2f}\n".format(range_x, range_y))
        f.write("Standard Deviation: x={:.2f}, y={:.2f}\n".format(std_x, std_y))
        f.write("############################################\n")


    num_points = len(centroids)
    colors = plt.cm.viridis(np.linspace(0, 1, num_points)) 

    # draw bounding box
    cv2.rectangle(ref_img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
    cv2.drawContours(ref_img, ref_contour, -1, (0, 255, 0), 2)

    for centroid in centroids:
        center = (int(centroid[0]), int(centroid[1]))
        cv2.circle(ref_img, center, 1, (0, 255, 0), -1)  


    if show_image:

        cv2.imshow("Geometric Center", ref_img)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty('Geometric Center', cv2.WND_PROP_VISIBLE) < 1 or key == ord('q'):
                print("break")
                break

        cv2.destroyAllWindows()
        # save the image
        cv2.imwrite(f"Dataset/task2/geometric_center_{name}", ref_img)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    # Plot the trajectory on the first subplot
    ax1.scatter(x_coords, y_coords, alpha=0.7, label='centroid', color=colors)
    ax1.axhline(np.mean(y_coords), color='r', linestyle='--', label='y mean')
    ax1.axvline(np.mean(x_coords), color='b', linestyle='--', label='x mean')
    ax1.set_xlabel('X ')
    ax1.set_ylabel('Y ')
    ax1.set_title('Distribution of Centroids')
    ax1.legend()

    # Plot the trajectory with time sequence on the second subplot
    for i in range(num_points - 1):
        ax2.plot(x_coords[i:i + 2], y_coords[i:i + 2], color=colors[i], linestyle='--', linewidth=2)
        ax2.text(x_coords[i], y_coords[i], str(i + 1), fontsize=8, color="black")
    ax2.scatter(x_coords[0], y_coords[0], color="green", label="Start", zorder=5)
    ax2.scatter(x_coords[-1], y_coords[-1], color="red", label="End", zorder=5)
    ax2.set_title("Centroid Trajectory with Time Sequence")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.legend()
    ax2.grid(True)


    if show_image:
        # Show the figure
        plt.show()

    # save the figure to path 
    if save_image:
        fig.savefig(f"{save_path}/trajectory_order_{name}")


def detect_outlier_area(areas):
    ''' 
    Relative difference in contour size
     
    Args:
        areas (list): list of contour areas
        
    Returns:
        list: filtered_indices'''

    filtered_indices = []
    mean_area = np.mean(areas)
    std_area = np.std(areas)

    for i in range(len(areas)):
        if areas[i] < mean_area + 2 * std_area or areas[i] > mean_area - 2 * std_area:
            filtered_indices.append(i)
            print(f"Area: {areas[i]:.2f}")
        else:
            print(f"Outlier detected (area: {areas[i]:.2f})")

    return filtered_indices


def detect_outlier_ellipse(contour):
    ''' 
    Absolute difference in contour shape 
    
    Args:
        contour (list): list of contour points
        
    Returns:
        bool: ellipse_flag
        float: aspect_ratio 
    '''
    ellipse_flag = False

    aspect_ratio_threshold = 1.15

    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (x, y), (major_axis, minor_axis), angle = ellipse

        aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)
        if aspect_ratio <= aspect_ratio_threshold:
            ellipse_flag = True

    return ellipse_flag, aspect_ratio



def detect_height(angle1, angle2, displacement):
    '''
    Triangular measurement approach for measuring height

    Args:
        angle1 (float): angle of the first camera
        angle2 (float): angle of the second camera
        displacement (float): displacement between the two cameras

    Returns:
        float: height of the object    
    '''
    angle1_rad = math.radians(angle1)
    angle2_rad = math.radians(angle2)
    height = (displacement * math.tan(angle1_rad) * math.tan(angle2_rad)) / (
                math.tan(angle2_rad) - math.tan(angle1_rad))

    return height



def fit_movement(center_x, center_y, timestep, name):
    """
    Fit the movement of the object with a pendulum model
    
    Args:
        center_x (list): list of x coordinates
        center_y (list): list of y coordinates
        timestep (list): list of time sequence
        name (str): name of the dataset
    """
    points = np.array([[x, y] for x, y in zip(center_x, center_y)]) 
    time = np.array(timestep)  
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))

    def pendulum_model(t, A, omega, phi, C):
        return A * np.cos(omega * t + phi) + C
    
    # Step 1: PCA
    mean = np.mean(points, axis=0)
    centered_points = points - mean  
    pca = PCA(n_components=2)
    pca.fit(centered_points)
    principal_axes = pca.components_

    # Step 2: Rotate the data
    rotated_points = centered_points @ principal_axes.T

    A_init_x = (np.max(rotated_points[:, 0]) - np.min(rotated_points[:, 0])) / 2  
    A_init_y = (np.max(rotated_points[:, 1]) - np.min(rotated_points[:, 1])) / 2
    def estimate_omega_fft(data, timestep):
        N = len(data)
        freq = fftfreq(N, d=np.mean(np.diff(timestep)))  
        fft_amplitude = np.abs(fft(data))
        dominant_freq = np.abs(freq[np.argmax(fft_amplitude[1:]) + 1])  
        return 2 * np.pi * dominant_freq

    # estimate omega
    omega_init_x = estimate_omega_fft(rotated_points[:, 0], time)
    omega_init_y = estimate_omega_fft(rotated_points[:, 1], time)
    # FIT the data
    print(f"Initial Amplitude: x={A_init_x:.2f}, y={A_init_y:.2f}")
    print(f"Initial Omega: x={omega_init_x:.2f}, y={omega_init_y:.2f}")
    params_x, _ = curve_fit(pendulum_model, time, rotated_points[:, 0], p0=[A_init_x, omega_init_x, 0, np.mean(rotated_points[:, 0])])
    params_y, _ = curve_fit(pendulum_model, time, rotated_points[:, 1], p0=[A_init_y, omega_init_y, 0, np.mean(rotated_points[:, 0])])

    # fitted function
    smoothed_x = pendulum_model(time, *params_x)
    smoothed_y = pendulum_model(time, *params_y)

    smoothed_x_origin = smoothed_x * principal_axes[0, 0] + smoothed_y * principal_axes[1, 0] + mean[0]
    smoothed_y_origin = smoothed_x * principal_axes[0, 1] + smoothed_y * principal_axes[1, 1] + mean[1]
    
    axs[1, 0].plot(points[:, 0], points[:, 1], 'o-', label='Original Trajectory', color='blue')
    axs[1, 0].plot(smoothed_x_origin, smoothed_y_origin, 'r-', label=f'Smoothed Trajectory)')
    axs[1, 0].set_title('Trajectory: Original vs Smoothed')
    axs[1, 0].set_xlabel('Original X')
    axs[1, 0].set_ylabel('Original Y')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # X' direction
    axs[0, 0].scatter(time, rotated_points[:, 0], color='b', s=10, label="Original X' Data")
    axs[0, 0].plot(time, smoothed_x, 'r-', linewidth=2, label="Fitted X' Function (FFT)")
    axs[0, 0].set_title("Principal Direction (X') - Fitted Function")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("X' Amplitude")
    axs[0, 0].legend()
    axs[0, 0].grid()

    # Y' direction
    axs[0, 1].scatter(time, rotated_points[:, 1], color='g', s=10, label="Original Y' Data")
    axs[0, 1].plot(time, smoothed_y, 'orange', linewidth=2, label="Fitted Y' Function (FFT)")
    axs[0, 1].set_title("Secondary Direction (Y') - Fitted Function")
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Y' Amplitude")
    axs[0, 1].legend()
    axs[0, 1].grid()

    axs[1, 1].plot(rotated_points[:, 0], rotated_points[:, 1], 'o-', label='Original Trajectory', color='blue')
    axs[1, 1].plot(smoothed_x, smoothed_y, 'r-', label=f'Smoothed Trajectory)')
    axs[1, 1].set_title('Trajectory: Original vs Smoothed')
    axs[1, 1].set_xlabel('Principal Component 1')
    axs[1, 1].set_ylabel('Principal Component 2')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    # plt.savefig(f"Dataset/task2/trajectory_{name}.png")
    plt.show()

if __name__ == "__main__":
    """
    Main function
    """
    # Pre-process 1: Clip video
    # clip_video(input_path="Dataset/task2/001_rotated.avi", output_path="Dataset/task2/2-b/video/001_rotated_clipped.avi", start_time=0, end_time=60)

    # Pre-process 2: Extract frames from given video
    # extract_frames(video_path="V/task2/2-b/video/001_rotated_clipped.avi", dst_folder="Dataset/task2/2-b/img/001_neo_long", EXTRACT_FREQUENCY=4)

    data_name = ["005_neo_long"]

    for i, name in enumerate(data_name):
        if name[2]=="1": thresh = 49.8
        elif name[2]=="2": thresh = 32.6
        elif name[2]=="3": thresh = 25.0
        elif name[2]=="5": thresh = 39.8
        print(f"Threshold {thresh} ...")
        print(f"Processing {name}...")  

        # Step 1: Read frames from directory
        folder_path = f"Dataset/task2/2-b/img/{name}"
        image_list = read_and_undistort_images(folder_path, camera_matrix, dist_coeffs)
        print(f"{len(image_list)} number of undistorted image were loaded")

        # Detect AO
        predict_masks = []
        contours = []
        centroids = []
        areas = []
        center_x = []
        center_y = []
        timestep = []
        for i, image in enumerate(image_list):
            # image = image_list[0]
            # predict_mask = extract_AO_color_ellipse(image, canny_threshold_1=0, canny_threshold_2=50)
            predict_mask = extract_AO_Kmeans(image, threshold=thresh)
            # predict_mask, img =  extract_AO_k_means_ellipse(original_img=image, predict_mask_color=predict_mask)
            predict_mask =  extract_AO_k_means_ellipse(original_img=image, predict_mask_color=predict_mask)
            if predict_mask is not False:
                predict_mask_gray = cv2.cvtColor(predict_mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
                area = cv2.countNonZero(predict_mask_gray)
                cx, cy, contour = detect_centriod(predict_mask_gray)
                # cv2.circle(ref_image, (int(filtered_centroids[i][0]), int(filtered_centroids[i][1])), 1, (0, 255, 0), -1)
                # cv2.imshow("Contours", img)
                # cv2.waitKey(0)
                # Detect Ellipse Outlier
                ellipse_flag, aspect_ratio = detect_outlier_ellipse(contour)
                if ellipse_flag:
                    predict_masks.append(predict_mask_gray)
                    contours.append(contour)
                    centroids.append((cx, cy))
                    center_x.append(cx)
                    center_y.append(cy)
                    timestep.append(i)
                    areas.append(area)
                    print(f"Aspect ratio: {aspect_ratio:.2f}")
                else:
                    print(f"Outlier detected (aspect ratio: {aspect_ratio:.2f})")


        # save centroid to txt file
        with open(f"Dataset/task2/2-b/centroid_{name}.txt", "w") as f:
            for i in range(len(centroids)):
                centroid = centroids[i]
                timestep_cur = timestep[i]
                f.write(f"{centroid[0]:.2f} {centroid[1]:.2f} {timestep_cur}\n")

        # Detect Size Outlier
        filtered_indices = detect_outlier_area(areas)
        filtered_pred_masks, filtered_centroids, filtered_contours = [], [], []
        for i in filtered_indices:
            filtered_pred_masks.append(predict_masks[i])
            filtered_centroids.append(centroids[i])
            filtered_contours.append(contours[i])

        # Plot trajectory
        ref_image = image_list[0].copy()  
        ref_contour = filtered_contours[0]
        analyse_trajectory(centroids, ref_image, ref_contour, save_path="Dataset/task2/2-b/", name=f"trajectory_{name}.png", show_image=False, save_image=False)
    

    for name in data_name:
        # read the txt file and plot the trajectory
        center_x = []
        center_y = []
        timestep = []
        with open(f"Dataset/task2/2-b/centroid_{name}.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.split()
                # print(data)
                center_x.append(float(data[0]))
                center_y.append(float(data[1]))
                timestep.append(float(data[2]))
        fit_movement(center_x, center_y, timestep, name)
        