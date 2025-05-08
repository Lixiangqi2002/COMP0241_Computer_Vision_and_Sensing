import cv2
import numpy as np
from task1 import extract_AO_Kmeans, extract_AO_k_means_ellipse
from task2ab import detect_centriod
import matplotlib.pyplot as plt


def plot_3d_sphere_with_points(center, radius, points_prev, points_curr, labels=None):
    """
    Visualize a 3D sphere and mark points on its surface.

    Parameters:
        center (tuple): Sphere center as (x, y, z).
        radius (float): Radius of the sphere.
        points_prev (list of np.ndarray): List of 3D points from the previous frame.
        points_curr (list of np.ndarray): List of 3D points from the current frame.
        labels (list of str): Optional, labels for the points.
    """
    phi, theta = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)

    x = center[0] + radius * np.sin(theta) * np.cos(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(theta)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='b', alpha=0.3, edgecolor='k')  # Draw sphere surface

    points_prev = np.array(points_prev)
    points_curr = np.array(points_curr)
    ax.scatter(points_prev[:, 0], points_prev[:, 1], points_prev[:, 2], color='r', s=50, label='Previous Points')
    ax.scatter(points_curr[:, 0], points_curr[:, 1], points_curr[:, 2], color='g', s=50, label='Current Points')

    if labels:
        for point, label in zip(points_curr, labels):
            ax.text(point[0], point[1], point[2], label, color='black', fontsize=10)

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3D Sphere with Points")
    plt.show()


def plot_3d_two_separate_spheres(center1, radius1, points1, points2, labels1=None, labels2=None):
    """
    Visualize two separate 3D spheres in two independent coordinate systems.

    Parameters:
        center1 (tuple): Center of the first sphere (x, y, z).
        radius1 (float): Radius of the first sphere.
        points1 (list of np.ndarray): List of 3D points for the first sphere.
        center2 (tuple): Center of the second sphere (x, y, z).
        radius2 (float): Radius of the second sphere.
        points2 (list of np.ndarray): List of 3D points for the second sphere.
        labels1 (list of str): Optional, labels for points on the first sphere.
        labels2 (list of str): Optional, labels for points on the second sphere.
    """
    # Generate mesh for spheres
    phi, theta = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)

    # First sphere
    x1 = center1[0] + radius1 * np.sin(theta) * np.cos(phi)
    y1 = center1[1] + radius1 * np.sin(theta) * np.sin(phi)
    z1 = center1[2] + radius1 * np.cos(theta)

    # Second sphere
    x2 = center1[0] + radius1 * np.sin(theta) * np.cos(phi)
    y2 = center1[1] + radius1 * np.sin(theta) * np.sin(phi)
    z2 = center1[2] + radius1 * np.cos(theta)

    points1 = np.array(points1)
    points2 = np.array(points2)

    # Plotting in two separate subplots
    fig = plt.figure(figsize=(14, 6))

    # First sphere subplot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(x1, y1, z1, color='b', alpha=0.3, edgecolor='k')
    ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2], color='red', s=50, label='Points on Sphere 1')
    if labels1:
        for point, label in zip(points1, labels1):
            ax1.text(point[0], point[1], point[2], label, color='black', fontsize=10)
    ax1.set_title("Sphere 1")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    # Second sphere subplot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(x2, y2, z2, color='g', alpha=0.3, edgecolor='k')
    ax2.scatter(points2[:, 0], points2[:, 1], points2[:, 2], color='red', s=50, label='Points on Sphere 2')
    if labels2:
        for point, label in zip(points2, labels2):
            ax2.text(point[0], point[1], point[2], label, color='black', fontsize=10)
    ax2.set_title("Sphere 2")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    Estimation of the average periods
    
    Args:
        real_time: bool, whether to run the script in real-time mode
        video_id: int, the video ID to process (1, 2, 3)
    """
    # Define constants and load video
    real_time = False
    video_id = 1  # 1,2,3

    filepath = 'Dataset/'
    if real_time:
        VIDEO_PATH = 2
        threshold_period = 1000
        video_id = 0
    else:
        if video_id == 1:
            VIDEO_PATH = filepath + '/task3/AOrotation_1.mp4'
            threshold_period = 130
        elif video_id == 2:
            VIDEO_PATH = filepath + '/task3/AOrotation_2.mp4'
            threshold_period = 175
        elif video_id == 3:
            VIDEO_PATH = filepath + '/task3/AOrotation_3.mp4'
            threshold_period = 185


    # Step 2: Initialize video capture and SIFT feature tracker
    cap = cv2.VideoCapture(VIDEO_PATH)
    first_frame_kmeans_mask = None
    first_frame = None

    while first_frame_kmeans_mask is None and first_frame is None:
        ret, first_frame = cap.read()
        # first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if not ret:
            print("Error: Could not read video.")
            exit(1)
        # cv2.imshow('First Frame', first_frame)
        # cv2.waitKey(1)

        # Mask the first frame (assuming extract_AO is defined elsewhere)
        first_frame_kmeans_mask = extract_AO_Kmeans(first_frame, 52)
        first_frame_mask, ellipse_contour = extract_AO_k_means_ellipse(first_frame, first_frame_kmeans_mask, task3=True)

    first_frame_mask = cv2.cvtColor(first_frame_mask, cv2.COLOR_BGR2GRAY)
    first_frame = cv2.bitwise_and(first_frame, first_frame, mask=first_frame_mask)

    # Detect SIFT features in the first frame
    sift = cv2.SIFT_create(nfeatures=100, contrastThreshold=0.02, edgeThreshold=10)
    gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    keypoints_first, descriptors_first = sift.detectAndCompute(gray_first, None)

    # BFMatcher for feature matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Placeholder for 3D points and transformations
    points_3d = []  # 3D points over time
    poses = []  # Camera poses over time
    frame_counter = 0
    rvec_prev = None  # Previous frame's rotation vector
    keypoints_prev = None
    timestamps = []
    timestamps.append(0)
    angular_velocities = []
    rotation_periods = []
    prev_frame = first_frame
    matches_to_previous = None
    descriptors_prev = None
    angular_velocity_sum = []
    avg_distance_sum = []
    avg_distance_first_sum = []
    flag_second_frame = True
    # Process video frame
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Video has ended. Exiting...")
            break

        frame_counter += 1

        # Apply mask and preprocess the frame
        # frame_mask = extract_AO(frame, canny_threshold_1=0, canny_threshold_2=150)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_mask = cv2.bitwise_and(frame, frame, mask=first_frame_mask)
        # Preprocess the frame
        gray_frame = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
        # Define the 3D sphere center (assume 2D (x, y) -> 3D (x, y, 0))
        cx, cy, contour = detect_centriod(gray_frame)
        if video_id == 0 or video_id == 1:
            SPHERE_CENTER_3D = np.array([cx, 0, cy])
        elif video_id == 2 or video_id == 3:
            SPHERE_CENTER_3D = np.array([cx, cy, 0])

        center, axes, orientation = ellipse_contour
        SPHERE_RADIUS = (max(axes) + min(axes)) / 2
        # print(f"Sphere Center: {SPHERE_CENTER_3D}, Sphere Radius: {SPHERE_RADIUS}")

        keypoints_current, descriptors_current = sift.detectAndCompute(gray_frame, None)

        # Match features with the first frame for period calculation
        matches_to_first = bf.match(descriptors_first, descriptors_current)
        matches_to_first = sorted(matches_to_first, key=lambda x: x.distance)

       
        # Calculate period using matches to the first frame
        if len(matches_to_first) > 0:
            avg_distance_first = np.mean([m.distance for m in matches_to_first])
            avg_distance_first_sum.append(avg_distance_first)
            print(f"Average Feature Matching Distance to the First Frame: {avg_distance_first:.2f}")
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            print(current_time)
            # print("cur_time: ", current_time - timestamps[-1])
            # print(f"threshold_period: {threshol/d_period}")
            if flag_second_frame:
                if real_time:
                    threshold_period = avg_distance_first
                print("Get Threshold for detecting Period: ", threshold_period)
                time_diff = current_time - timestamps[-1] + 50
                print("second_time: ", time_diff)
                flag_second_frame = False
            else:
                if avg_distance_first < threshold_period and (current_time - timestamps[-1] > time_diff):
                    rotation_period = current_time - timestamps[-1]
                    rotation_periods.append(np.array([current_time, rotation_period, avg_distance_first]))
                    timestamps.append(current_time)
                    print(
                        f"################################################ Rotation Period: {rotation_period:.2f}s ################################################")

        # Display matches
        frame_with_matches = cv2.drawMatches(first_frame, keypoints_first, frame, keypoints_current, matches_to_first,
                                             None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Cycle Period', frame_with_matches)

        # Update variables for the next iteration
        descriptors_prev = descriptors_current
        keypoints_prev = keypoints_current
        prev_frame = frame

        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Exiting...")
            cap.release()
            break

    # cap.release()
    cv2.destroyAllWindows()

    avg_distance_first_mean = np.mean(avg_distance_first_sum)
    avg_distance_first_variance = np.var(avg_distance_first_sum)


    # plot avg_distance_first_sum
    plt.figure(figsize=(10, 5))
    plt.plot(avg_distance_first_sum, color='tab:red', label='Average Distance')
    plt.axhline(y=avg_distance_first_mean, color='red', linestyle='--',
                label=f'Mean Average Distance: {np.mean(avg_distance_first_sum):.2f}')
    plt.fill_between(range(len(avg_distance_first_sum)), avg_distance_first_mean - avg_distance_first_variance,
                     avg_distance_first_mean + avg_distance_first_variance, color='red', alpha=0.2,
                     label='Variance Average Distance')
    # Plot rotation periods
    for timestep, period, avg_dis in rotation_periods:
        plt.axvline(x=timestep*10, color='blue', linestyle='--', label=f'Rotation Period: {period:.2f}s, Avg Distance: {avg_dis:2f}')
    plt.xlabel('Frame')
    plt.ylabel('Average Distance')
    plt.legend()
    plt.title('Average Distance for Matches Between the First and the Current Frame')
    # plt.savefig(filepath + f'/task3/avg_distance_first_sum_{real_time}_{video_id}.png')
    plt.show()