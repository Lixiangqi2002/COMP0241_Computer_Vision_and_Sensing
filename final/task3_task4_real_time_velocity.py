import math

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


def compute_displacement_vectors(matches, keypoints_prev, keypoints_current):
    """
    Compute 2D displacement vectors for matches.

    Parameters:
        matches (list): List of DMatch objects.
        keypoints_prev (list): List of KeyPoint objects from the previous frame.
        keypoints_current (list): List of KeyPoint objects from the current frame.

    Returns:
        displacements (list): List of 2D displacement vectors.  # (prev_pt, curr_pt)
    """
    displacements = []
    for m in matches:
        prev_pt = np.array(keypoints_prev[m.queryIdx].pt)
        # print("Previous:",prev_pt)
        curr_pt = np.array(keypoints_current[m.trainIdx].pt)
        # print("Current:",curr_pt)
        if np.linalg.norm(curr_pt - prev_pt) < 5:
            displacements.append((prev_pt, curr_pt))
            # print(f"Displacement: {prev_pt, curr_pt}")
    # print("Displacements:", displacements)
    return displacements

def get_rotation_axis_angle(radius, axis_point, center):
    """
    Compute the angle between the rotation axis and the vertical axis.
    
    Parameters:
        radius (float): Sphere radius.
        axis_point (list): Point on the rotation axis.
        center (tuple): Sphere center.
        
    Returns:
        angle (float): Angle between the rotation axis and the vertical axis.
        pos (str): Relative position of the rotation axis.
    """
    axis_point = axis_point[0]
    distance = math.sqrt((axis_point[0]-center[0])**2 + (axis_point[1] - center[1])**2)
    ratio = distance/radius
    angle = np.arccos(ratio)
    pos = "Middle"
    # print("Ratio: ", ratio)
    if 0.95 < ratio < 1.05:
        pos = "Middle"
    elif ratio<0.55:
        pos = "center"
    else:
        if center[1] < axis_point[1] - 10 :
            pos = "bottom"
        elif center[1] > axis_point[1] + 10:
            pos = "top"
    # print("Relative Position: ", pos)
    return angle, pos


# Callback function to handle mouse clicks
def select_point(event, x, y, flags, param):
    """
    Callback function to handle mouse clicks.
    """
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        x, y = to_centered_coordinates(x, y)
        print(f"Selected point: ({x}, {y})")
        selected_points.append((x, y))  # Save the selected point


def rotation_point(event, x, y, flags, param):
    """
    Callback function to handle mouse clicks for selecting the rotation axis point.
    """
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        x, y = to_centered_coordinates(x, y)
        print(f"Rotation axis point: ({x}, {y})")
        rotation_point.append([x,y])

def to_3d_vector(rotation_axis_angle, position, point_2d, sphere_center, sphere_radius):
    """
    Convert a 2D point on the sphere projection to a 3D point.

    Parameters:
        rotation_axis_angle (float): Angle between the rotation axis and the vertical axis.
        position (str): Relative position of the rotation axis.
        point_2d (tuple): 2D point on the sphere projection.
        sphere_center (tuple): Sphere center coordinates.
        sphere_radius (float): Sphere radius.

    Returns:
        new_point_3d (np.ndarray): 3D point on the sphere surface    
    """
    uc, m, vc = sphere_center

    x, z = point_2d
    y = np.sqrt(sphere_radius ** 2 - (x - uc) ** 2 - (z - vc) ** 2)
    point_3d = np.array([x,y,z])
    if position=="top":
        rotation_axis_angle = rotation_axis_angle
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(rotation_axis_angle), -np.sin(rotation_axis_angle)],
                        [0, np.sin(rotation_axis_angle), np.cos(rotation_axis_angle)]])
    elif position=="bottom":
        rotation_axis_angle = - rotation_axis_angle
        # print(rotation_axis_angle)
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(rotation_axis_angle), -np.sin(rotation_axis_angle)],
                        [0, np.sin(rotation_axis_angle), np.cos(rotation_axis_angle)]])
    elif position=="center":
        rotation_axis_angle = -3.14 +rotation_axis_angle
        # print(rotation_axis_angle)
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(rotation_axis_angle), -np.sin(rotation_axis_angle)],
                        [0, np.sin(rotation_axis_angle), np.cos(rotation_axis_angle)]])
    else:
        R_x = np.identity(3)
    new_point_3d = np.dot(R_x, point_3d)
    # print(f"3D Point: ({x}, {y}, {z})")
    return new_point_3d


def compute_radius_and_angular_velocity(displacements_3d, sphere_center, sphere_radius, position, timestep):
    """
    Compute radii and approximate angular velocities.

    Parameters:
        displacements_3d (list): List of 3D displacement vectors.
        sphere_center (tuple): Sphere center coordinates.
        sphere_center_radius (float): Sphere center radius.

    Returns:
        angular_velocities (list): List of approximate angular velocities.      
    """
    angular_velocities = []
    for prev, curr in displacements_3d:
        displacement_vector = curr - prev
        displacement_vector = displacement_vector/timestep
        radius = np.sqrt((curr[0] - sphere_center[0]) ** 2 + (curr[2] - sphere_center[2]) ** 2)
        angular_velocity = np.linalg.norm(displacement_vector) / radius
        # print(f"Angular Velocity: {angular_velocity:.6f}")
        if angular_velocity>0:
            angular_velocities.append(angular_velocity)
    return angular_velocities


def calculate_linear_velocities(points, angular_vel, sphere_center, sphere_radius, lattiudes):
    """
    Calculate linear velocities for points on the sphere surface. 
    Latitude is calculated using the displacement from the sphere center.
    
    Parameters:
        points (list): List of 3D points on the sphere projection.
        angular_vel (float): Angular velocity of the sphere.
        sphere_center (tuple): Sphere center coordinates.
        sphere_radius (float): Sphere radius.
    
    Returns:
        velocities (list): List of linear velocities for each point.
    """
    velocities = []
    for i in range(len(points)):
        point = points[i]
        lattitude = lattiudes[i]
        # print("Point and Lattitude: ", point, lattitude)
        # displacement = np.abs(point[2] - sphere_center[2])
        # lattitude = np.arcsin(displacement/sphere_radius)
        lattitude = lattitude * (np.pi / 180)
        r_pixel = sphere_radius * np.cos(lattitude)
        radius = r_pixel/sphere_radius * (6.07/2)
        # print("real radius： ", radius)
        velocity = angular_vel * radius
        velocities.append(velocity)
    return velocities


def get_sphere_frame(pos, center_2D):
    u, v = center_2D
    return np.array([u,0,v])


def to_centered_coordinates(x, y):
    x_centered = x - ellipse_center_x
    y_centered = y - ellipse_center_y
    return x_centered, y_centered


if __name__ == "__main__":
    # Define constants and load video
    real_time = False
    video_id =  4 # 1,2,3,4
    filepath  = 'Dataset/'
    if real_time:
        VIDEO_PATH = 2
        threshold_period = 185
        video_id = 0
    else:
        if video_id == 1:
            VIDEO_PATH = filepath + '/task4/AOrotation_1.mp4'
        elif video_id == 2:
            VIDEO_PATH = filepath + '/task4/AOrotation_2.mp4'
        elif video_id == 3:
            VIDEO_PATH = filepath + '/task4/AOrotation_3.mp4'
        elif video_id == 4:
            VIDEO_PATH = filepath + '/task4/video_floor_7.avi'
            
    # Step 2: Initialize video capture and SIFT feature tracker
    cap = cv2.VideoCapture(VIDEO_PATH)
    timestep = 0.1
    first_frame_kmeans_mask = None
    first_frame = None
    while first_frame_kmeans_mask is None and first_frame is None:
        ret, first_frame = cap.read()

        if not ret:
            print("Error: Could not read video.")
            exit(1)
        first = first_frame.copy()
        # Mask the first frame (assuming extract_AO is defined elsewhere)
        first_frame_kmeans_mask = extract_AO_Kmeans(first_frame, 52)
        first_frame_mask, ellipse_contour = extract_AO_k_means_ellipse(first_frame, first_frame_kmeans_mask, task3=True)
        first_frame_mask = cv2.cvtColor(first_frame_mask, cv2.COLOR_BGR2GRAY)
        first_frame = cv2.bitwise_and(first_frame, first_frame, mask=first_frame_mask)
        # Define the 2D sphere center
        cx, cy, contour = detect_centriod(first_frame_mask)
        ellipse_center_x, ellipse_center_y = cx, cy
        cx, cy = to_centered_coordinates(cx, cy)
        SPHERE_CENTER = np.array([cx, cy])
        print("Sphere Center: ", SPHERE_CENTER)
        # SPHERE_CENTER_3D = np.array([cx, 0, cy])
        center, axes, orientation = ellipse_contour
        SPHERE_RADIUS = (max(axes) + min(axes)) / 4
        # print(SPHERE_RADIUS)
        # print(f"Sphere Center: {SPHERE_CENTER_3D}, Sphere Radius: {SPHERE_RADIUS}")
    cv2.ellipse(first, ellipse_contour, (0, 0, 255), 2)  # Draw yellow ellipse
    image_width = first.shape[1]
    image_height = first.shape[0]
    # image_center_x = image_width / 2
    # image_center_y = image_height / 2
    # Display the first frame and set up the mouse callback
    cv2.namedWindow("Select a Point")
    print("###################### Step 1: Select Rotation Axis Endpoint ######################")
    print("Click on the frame to select a point on the rotation axis, could be bottom point or top point. ")
    print("DON'T SELECT OUTSIDE THE YELLOW ELLIPSE!")
    print("Press 'q' to quit.")
    print("###################### Step 2: Select Points for Velocity Tracking ######################")

    cv2.setMouseCallback("Select a Point", rotation_point)
    # 1. Selected the top or bottom point on the rotation axis
    rotation_point = []
    while len(rotation_point)==0:
        # Show the first frame
        cv2.imshow("Select a Point", first)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    print(f"Rotation point: {rotation_point}")
    # 2. selected points for tracking
    cv2.setMouseCallback("Select a Point", select_point)
    print("Click on the frame to select a point. Press 'q' to quit.")
    selected_points = []
    while True:
        # Show the first frame
        cv2.imshow("Select a Point", first_frame_mask)
        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    num_points = len(selected_points)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # if num_frames>200:
    #     num_frames = 200
    selected_points_lin_vel = np.zeros((num_points, num_frames))

    # Save the selected points to a file
    if selected_points:
        # np.save("selected_points.npy", np.array(selected_points))
        print(f"Selected points saved: {selected_points}")
    else:
        print("No points were selected.")
    selected_points = np.array(selected_points)
    selected_points_transpose = selected_points.T

    # Detect SIFT features in the first frame
    sift = cv2.SIFT_create(nfeatures=100, contrastThreshold=0.02, edgeThreshold=10)
    gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    keypoints_first_1, descriptors_first = sift.detectAndCompute(gray_first, None)
    keypoints_first = []
    for kp in keypoints_first_1:
        x_centered = kp.pt[0] - ellipse_center_x
        y_centered = kp.pt[1] - ellipse_center_y
        centered_kp = cv2.KeyPoint(x_centered, y_centered, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        keypoints_first.append(centered_kp)
        # BFMatcher for feature matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Placeholder for 3D points and transformations
    points_3d = []  # 3D points over time
    poses = []  # Camera poses over time
    frame_counter = 0

    rvec_prev = None  # Previous frame's rotation vector
    keypoints_prev = None
    timestamps = []
    angular_velocities = []
    rotation_periods = []
    prev_frame = first_frame
    timestamps = []
    timestamps.append(0)
    matches_to_previous = None
    descriptors_prev = None
    angular_velocity_sum = []
    avg_distance_sum = []
    avg_distance_first_sum = []
    flag_second_frame = True
    keypoints_prev_1 = None
    frame_id = 0
    peroid_arr = []
    # Process video frames
    while cap.isOpened():
        # if frame_id >= num_frames:
        #     break
        ret, frame = cap.read()

        if not ret:
            print("Video has ended. Exiting...")
            break

        frame_counter += 1

        # Apply mask and preprocess the frame
        frame_mask = cv2.bitwise_and(frame, frame, mask=first_frame_mask)
        # Preprocess the frame
        gray_frame = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)

        keypoints_current_1, descriptors_current = sift.detectAndCompute(gray_frame, None)
        keypoints_current = []

        for kp in keypoints_current_1:
            x_centered = kp.pt[0] - ellipse_center_x
            y_centered = kp.pt[1] - ellipse_center_y
            centered_kp = cv2.KeyPoint(x_centered, y_centered, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            keypoints_current.append(centered_kp)

        # Match features with the first frame for period calculation
        matches_to_first = bf.match(descriptors_first, descriptors_current)
        matches_to_first = sorted(matches_to_first, key=lambda x: x.distance)

        # Match features with the previous frame for angular velocity
        if keypoints_prev is not None and descriptors_prev is not None:
            matches_to_previous = bf.match(descriptors_prev, descriptors_current)
            matches_to_previous = sorted(matches_to_previous, key=lambda x: x.distance)
            matches_to_previous = matches_to_previous[:30]
            avg_distance = np.mean([m.distance for m in matches_to_previous])
            # print(f"Average Feature Matching Distance: {avg_distance:.2f}")
            avg_distance_sum.append(avg_distance)
            if len(matches_to_previous) > 0:
                displacements = compute_displacement_vectors(matches_to_previous, keypoints_prev, keypoints_current)
                rotation_axis_angle, top_or_bottom = get_rotation_axis_angle(SPHERE_RADIUS, rotation_point, SPHERE_CENTER)
                SPHERE_CENTER_3D = get_sphere_frame(top_or_bottom, SPHERE_CENTER)
                displacements_3d = [(to_3d_vector(rotation_axis_angle, top_or_bottom, d[0], SPHERE_CENTER_3D, SPHERE_RADIUS),
                                     to_3d_vector(rotation_axis_angle, top_or_bottom, d[1], SPHERE_CENTER_3D, SPHERE_RADIUS)) for d in displacements]
                # plot_3d_two_separate_spheres(SPHERE_CENTER_3D, SPHERE_RADIUS, [d[0] for d in displacements_3d],
                #                              [d[1] for d in displacements_3d])
                selected_points_3d = to_3d_vector(rotation_axis_angle, top_or_bottom, selected_points_transpose, SPHERE_CENTER_3D, SPHERE_RADIUS)
                latitudes = []
                selected_points_3d = selected_points_3d.T
                # print("Selected Points 3D: ", selected_points_3d)
                for point in selected_points_3d:
                    point = point.T
                    displacement = np.abs(point[2] - SPHERE_CENTER_3D[2])
                    latitude = np.arcsin(displacement / SPHERE_RADIUS)
                    latitudes.append(latitude * (180 / np.pi))
                # print("Latitudes: ", latitudes)
                angular_velocities = compute_radius_and_angular_velocity(displacements_3d, SPHERE_CENTER_3D, SPHERE_RADIUS, top_or_bottom, timestep)
                angular_velocity = np.mean(angular_velocities)
                if angular_velocity > 0 :
                    print(f"Angular Velocity = {angular_velocity:.4f} rad/s")
                    angular_velocity_deg = angular_velocity * (180 / np.pi)
                    # print(f"Angular Velocity = {angular_velocity_deg:.2f} degrees/s")
                    angular_velocity_sum.append(angular_velocity)
                    cur_peroid = 360 / angular_velocity_deg
                    peroid_arr.append(cur_peroid)
                    print(f"Period = {cur_peroid:.4f} s")
                    velocities_selected_points = calculate_linear_velocities(selected_points_3d, angular_velocity, SPHERE_CENTER_3D, SPHERE_RADIUS, latitudes)

                    for i in range(len(velocities_selected_points)):
                        point_idx = selected_points[i]
                        velocity = velocities_selected_points[i]
                        print(f"Point {point_idx} linear velocity: {velocity:.6f} m/s")
                        selected_points_lin_vel[i, frame_id] = velocity

        if matches_to_previous is not None or keypoints_prev is not None:
            frame_with_matches = cv2.drawMatches(prev_frame, keypoints_prev_1, frame, keypoints_current_1,
                                                 matches_to_previous, None,
                                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('Angular Velocity', frame_with_matches)
        else:
            print("No matches to draw.")

        # Update variables for the next iteration
        descriptors_prev = descriptors_current
        keypoints_prev = keypoints_current
        keypoints_prev_1 = keypoints_current_1
        prev_frame = frame
        frame_id += 1

        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Exiting...")
            cap.release()
            break

    # cap.release()
    cv2.destroyAllWindows()
    # print(selected_points_lin_vel)

    # Calculate mean and variance of the period array
    period_mean = np.mean(peroid_arr)
    period_variance = np.var(peroid_arr)
    # print(peroid_arr)
    # print(angular_velocity_sum)
    # Plot the period array with mean and variance
    plt.figure(figsize=(10, 5))
    plt.plot(peroid_arr, color='tab:green', label='Period')
    plt.axhline(y=period_mean, color='green', linestyle='--',
                label=f'Mean Period: {period_mean:.2f}')
    # plt.fill_between(range(len(peroid_arr)), period_mean - period_variance,
    #                  period_mean + period_variance, color='green', alpha=0.2,
    #                  label='Variance Period')
    plt.xlabel('Frame')
    plt.ylabel('Period (s)')
    plt.legend()
    plt.title('Period Array')
    plt.savefig(filepath + f'/task4/period_arr_{real_time}_{video_id}.png')
    plt.show()

    angular_velocity_mean = np.mean(angular_velocity_sum)
    angular_velocity_variance = np.var(angular_velocity_sum)
    avg_distance_mean = np.mean(avg_distance_sum)
    avg_distance_variance = np.var(avg_distance_sum)
    
    # plot angular_velocity_sum
    plt.figure(figsize=(10, 5))
    plt.plot(angular_velocity_sum, color='tab:blue', label='Angular Velocity')
    plt.axhline(y=angular_velocity_mean, color='blue', linestyle='--',
                label=f'Mean Angular Velocity: {np.mean(angular_velocity_sum):.5f}')
    plt.fill_between(range(len(angular_velocity_sum)), angular_velocity_mean - angular_velocity_variance,
                     angular_velocity_mean + angular_velocity_variance, color='blue', alpha=0.2,
                     label='Variance Angular Velocity')
    plt.xlabel('Frame')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.title('Angular Velocity Sum')
    plt.savefig(filepath + f'/task4/angular_velocity_sum_{real_time}_{video_id}.png')
    plt.show()

    # plot avg_distance_sum
    plt.figure(figsize=(10, 5))
    plt.plot(avg_distance_sum, color='tab:red', label='Average Distance')
    plt.axhline(y=avg_distance_mean, color='red', linestyle='--',
                label=f'Mean Average Distance: {np.mean(avg_distance_sum):.3f}')
    plt.fill_between(range(len(avg_distance_sum)), avg_distance_mean - avg_distance_variance,
                     avg_distance_mean + avg_distance_variance, color='red', alpha=0.2,
                     label='Variance Average Distance')
    plt.xlabel('Frame')
    plt.ylabel('Average Distance')
    plt.legend()
    plt.title('Average Distance for Matches Between the Current and the Previous Frame')
    plt.savefig(filepath + f'/task4/avg_distance_sum_{real_time}_{video_id}.png')
    plt.show()

    # Remove rows where the first dimension of selected_points_lin_vel is 0
    selected_points_lin_vel = selected_points_lin_vel[~np.all(selected_points_lin_vel == 0, axis=1)]

    # Plot velocity vs latitude for all points
    plt.figure(figsize=(12, 8))
    for i in range(selected_points_lin_vel.shape[0]):
        latitude_cur = latitudes[i]
        plt.plot(
            np.arange(selected_points_lin_vel.shape[1]),
            selected_points_lin_vel[i, :],
            label=f"Point with Latitude {latitude_cur:.2f}°",
            alpha=0.6
        )

    # Configure the plot
    plt.title("Linear Velocity vs Latitude", fontsize=16)
    plt.xlabel("Frame", fontsize=14)
    plt.ylabel("Linear Velocity (m/s)", fontsize=14)
    plt.legend(title="Points", loc='upper right', fontsize=10, ncol=2)
    plt.savefig(filepath + f'/task4/linear_velocity_sum_{real_time}_{video_id}.png')
    plt.grid(False)
    plt.tight_layout()

    # Show the plot
    plt.show()

