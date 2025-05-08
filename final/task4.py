import os

from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
from sklearn.decomposition import PCA
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import numpy as np
import colorsys
from scipy.optimize import least_squares
from scipy.spatial import distance_matrix

def read_ply_file(file_path):
    ply_data = PlyData.read(file_path)
    vertex = ply_data['vertex']
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    return points


def plot_point_cloud(points, title):
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title(title)
    plt.show()


def find_dense_hemisphere(points):

    distances = np.linalg.norm(points, axis=1)
    # find the center point with the smallest distance to all other points
    center_index = np.argmin(distances)
    center_point = points[center_index]

    # calculate the distance to the center point
    distances_to_center = np.linalg.norm(points - center_point, axis=1)
    # find the median distance which is the radius of the hemisphere
    median_distance = np.median(distances_to_center)
    dense_hemisphere = points[distances_to_center <= median_distance]

    return dense_hemisphere, center_point


def project_to_plane(points, normal_vector):
    # project points to a plane: projection matrix = I - nn^T
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    projection_matrix = np.eye(3) - np.outer(normal_vector, normal_vector)
    projected_points = points @ projection_matrix.T
    return projected_points

def filter_near_points(points, min_distance):
    distances = np.linalg.norm(points, axis=1)
    filtered_points = points[distances >= min_distance]
    return filtered_points

def segmentation(name):
    pcd = o3d.io.read_point_cloud(f"Dataset/rtabmap_pointCloud/{name}.ply")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    def rgb_to_hsv(rgb):
        return colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])

    hsv_colors = np.apply_along_axis(rgb_to_hsv, 1, colors)

    blue_mask = (hsv_colors[:, 0] >= 0.55) & (hsv_colors[:, 0] <= 0.67) & (hsv_colors[:, 1] > 0.3)
    green_mask = (hsv_colors[:, 0] >= 0.2) & (hsv_colors[:, 0] <= 0.42) & (hsv_colors[:, 1] > 0.4)

    blue_points = points[blue_mask]
    blue_colors = colors[blue_mask]
    blue_pcd = o3d.geometry.PointCloud()
    blue_pcd.points = o3d.utility.Vector3dVector(blue_points)
    blue_pcd.colors = o3d.utility.Vector3dVector(blue_colors)

    green_points = points[green_mask]
    green_colors = colors[green_mask]
    green_pcd = o3d.geometry.PointCloud()
    green_pcd.points = o3d.utility.Vector3dVector(green_points)
    green_pcd.colors = o3d.utility.Vector3dVector(green_colors)

    combined_pcd = blue_pcd + green_pcd
    o3d.io.write_point_cloud(f"Dataset/rtabmap_pointCloud/{name}_segmentation.ply", combined_pcd)
    print(f"Saved {name}_segmentation.ply")
    o3d.visualization.draw_geometries([combined_pcd], window_name="Combined Region")
    return f"{name}_segmentation"

def main(name):
    points = read_ply_file(f"Dataset/rtabmap_pointCloud/{name}_sphere.ply")
    plot_point_cloud(points, 'All Point Clouds')

    dense_hemisphere, center_point = find_dense_hemisphere(points)

    pca = PCA(n_components=3)
    pca.fit(dense_hemisphere)
    normal_vector = pca.components_[-1]

    projected_points = project_to_plane(dense_hemisphere - center_point, normal_vector)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, label='All Points')
    ax.scatter(dense_hemisphere[:, 0], dense_hemisphere[:, 1], dense_hemisphere[:, 2], s=1, color='r',
               label='Dense Hemisphere')

    # projected points
    ax.scatter(projected_points[:, 0] + center_point[0], projected_points[:, 1] + center_point[1],
               projected_points[:, 2] + center_point[2], s=1, color='g', label='Projected Points')

    ax.set_title('3D Point Cloud with Projected Points')
    ax.legend()
    # plt.show()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(projected_points[:, 1], projected_points[:, 2], s=1)
    ax.set_title('Projection of the Most Dense Hemisphere')
    ax.set_xlabel('Projected X')
    ax.set_ylabel('Projected Y')
    plt.savefig(f"Dataset/task4/earth_projection_{name}.png", dpi=300)
    plt.show()

    projection = cv2.imread(f"Dataset/task4/earth_projection_{name}.png")
    projection = cv2.cvtColor(projection, cv2.COLOR_BGR2GRAY)

    # 1. circle center
    center = np.mean(projected_points, axis=0)

    # 2. distance to center
    distances = np.linalg.norm(projected_points - center, axis=1)

    # 3. find boundary points by KNN
    k = 20  # KNN
    projected_points = projected_points[:, 1:3]
    nbrs = NearestNeighbors(n_neighbors=k).fit(projected_points)
    distances, _ = nbrs.kneighbors(projected_points)

    # find boundary points with largest distance
    avg_distances = distances.mean(axis=1)
    if name=="cloud_01_segmentation" or "earth_sphere_only_1":
        precentage = 99
    boundary_points = projected_points[avg_distances >= np.percentile(avg_distances, precentage)]
    print(len(boundary_points))
    diameter_all = []
    # 4. diameter
    max_diameter = 0
    for i in range(len(boundary_points)):
        for j in range(i + 1, len(boundary_points)):
            diameter = np.linalg.norm(boundary_points[i] - boundary_points[j])
            max_diameter = max(max_diameter, diameter)
            diameter_all.append(diameter)
            # print(diameter)
    print(f"max diameter: {max_diameter}")
    print(f"mean diameter: {np.mean(diameter_all)}")

    counts, bins, _ = plt.hist(diameter_all, bins=50, color='blue', alpha=0.7, label='Diameter Distribution')
    max_count_index = np.argmax(counts)
    x_at_highest_y = (bins[max_count_index] + bins[max_count_index + 1]) / 2  # 找到 bin 的中心
    plt.axvline(x_at_highest_y, color='purple', linestyle='--', label=f'Y Max X Coord: {x_at_highest_y:.2f}')
    plt.title("Distribution of Diameters Between Boundary Points")
    plt.xlabel("Diameter Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.savefig(f"Dataset/task4/diameter_distribution_{name}.png", dpi=300)
    plt.show()

    
    plt.figure(figsize=(8, 8))
    plt.scatter(projected_points[:, 0], projected_points[:, 1], alpha=0.5, label="Points")
    plt.scatter(center[0], center[1], color="red", label="Center")
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], color="blue", label="Boundary Points")
    plt.title("Diameter Calculation")
    plt.legend()
    plt.savefig(f"Dataset/task4/diameter_calculation_{name}.png", dpi=300)

    plt.figure(figsize=(8, 5))
    plt.scatter(points[:, 0], points[:, 1], alpha=0.5, label="Points")
    plt.scatter(center[0], center[1], color="red", label="Center")
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], color="blue", label="Boundary Points")
    plt.title("Diameter Calculation")
    plt.legend()
    plt.savefig(f"Dataset/task4/diameter_calculation_3d_{name}.png", dpi=300)


if __name__ == '__main__':
    """
    two options:
    1. collect point clouds for all places, then segment the point cloud -----> use name "cloud_01"
    2. use the segmentated point cloud ------> use name "earth_sphere_only_1"
        the segmentation process is finished by ros2 package image_masker
    """
    
    # 1. if needed, do the segmentation after collecting data
    name_list = ["earth_sphere_only_1", "cloud_01"]
    for name in name_list:
        if name == "cloud_01":
            file_name = segmentation(name)
        else:
            file_name = name
        # 2. use Matlab file fitSphere.mlx for fitting the curve plane
        # 3. find the point cloud projection and diameters in all directions
        main(file_name)