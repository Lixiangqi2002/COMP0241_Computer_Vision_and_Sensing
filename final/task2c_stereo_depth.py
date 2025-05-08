import cv2
import numpy as np
import glob
import os
from task2c_calibration import calibrate_stereo_cameras
import matplotlib.pyplot as plt


# load from the calibration file
def load_calibration_data(calibration_file):
    """
    Load the calibration data for stereo cameras from a file.

    Args:
        calibration_file (str): Path to the calibration data file.

    Returns:
        tuple: Left camera matrix, distortion coefficients, right camera matrix, distortion coefficients, rotation matrix, translation matrix, essential matrix, fundamental matrix.
    """
    with np.load(calibration_file) as data:
        mtxL = data['mtxL']
        distL = data['distL']
        mtxR = data['mtxR']
        distR = data['distR']
        R = data['R']
        T = data['T']
        E = data['E']
        F = data['F']    
    
    # # You can print the parameters to verify if they are loaded successfully
    # print("Left camera matrix:\n", mtxL)
    # print("Right camera matrix:\n", mtxR)

    return mtxL, distL, mtxR, distR, R, T, E, F


def undistort_and_rectify_stereo_images(imgL, imgR, mtxL, distL, mtxR, distR, R, T, h, w):    
    
    # Get the optimal new camera matrix
    newcameramtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (w,h), 1, (w,h))
    newcameramtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (w,h), 1, (w,h))
    
    # Compute the undistortion and rectification transformation map
    mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, None, newcameramtxL, (w,h), cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, None, newcameramtxR, (w,h), cv2.CV_32FC1)
    
    # Apply the transformation map to the images
    rectified_imgL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectified_imgR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
    
    # Crop the images (optional)
    # x, y, w, h = roiL
    # undistorted_imgL = undistorted_imgL[y:y+h, x:x+w]
    # x, y, w, h = roiR
    # undistorted_imgR = undistorted_imgR[y:y+h, x:x+w]
    
    return rectified_imgL, rectified_imgR

def compute_disparity_map(imgL, imgR):
    # 1. Convert to grayscale
    # imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    # imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    print(f"Processed left image size: {imgL.shape}, type: {imgL.dtype}")
    print(f"Processed right image size: {imgR.shape}, type: {imgR.dtype}")

    # print(f"Processed left image size: {imgL_gray.shape}, type: {imgL_gray.dtype}")
    # print(f"Processed right image size: {imgR_gray.shape}, type: {imgR_gray.dtype}")

    # Create StereoSGBM object
    min_disparity = 0
    num_disparities = 16 * 5  # Must be a multiple of 16
    block_size = 9
    stereo = cv2.StereoSGBM_create(minDisparity=min_disparity,
                                   numDisparities=num_disparities,
                                   blockSize=block_size,
                                   P1=8 * 3 * block_size ** 2,
                                   P2=32 * 3 * block_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32)
    # 4. Compute disparity
    try:
        disparity = stereo.compute(imgL, imgR)
        return disparity.astype(np.float32) / 16.0
    except cv2.error as e:
        print(f"Disparity computation error: {e}")
        print(f"Input image information:")
        # print(f"Left grayscale image: {imgL_gray.shape}, {imgL_gray.dtype}")
        # print(f"Right grayscale image: {imgR_gray.shape}, {imgR_gray.dtype}")
        raise

def compute_depth_map(disparity, focal_length, baseline):
    depth = np.zeros(disparity.shape, dtype=np.float32)
    depth[disparity > 0] = (focal_length * baseline) / disparity[disparity > 0]
    # depth_map = cv2.reprojectImageTo3D(disparity, Q)

    return depth


if __name__ == "__main__":


    mtxL, distL, mtxR, distR, R, T, E, F = load_calibration_data("stereo_calib.npz")

    # load images
    index=6
    imgL = cv2.imread(f"Dataset/task2/2-c/2-c_stereo/calibration_image/left_{index}.jpg")
    imgR = cv2.imread(f"Dataset/task2/2-c/2-c_stereo/calibration_image/right_{index}.jpg")
    h, w = imgL.shape[:2]
    
    # undistort images
    rectified_imgL, rectified_imgR= undistort_and_rectify_stereo_images(imgL, imgR, mtxL, distL, mtxR, distR, R, T, h, w)

    # 2. Convert to grayscale
    grayL = cv2.cvtColor(rectified_imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectified_imgR, cv2.COLOR_BGR2GRAY)

    # 3. Compute disparity
    disparity = compute_disparity_map(rectified_imgL, rectified_imgR)

    # 4. Compute depth
    focal_length = mtxL[0, 0]  # Assume the focal length is the same for both cameras
    baseline = np.linalg.norm(T)  # Baseline length
    depth = compute_depth_map(disparity, focal_length, baseline)

    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].imshow(cv2.cvtColor(rectified_imgL, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Left Image")
    axes[0, 0].axis('off')
    axes[0, 1].imshow(cv2.cvtColor(rectified_imgR, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Right Image")
    axes[0, 1].axis('off')
    axes[1, 0].imshow((disparity / np.max(disparity) * 255).astype(np.uint8), cmap='gray')
    axes[1, 0].set_title("Disparity Map")
    axes[1, 0].axis('off')
    axes[1, 1].imshow((depth / np.max(depth) * 255).astype(np.uint8), cmap='plasma')
    axes[1, 1].set_title("Depth Map")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    # cv2.imshow("Disparity", (disparity / np.max(disparity) * 255).astype(np.uint8))
    # cv2.imshow("Depth", (depth / np.max(depth) * 255).astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
