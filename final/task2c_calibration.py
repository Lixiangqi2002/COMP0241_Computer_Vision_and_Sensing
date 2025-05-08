import cv2
import numpy as np
import glob
import os

def calibrate_stereo_cameras(chessboard_size=(7, 4), square_size=34.0, left_image_pattern='Dataset/task2/2-c/2-c_stereo/calibration_image/left_*.jpg', right_image_pattern='Dataset/task2/2-c/2-c_stereo/calibration_image/right_*.jpg', output_dir="output_chessboard_corners"):
    """
    Calibrate stereo cameras.

    Args:
        chessboard_size (tuple): Size of the chessboard.
        square_size (float): Size of each square on the chessboard.
        left_image_pattern (str): File pattern for left camera images.
        right_image_pattern (str): File pattern for right camera images.
        output_dir (str): Directory to save images with drawn corners.
    
    Returns:
        tuple: Left camera intrinsic parameters, distortion coefficients, right camera intrinsic parameters, distortion coefficients, rotation matrix, translation matrix, essential matrix, fundamental matrix.
    """
    # Prepare 3D points of the chessboard
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Lists to store 3D points and 2D points for all images
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    # Read left and right camera images
    images_left = glob.glob(left_image_pattern)
    images_right = glob.glob(right_image_pattern)

    print(f"Found {len(images_left)} left camera images and {len(images_right)} right camera images")

    # Output directory (to save images with drawn corners)
    os.makedirs(output_dir, exist_ok=True)

    for img_left, img_right in zip(images_left, images_right):
        imgL = cv2.imread(img_left)
        imgR = cv2.imread(img_right)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)

        if retL and retR:
            objpoints.append(objp)
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)

            # Draw corners
            cv2.drawChessboardCorners(imgL, chessboard_size, cornersL, retL)
            cv2.drawChessboardCorners(imgR, chessboard_size, cornersR, retR)

            # Display images
            cv2.imshow("Left Camera Chessboard Corners", imgL)
            cv2.imshow("Right Camera Chessboard Corners", imgR)
            cv2.waitKey(500)  # Wait for 1 second

            # Save result images
            output_path = os.path.join(output_dir, f"left_{os.path.basename(img_left)}")
            cv2.imwrite(output_path, imgL)
            output_path = os.path.join(output_dir, f"right_{os.path.basename(img_right)}")
            cv2.imwrite(output_path, imgR)
        else:
            print(f"Failed to find corners: {retL}{retR}")

    cv2.destroyAllWindows()

    # Calibrate single cameras
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

    # Stereo camera calibration
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, mtxL, distL, mtxR, distR, grayL.shape[::-1], criteria=criteria, flags=flags)

    print(f"Left camera intrinsic parameters:\n{mtxL}")
    print(f"Left camera distortion coefficients:\n{distL}")
    print(f"Right camera intrinsic parameters:\n{mtxR}")
    print(f"Right camera distortion coefficients:\n{distR}")
    print(f"Rotation matrix:\n{R}")
    print(f"Translation matrix:\n{T}")
    print(f"Essential matrix:\n{E}")
    print(f"Fundamental matrix:\n{F}")

    # Save calibration results
    np.savez('stereo_calib.npz', mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T, E=E, F=F)
    print(f"Calibration results saved to stereo_calib.npz")

    return mtxL, distL, mtxR, distR, R, T, E, F

if __name__ == "__main__":
    calibrate_stereo_cameras()
