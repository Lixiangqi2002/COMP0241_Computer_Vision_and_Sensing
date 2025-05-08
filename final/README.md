# COMP0241 Coursework Group 10 

## Task 1: Extract AO (Segmentation)

### Subtasks

**1. Task 1.a: Individual Segmentation Methods**

This task evaluates three individual segmentation methods:
* **K-means clustering**:
   - The algorithm clusters image pixels into different groups to identify regions of interest.
* **Ellipse fitting**:
   - Uses contour detection to fit ellipses around AO regions.
* **Color thresholding**:
   - Segments images based on specific HSV color ranges.

**2.  Task 1.b: Combined Segmentation Methods**

This task combines individual methods to improve accuracy:
* **Color thresholding + Ellipse fitting**.
* **K-means clustering + Ellipse fitting**.


**3.  Task 1.c: Performance Evaluation**

The script evaluates the segmentation methods using **Receiver Operating Characteristic (ROC)** curves and calculates the False Positive Rate (FPR) and True Positive Rate (TPR).

### Run Code

Run code `task1.py`, with specified parameter `Dataset2` and `extract_ellipse`.

#### Parameters

- **`Dataset2`**:
  - Set to `True` to use the custom dataset (`neo_images` and `neo_masks`) collected using the Arducam camera.
  - Set to `False` to use the original dataset provided in the task.

- **`extract_ellipse`**:
  - Set to `True` to enable ellipse fitting as part of the segmentation process.
  - Set to `False` to skip ellipse fitting and focus on direct segmentation methods.
  - If `extract_ellipse` is set to `False`, the program plots the ROC curve to visualize the performance of different segmentation methods.

---

## Task 2: Estimate Dynanmics of AO 

### Subtasks 2.a and 2.b: Trajectory Analysis and Pendulum Motion Fitting

#### Task 2.a: Trajectory Analysis
1. **Centroid Detection**:
   - Detects centroids of the target object in each frame using:
     - **K-means clustering**.
     - **Ellipse fitting** for filtering non-elliptical outliers.
   - Outputs detected centroids as `(x, y)` coordinates for each frame.

2. **Trajectory Visualization**:
   - Plots the trajectory of centroids across frames.
   - Generates bounding boxes around detected areas and visualizes trajectory evolution over time.

3. **Outlier Detection**:
   - Removes size outliers using contour area statistics.
   - Filters non-elliptical contours using aspect ratio thresholds.

#### Task 2.b: Pendulum Motion Fitting
1. **Principal Component Analysis (PCA)**:
   - Performs PCA to identify the principal motion direction of the object.
   - Rotates the data into principal axes for simplified motion analysis.

2. **Pendulum Model Fitting**:
   - Fits a pendulum motion model to the trajectory data:
     \[
     y(t) = A \cdot \cos(\omega \cdot t + \phi) + C
     \]
   - Uses Fast Fourier Transform (FFT) to estimate initial angular velocity (\( \omega \)).

3. **Smoothed Motion Trajectory**:
   - Outputs a smoothed trajectory overlaid with the original motion.

Run code `task2ab.py`.

#### Workflow

1. **Video Preprocessing**:
   - Clips videos to the desired duration.
   - Extracts frames at specified intervals.

2. **Centroid Detection**:
   - Detects and records centroids for each frame.
   - Applies outlier filtering based on contour size and shape.

3. **Trajectory Analysis**:
   - Visualizes centroid trajectories and analyzes motion patterns:
     - Average position.
     - Swing range (\( x, y \)).
     - Standard deviation.

4. **Pendulum Fitting**:
   - Fits the motion data to the pendulum model.
   - Outputs fitted parameters, including amplitude (\( A \)), angular velocity (\( \omega \)), and phase (\( \phi \)).


#### Parameters

- **Dataset**:
  - Images are loaded from `Dataset/task2/2-b/img/{name}`.
  - Camera calibration matrix and distortion coefficients are applied to undistort the images.

- **Threshold**:
  - K-means clustering threshold is dataset-specific, which is from task 1 result (tuned in task 1):
    - **`1`**: \( 49.8 \)
    - **`2`**: \( 32.6 \)
    - **`3`**: \( 25.0 \)
    - **`5`**: \( 39.8 \)


### Subtasks 2.c: Estimate Height 

#### From stereo depth
 1. Stereo Camera Calibration
- Loads pre-calibrated stereo camera parameters from a `.npz` file:
  - **Camera matrices** (`mtxL`, `mtxR`).
  - **Distortion coefficients** (`distL`, `distR`).
  - **Extrinsic parameters** (`R`, `T`, `E`, `F`).

 2. Image Rectification
- Undistorts and rectifies stereo images using calibration parameters to align the image planes.

 3. Disparity Map Computation
- Computes the disparity map using **StereoSGBM** (Semi-Global Block Matching):
  - Parameters:
    - Minimum disparity: \(0\).
    - Number of disparities: Multiple of 16 (default: \(80\)).
    - Block size: \(9\).
    - Tuning parameters: \(P1, P2, uniquenessRatio, speckleWindowSize\).
    
4. Depth Map Computation
- Converts the disparity map to a depth map using the stereo camera baseline and focal length:
  \[
  \text{Depth}(x, y) = \frac{\text{focal length} \times \text{baseline}}{\text{disparity}(x, y)}
  \]

5. Workflow
Run `task2c_capture.py` for capture the pair images with baseline for stereo vision disparity estimation.
Run `task2c_calibration.py` for undistortion and rectification of the captured images.
Run `task2c_stereo_depth.py` for estimating the depth of the AO in the calculatedd depth image from pair images.
- **Calibration Data**:
  - File: `stereo_calib.npz`.
  - Contains intrinsic and extrinsic stereo camera parameters.

- **Stereo Images**:
  - Input: Stereo images from `2-c_stereo/calibration_image/`.
  - Example files: `left_6.jpg`, `right_6.jpg`.

- **StereoSGBM Settings**:
  - Minimum disparity: \(0\).
  - Number of disparities: \(16 \times 5 = 80\).
  - Block size: \(9\).
  - Additional tuning parameters (`P1`, `P2`, etc.).

#### From point cloud

The RGB and depth images are extracted from the `cloud_02.db`, genenrated from the RTAB-MAP.
1. Load and Resize Images
- **Input**:
  - RGB image: `Dataset/task2/2-c/2-c_pointCloud/RGB_db.jpg`
  - Depth image: `Dataset/task2/2-c/2-c_pointCloud/Depth_db.jpg`
- Both images are resized to \(640 \times 400\) for uniformity.

 2. Depth Image Histogram
- Compute and display a histogram of pixel intensities in the depth image.
- **Histogram Details**:
  - X-axis: Pixel intensity values.
  - Y-axis: Frequency of each intensity.

 3. Region of Interest (ROI) Extraction
- The ROI is extracted using **K-means clustering** and **ellipse fitting**:
  1. **K-means clustering**: Segments the RGB image into clusters to identify regions of interest.
  2. **Ellipse fitting**: Refines the segmentation by fitting an ellipse around the ROI.
- The final ROI mask is converted to grayscale for further processing.

 4. Masked Depth Image
- Apply the ROI mask to the depth image:
  - Pixels outside the ROI are excluded.
  - Retains depth values corresponding to the segmented region.

 5. Depth Distribution Analysis
- Compute a histogram of depth values in the ROI.
- Identify the pixel intensity value with the highest frequency (mode) in the ROI.

Run `task2c_point_cloud_segmentation_height.py`.
- **Segmentation Threshold**:
  - \( \text{threshold} = 100 \): Used in K-means clustering for segmentation.


---

## Task 3-1: Average Periods Estimation

### Subtasks

**1. Feature Matching and Period Detection**

This task identifies and tracks features across consecutive video frames to detect rotation periods. By analyzing feature matching patterns, the average distance between matches is calculated, and rotation periods are estimated.

*Feature Matching*:
   - Detects and matches keypoints between frames using SIFT (Scale-Invariant Feature Transform) features.
   - Matches are refined using BFMatcher with `L2` norm.

*Period Detection*:
   - Periods are calculated based on the average feature matching distances between frames and a specified threshold.

**2. 3D Sphere Reconstruction**

This task reconstructs the motion of a 3D sphere by projecting keypoints onto its surface. Key functionalities include:

*3D Sphere Visualization*:
   - Visualizes the 3D sphere and tracked keypoints across frames.
*3D Point Transformation*:
   - Converts 2D keypoints to 3D points using the sphere's center, radius, and camera parameters.

**3. Performance Analysis**

The script evaluates the feature matching results and detects angular velocity and rotation periods. It visualizes the following:

*Average Feature Matching Distance*:
   - Plots the average distances between features across frames.
*Detected Periods*:
   - Highlights detected periods as vertical lines on the average distance plot.
### Run Code
Run code `task3.py`, with specified parameter `real_time` and `video_id`.

- **`real_time`**:
  - Set to `True` to enable real-time estimation of periods.
  - Set to `False` to process pre-recorded videos.

- **`video_id`**:
  - Set to `1`, `2`, or `3` to process one of the three pre-recorded videos.
  - Ignored when `real_time` is set to `True`.

---

## Task 4-1: AO Sphere Diameter Estimation

### Subtasks

1. Point Cloud Segmentation

This task segments point clouds into regions of interest using HSV-based color filtering:
* **Blue Region Detection**:
   - Detects regions within the blue HSV range to identify specific parts of the point cloud.
* **Green Region Detection**:
   - Identifies regions within the green HSV range.
* **Combined Segmentation**:
   - Combines both blue and green regions into a unified segmented point cloud.

2. Curve Surface fitting

Use the `fit_curve.m` or `fitSphere.mlx` for fitting the hemisphere as a curve surface.

3.  Dense Hemisphere Extraction

This task identifies the densest region of the point cloud:
* **Center Point Calculation**:
   - Finds the point closest to all others as the center of the dense region.
* **Hemisphere Filtering**:
   - Extracts points within a median radius of the center.

4.  Projection and Diameter Calculation

Projects the dense hemisphere onto a plane and calculates the sphere’s diameter:
* **Projection**:
   - Projects the dense hemisphere onto a plane using PCA to simplify 3D analysis.
* **Diameter Estimation**:
   - Estimates the maximum diameter by analyzing distances between boundary points.

5.  Visualization

This task visualizes the results in both 2D and 3D:
* **3D Visualization**:
   - Visualizes the entire point cloud, dense hemisphere, and projections.
* **2D Visualization**:
   - Displays the projection plane and boundary points for diameter calculation.


### Run Code

Run code `task4.py` with the specified `.ply` dataset.

- **`name`**:
  - Use `"cloud_01"` to process the raw collected point cloud and perform segmentation.
  - Use `"earth_sphere_only_1"` to process segmented point clouds.

---

## Task 3-2 and Task 4-2: Real Time Estimation of Angular Velocity, Periods, Latitude, and Linear Velocity


### Subtasks

#### Task 3-2: Real-Time Period and Angular Velocity Estimation

1. **Angular Velocity Calculation**:
   - Tracks feature points across consecutive frames to compute 3D displacement vectors.
   - Uses the displacement vectors and sphere geometry to estimate the angular velocity.

2. **Period Estimation**:
   - Derives the rotation period using the angular velocity and tracks its variations over time.

3. **Displacement Analysis**:
   - Matches feature points between frames and analyzes their average displacement for validation.

#### Task 4-2: Latitude and Linear Velocity Estimation

1. **Latitude Calculation**:
   - Computes the latitude for selected points on the sphere surface based on their displacement from the rotation axis.

2. **Linear Velocity Estimation**:
   - Derives the linear velocity of points based on angular velocity and latitude.

3. **Visualization**:
   - Plots real-time variations in angular velocity, rotation periods, and linear velocity for selected points.



### Run Code

Run code `task3_task4_real_time_velocity.py`.

#### Parameters

- **`real_time`**:
  - Set to `True` for real-time processing of the video stream.
  - Set to `False` for offline processing of pre-recorded videos.

- **`video_id`**:
  - `1`, `2`, `3`: Pre-recorded AO rotation videos with full rotation cycle.
  - `4`: This video does not contain a full rotation cycle, only used for task 4-2.


### Workflow

1. **Initialize Video**:
   - Load the specified video using the `video_id` parameter.
   - Preprocess the first frame to define the sphere center and radius.

2. **Feature Tracking**:
   - Use SIFT (Scale-Invariant Feature Transform) to extract and match feature points across frames.
   - Compute 3D displacement vectors for matched points.

3. **Angular Velocity and Period Calculation**:
   - Estimate the angular velocity and derive rotation periods.
   - Track the variation in angular velocity over time.

4. **Latitude and Linear Velocity Estimation**:
   - Compute latitude for selected points on the sphere.
   - Estimate linear velocity for each point based on angular velocity and latitude.

5. **Visualization**:
   - Generate and save the following plots:
     - Angular velocity vs. time.
     - Rotation period vs. time.
     - Linear velocity vs. latitude for selected points.



### Notes

1. **Mouse Selection**:
   - During runtime, the user selects:
     1. A point on the rotation axis (e.g., top or bottom point of the sphere).
     2. Points of interest on the sphere for velocity tracking.