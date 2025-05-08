# use given dataset images and masks
import math

import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.metrics import auc
import matplotlib.pyplot as plt

from aux_functions import remove_noise_and_display


def extract_AO_Color_Threshold(original_img=None, threshold=50):
    img = original_img.copy()

    # Convert BGR image to HSV
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for blue and green
    # Blue range (adjust these values to fit your application)
    lower_blue_1 = np.array([90, threshold, threshold])  # Hue: 100-140, Saturation & Value thresholds
    upper_blue_1 = np.array([140, 255, 255])

    lower_green_1 = np.array([35, threshold, threshold])  # Hue: 40-80, Saturation & Value thresholds
    upper_green_1 = np.array([150, 255, 255])

    # Create masks for blue and green
    blue_mask_1 = cv2.inRange(hsv_image, lower_blue_1, upper_blue_1)
    green_mask_1 = cv2.inRange(hsv_image, lower_green_1, upper_green_1)

    # Apply dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blue_mask_1 = cv2.morphologyEx(blue_mask_1, cv2.MORPH_DILATE, kernel)
    green_mask_1 = cv2.morphologyEx(green_mask_1, cv2.MORPH_DILATE, kernel)

    # Combine masks (extract areas of blue and green)
    combined_mask = cv2.bitwise_or(blue_mask_1, green_mask_1)

    predict_mask_color_thresholding = cv2.bitwise_and(img, img, mask=combined_mask)

    cv2.imshow("Color Thresholding only Combined Mask", combined_mask)
    # cv2.imshow("Result", predict_mask_color_thresholding)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()

    return predict_mask_color_thresholding


def extract_AO_Ellipse(original_img=None, result=None, canny_thresholds_1=50, canny_thresholds_2=150, plot=False):
    img = original_img.copy()
    gray_img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # gray_img = cv2.GaussianBlur(gray_img, (5,5), 1.0)
    # predict_mask_color = remove_noise_and_display(gray_img, min_area=5000)

    predict_mask_color = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)[1]
    predict_mask_color = cv2.Canny(predict_mask_color, threshold1=canny_thresholds_1, threshold2=canny_thresholds_2)

    # cv2.imshow("Before dilation", predict_mask_color)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    predict_mask_color = cv2.morphologyEx(predict_mask_color, cv2.MORPH_DILATE, kernel)
    # cv2.imshow("Predict Mask GRAY", predict_mask_color)
    # Find contours
    contours, _ = cv2.findContours(predict_mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ellipses = []
    for contour in contours:
        e = 0.01 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, e, True)
        if len(contour) >= 5:
            # 1. Detect ellipse
            ellipse = cv2.fitEllipse(contour)
            # Ellipse center coordinates; length of the major and minor axes (diameter); ellipse rotation angle
            center, axes, orientation = ellipse
            # 2. Check if the ellipse center is inside the image
            if 0 <= center[0] <= predict_mask_color.shape[0] and 0 <= center[1] <= predict_mask_color.shape[1]:
                ellipses.append(ellipse)
                # cv2.ellipse(img, ellipse, (0, 255, 255), 2)  # Draw yellow ellipse
        else:
            print("Contour has fewer than 5 points, cannot fit an ellipse.")
            return

    # 3. Filter ellipses
    # Sort ellipses by area and find the largest one
    if len(ellipses) == 0:
        print("NO ELLIPSE FOUND")
        return False
    # else:
    # print("ELLIPSE FOUND")
    sorted_ellipses = sorted(ellipses, key=lambda item: item[1][0] * item[1][1], reverse=True)
    cv2.ellipse(img, sorted_ellipses[0], (0, 255, 255), 2)  # Draw yellow ellipse

    # Create a black mask with the same size as the original image
    predict_mask_ellipse = np.zeros_like(img, dtype=np.uint8)
    cv2.ellipse(predict_mask_ellipse, sorted_ellipses[0], (255, 255, 255), -1)

    cv2.imshow("Fit Ellipse only Detected Circles", img)
    # cv2.imshow("Result", processed_mask)
    # cv2.imshow("Result", processed_mask)
    # cv2.imshow("Predict Mask Ellipse", predict_mask_ellipse)
    cv2.waitKey(1)
    return predict_mask_ellipse


def extract_AO_Kmeans(original_img=None, threshold=50):
    # Convert image from BGR to RGB (OpenCV loads images as BGR by default)
    img = original_img.copy()

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape image into a 2D array, where rows are pixels and columns are RGB channels
    pixels = image_rgb.reshape(-1, 3)

    # Set K value (number of clusters)
    k = 75
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    # Get labels for each pixel
    labels = kmeans.labels_
    # Get the RGB values of cluster centers
    cluster_centers_rgb = kmeans.cluster_centers_
    # Convert RGB colors to HLS
    cluster_centers_hls = cv2.cvtColor(np.uint8([cluster_centers_rgb]), cv2.COLOR_RGB2HLS)[0]

    # Visualize cluster centers
    # plt.figure(figsize=(8, 4))
    # for i, center in enumerate(cluster_centers_rgb):
    #     plt.subplot(1, k, i + 1)
    #     plt.imshow(np.ones((10, 10, 3), dtype=np.uint8) * center.astype(np.uint8))
    #     plt.axis('off')
    #     plt.title(f'Cluster {i} Center')
    # print("Sorted RGB Centers:", cluster_centers_rgb)
    # print("Sorted HSV Centers:", cluster_centers_hsv)

    # Select cluster labels based on hue > threshold
    selected_labels = []
    segmented_image = np.zeros_like(image_rgb)
    for i in range(len(cluster_centers_hls)):
        if 160 > cluster_centers_hls[i][0] > threshold and 100 < cluster_centers_hls[i][1] < 220 and 5 < cluster_centers_hls[i][2]:  # Select blue regions
            selected_labels.append(i)
            # Traverse labels and retain selected pixel labels
            segmented_image[labels.reshape(image_rgb.shape[0], image_rgb.shape[1]) == i] = image_rgb[
                labels.reshape(image_rgb.shape[0], image_rgb.shape[1]) == i]


    # Visualize selected blue regions
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    segmented_image = cv2.medianBlur(segmented_image, 3)
    # cv2.imshow("Median Blur", predict_mask_color)
    # Create elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    predict_mask_color = cv2.morphologyEx(segmented_image, cv2.MORPH_DILATE, kernel)

    # cv2.imshow("DILATE", predict_mask_color)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    # Perform morphological closing (fill gaps)
    processed_mask = cv2.morphologyEx(predict_mask_color, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("CLOSE", processed_mask)
    # Perform morphological opening (remove small regions)
    segmented_image = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("OPEN", predict_mask_color)
    # segmented_image = cv2.cvtColor(predict_mask_color, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Predict Mask", predict_mask_color)
    cv2.imshow("Kmeans only Segmented Regions", segmented_image)
    cv2.waitKey(1)
    return segmented_image


def extract_AO_color_ellipse(original_img=None, predict_mask_color=None):
    img = original_img.copy()

    predict_mask_color = remove_noise_and_display(predict_mask_color, min_area=5000)
    # cv2.imshow("Before dilation", predict_mask_color)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    predict_mask_color = cv2.morphologyEx(predict_mask_color, cv2.MORPH_DILATE, kernel)

    # cv2.imshow("Predict Mask GRAY", predict_mask_color)
    # Find contours
    contours, _ = cv2.findContours(predict_mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ellipses = []
    for contour in contours:
        e = 0.01 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, e, True)
        if len(contour) >= 5:
            # 1. Detect ellipse
            ellipse = cv2.fitEllipse(contour)
            # Ellipse center coordinates; length of the major and minor axes (diameter); ellipse rotation angle
            center, axes, orientation = ellipse
            # 2. Check if the ellipse center is inside the image
            if 0 <= center[0] <= predict_mask_color.shape[0] and 0 <= center[1] <= predict_mask_color.shape[1]:
                ellipses.append(ellipse)
                # cv2.ellipse(img, ellipse, (0, 255, 255), 2)  # Draw yellow ellipse
        else:
            print("Contour has fewer than 5 points, cannot fit an ellipse.")
            return

    # 3. Filter ellipses
    # Sort ellipses by area and find the largest one
    if len(ellipses) == 0:
        print("NO ELLIPSE FOUND")
        return False
    # else:
    # print("ELLIPSE FOUND")
    sorted_ellipses = sorted(ellipses, key=lambda item: item[1][0] * item[1][1], reverse=True)
    cv2.ellipse(img, sorted_ellipses[0], (0, 255, 255), 2)  # Draw yellow ellipse

    # Create a black mask with the same size as the original image
    predict_mask_ellipse = np.zeros_like(img, dtype=np.uint8)
    cv2.ellipse(predict_mask_ellipse, sorted_ellipses[0], (255, 255, 255), -1)
    # cv2.imwrite(f'Dataset/task1/color_ellipse_neo/{name}_1.b_predicted_color.png', img)
    # cv2.imwrite(f'Dataset/task1/color_ellipse/{name}_1.b_predicted_color.png', img)

    cv2.imshow("Detected Circles Color Thresholding + Fit Ellipse", img)
    # cv2.imshow("Result", processed_mask)
    # cv2.imshow("Predict Mask Ellipse", predict_mask_ellipse)
    cv2.waitKey(1)
    return predict_mask_ellipse


def extract_AO_k_means_ellipse(original_img=None, predict_mask_color=None, task3=False):
    img = original_img.copy()

    predict_mask_color = remove_noise_and_display(predict_mask_color, min_area=5000)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    predict_mask_color = cv2.morphologyEx(predict_mask_color, cv2.MORPH_DILATE, kernel)

    # cv2.imshow("Predict Mask GRAY", predict_mask_color)
    # Find contours
    contours, _ = cv2.findContours(predict_mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ellipses = []
    for contour in contours:
        e = 0.01 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, e, True)
        if len(contour) >= 5:
            # 1. Detect ellipse
            ellipse = cv2.fitEllipse(contour)
            # Ellipse center coordinates; length of the major and minor axes (diameter); ellipse rotation angle
            center, axes, orientation = ellipse
            # 2. Check if the ellipse center is inside the image
            if 0 <= center[0] <= predict_mask_color.shape[0] and 0 <= center[1] <= predict_mask_color.shape[1]:
                ellipses.append(ellipse)
                # cv2.ellipse(img, ellipse, (0, 255, 255), 2)  # Draw yellow ellipse
        else:
            print("Contour has fewer than 5 points, cannot fit an ellipse.")
            return

    # 3. Filter ellipses
    # Sort ellipses by area and find the largest one
    if len(ellipses) == 0:
        print("NO ELLIPSE FOUND")
        return False
    # else:
        # print("ELLIPSE FOUND")
    sorted_ellipses = sorted(ellipses, key=lambda item: item[1][0] * item[1][1], reverse=True)
    cv2.ellipse(img, sorted_ellipses[0], (0, 255, 255), 2)  # Draw yellow ellipse

    # Create a black mask with the same size as the original image
    predict_mask_ellipse = np.zeros_like(img, dtype=np.uint8)
    cv2.ellipse(predict_mask_ellipse, sorted_ellipses[0], (255, 255, 255), -1)
    # elif plot=="1.b_kmeans":
    # cv2.imwrite(f'Dataset/task1/kmeans_ellipse_neo/{name}_1.b_predicted_kmeans.png', img)
    # cv2.imwrite(f'Dataset/task1/kmeans_ellipse/{name}_1.b_predicted_kmeans.png', img)

    cv2.imshow("Detected Circles Kmeans + Fitting Ellipse", img)
    # cv2.imshow("Result", processed_mask)
    # cv2.imshow("Predict Mask Ellipse", predict_mask_ellipse)
    cv2.waitKey(1)
    if task3:
        return predict_mask_ellipse, sorted_ellipses[0]
    else:
        return predict_mask_ellipse

def compute_fp_tp(mask_img, predict_mask=None):
    if mask_img is None or predict_mask is None:
        # print("One of the input masks is None.")
        return 0, 0, float('inf')

    if mask_img.size == 0 or predict_mask.size == 0:
        # print("One of the input masks is empty.")
        return 0, 0, float('inf')

    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, mask_img_binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)  # Binary threshold

    predict_mask = cv2.cvtColor(predict_mask, cv2.COLOR_BGR2GRAY)

    # Normalize values to 0 and 1
    mask_img_binary = (mask_img_binary.flatten() > 0).astype(int)
    predict_mask = (predict_mask.flatten() > 0).astype(int)

    tp = np.sum((predict_mask == 1) & (mask_img_binary == 1))  # Predicted as 1 and actually 1
    fp = np.sum((predict_mask == 1) & (mask_img_binary == 0))  # Predicted as 1 but actually 0
    fn = np.sum((predict_mask == 0) & (mask_img_binary == 1))
    tn = np.sum((predict_mask == 0) & (mask_img_binary == 0))

    fp = fp / (fp + tn)
    tp = tp / (fn + tp)

    ratio = math.sqrt((fp-0)**2 + (tp-1)**2)
    print(ratio)
    return fp, tp, ratio

def read_values(file_path):
    with open(file_path, "r") as file:
        values = file.readlines()
        values = [float(value.strip()) for value in values]  # 转换为浮点数
    return values

def plot_roc(file_paths):
    fpr_k_means = read_values(file_paths["k_means"])
    tpr_k_means = read_values(file_paths["k_means_tpr"])
    fpr_cv_ellipse = read_values(file_paths["cv_ellipse"])
    tpr_cv_ellipse = read_values(file_paths["cv_ellipse_tpr"])
    fpr_color_thresholding = read_values(file_paths["color_thresholding"])
    tpr_color_thresholding = read_values(file_paths["color_thresholding_tpr"])
    fpr_all_color = read_values(file_paths["all_color"])
    tpr_all_color = read_values(file_paths["all_color_tpr"])
    fpr_all_kmeans = read_values(file_paths["all_kmeans"])
    tpr_all_kmeans = read_values(file_paths["all_kmeans_tpr"])

    plt.figure()
    plt.scatter(fpr_k_means, tpr_k_means, label="K-Means Method", color="blue", s=2)
    plt.scatter(fpr_cv_ellipse, tpr_cv_ellipse, label="Ellipse Method", color="green", s=2)
    plt.scatter(fpr_color_thresholding, tpr_color_thresholding, label="Color Thresholding Method", color="gold", s=2)
    plt.scatter(fpr_all_color, tpr_all_color, label="Color + Ellipse Method", color="red", s=2)
    plt.scatter(fpr_all_kmeans, tpr_all_kmeans, label="K-Means + Ellipse Method", color="purple", s=2)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    # plt.savefig(f'Dataset/task1/roc.png', format='png', dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    """
    Task 1: Segmentation of the Aortic Valve
    
    Args:
        Dataset2: True if using the new dataset, False if using the original dataset
        extract_ellipse: True if extracting the ellipse, False if not extracting the ellipse
    """
    Dataset2 = True
    extract_ellipse = True

    if Dataset2==True:
        base_path = "Dataset/task1/fpr_tpr_files_neo" 
    else:
        base_path = "Dataset/task1/fpr_tpr_files"
    file_paths = {
        "k_means": os.path.join(base_path, "k_means.txt"),
        "cv_ellipse": os.path.join(base_path, "cv_ellipse.txt"),
        "color_thresholding": os.path.join(base_path, "color_thresholding.txt"),
        "all_color": os.path.join(base_path, "all_color.txt"),
        "all_kmeans": os.path.join(base_path, "all_kmeans.txt"),
        "k_means_tpr": os.path.join(base_path, "k_means_tpr.txt"),
        "cv_ellipse_tpr": os.path.join(base_path, "cv_ellipse_tpr.txt"),
        "color_thresholding_tpr": os.path.join(base_path, "color_thresholding_tpr.txt"),
        "all_color_tpr": os.path.join(base_path, "all_color_tpr.txt"),
        "all_kmeans_tpr": os.path.join(base_path, "all_kmeans_tpr.txt"),
        "best_threshold_kmeans": os.path.join(base_path, "best_threshold_kmeans.txt"),
        "best_threshold_cv_ellipse": os.path.join(base_path, "best_threshold_cv_ellipse.txt"),
        "best_threshold_color_thresholding": os.path.join(base_path, "best_threshold_color_thresholding.txt"),
    }

    for path in file_paths.values():
        if not os.path.exists(path):  # 检查文件是否存在
            with open(path, "w") as file:
                file.write("")  # 创建空文件

    best_fpr_k_means_arr, best_tpr_k_means_arr = [], []
    best_fpr_cv_ellipse_arr, best_tpr_cv_ellipse_arr = [], []
    best_fpr_color_thresholding_arr, best_tpr_color_thresholding_arr = [], []
    best_fpr_all_color_arr, best_tpr_all_color_arr = [], []
    best_fpr_all_kmeans_arr, best_tpr_all_kmeans_arr = [], []


    if extract_ellipse:
        for name_i in range(0, 10):
            for name_j in range(0, 10):
                if Dataset2==True:
                    name = "000" + str(name_i) + str(name_j)
                    print(name)
                    original_img = cv2.imread(f'Dataset/task1/neo_images/{name}.jpg')
                    mask_img = cv2.imread(f'Dataset/task1/neo_masks/{name}.png')
                else:
                    name = "0000" + str(name_i) + str(name_j)
                    print(name)
                    original_img = cv2.imread(f'Dataset/task1/images/{name}.png')
                    mask_img = cv2.imread(f'Dataset/task1/masks/{name}.png')
                    original_img = cv2.resize(original_img, (640, 400))
                    mask_img = cv2.resize(mask_img, (640, 400))

                ########################################################### Task 1.a ###################################################################
                # 1. K-means
                fpr_k_means_arr = []
                tpr_k_means_arr = []
                ratio_k_means_best = 100
                threshold_kmeans_best = 0
                threshold_kmeans = np.arange(0, 100, 5)
                for threshhold_k in threshold_kmeans:
                    predicted_kmeans = extract_AO_Kmeans(original_img, threshhold_k)
                    fpr_k_means, tpr_k_means, ratio = compute_fp_tp(mask_img, predicted_kmeans)
                    if ratio < ratio_k_means_best:
                        ratio_k_means_best = ratio
                        best_fpr_k_means = fpr_k_means
                        best_tpr_k_means = tpr_k_means
                        threshold_kmeans_best = threshhold_k
                    fpr_k_means_arr.append(fpr_k_means)
                    tpr_k_means_arr.append(tpr_k_means)
                # save best segmentation
                predicted_kmeans = extract_AO_Kmeans(original_img, threshold_kmeans_best)
                # cv2.imwrite(f'Dataset/task1/kmeans/{name}_1.a_predicted_kmeans.png', predicted_kmeans)
                # cv2.imwrite(f'Dataset/task1/kmeans_neo/{name}_1.a_predicted_kmeans.png', predicted_kmeans)
                with open(file_paths["k_means"], "a") as file:
                    file.write(f"{best_fpr_k_means}\n")
                with open(file_paths["k_means_tpr"], "a") as file:
                    file.write(f"{best_tpr_k_means}\n")
                with open(file_paths["best_threshold_kmeans"], "a") as file:
                    file.write(f"{threshold_kmeans_best}\n")

                # 2. OpenCV find contours ellipse
                fpr_cv_ellipse_arr = []
                tpr_cv_ellipse_arr = []
                canny_thresholds1 = 0
                ratio_cv_ellipse_best = 100
                threshold_cv_ellipse_best = 0
                for canny_thresholds2 in np.arange(canny_thresholds1, 300, 1):
                    predict_mask_ellipse = extract_AO_Ellipse(original_img, original_img, canny_thresholds1, canny_thresholds2)
                    fpr_cv_ellipse, tpr_cv_ellipse, ratio = compute_fp_tp(mask_img, predict_mask_ellipse)
                    if ratio < ratio_cv_ellipse_best:
                        ratio_cv_ellipse_best = ratio
                        best_fpr_cv_ellipse = fpr_cv_ellipse
                        best_tpr_cv_ellipse = tpr_cv_ellipse
                        threshold_cv_ellipse_best = canny_thresholds2
                    fpr_cv_ellipse_arr.append(fpr_cv_ellipse)
                    tpr_cv_ellipse_arr.append(tpr_cv_ellipse)
                # save best segmentation
                predict_mask_ellipse = extract_AO_Ellipse(original_img, original_img, canny_thresholds1, threshold_cv_ellipse_best, plot="1.a")
                if predict_mask_ellipse is not None:
                    # cv2.imwrite(f'Dataset/task1/cv_ellipse/{name}_1.a_predicted_cv_ellipse_mask.png', predict_mask_ellipse)
                    # cv2.imwrite(f'Dataset/task1/cv_ellipse_neo/{name}_1.a_predicted_cv_ellipse_mask.png', predict_mask_ellipse)
                    with open(file_paths["cv_ellipse"], "a") as file:
                        file.write(f"{best_fpr_cv_ellipse}\n")
                    with open(file_paths["cv_ellipse_tpr"], "a") as file:
                        file.write(f"{best_tpr_cv_ellipse}\n")
                    with open(file_paths["best_threshold_cv_ellipse"], "a") as file:
                        file.write(f"{threshold_cv_ellipse_best}\n")
                else:
                    with open(file_paths["best_threshold_cv_ellipse"], "a") as file:
                        file.write("\n")

                # 3. Color thresholding
                fpr_color_thresholding_arr = []
                tpr_color_thresholding_arr = []
                ratio_color_thresholding_best = 100
                threshold_color_thresholding_best = 0
                thresholds = np.arange(0, 256, 20)
                for threshold in thresholds:
                    predict_mask = extract_AO_Color_Threshold(original_img, threshold)
                    fpr_color_thresholding, tpr_color_thresholding, ratio = compute_fp_tp(mask_img, predict_mask)
                    if ratio < ratio_color_thresholding_best:
                        ratio_color_thresholding_best = ratio
                        best_fpr_color_thresholding = fpr_color_thresholding
                        best_tpr_color_thresholding = tpr_color_thresholding
                        threshold_color_thresholding_best = threshold
                    fpr_color_thresholding_arr.append(fpr_color_thresholding)
                    tpr_color_thresholding_arr.append(tpr_color_thresholding)
                # save the segmentation
                predict_color_thresholding_mask = extract_AO_Color_Threshold(original_img, threshold_color_thresholding_best)
                # cv2.imwrite(f'Dataset/task1/color_thresholding/{name}_1.a_predicted_color_thresholding.png', predict_color_thresholding_mask)
                # cv2.imwrite(f'Dataset/task1/color_thresholding_neo/{name}_1.a_predicted_color_thresholding.png', predict_color_thresholding_mask)
                
                with open(file_paths["color_thresholding"], "a") as file:
                    file.write(f"{best_fpr_color_thresholding}\n")
                with open(file_paths["color_thresholding_tpr"], "a") as file:
                    file.write(f"{best_tpr_color_thresholding}\n")
                with open(file_paths["best_threshold_color_thresholding"], "a") as file:
                    file.write(f"{threshold_color_thresholding_best}\n")

                # # ########################################################### Task 1.b ###################################################################
                # 1. color thresholding + ellipse
                fpr_all_color_arr = []
                tpr_all_color_arr = []
                predict_mask_color = predict_color_thresholding_mask
                predict_mask = extract_AO_color_ellipse(original_img, predict_mask_color)
                best_fpr_all_color, best_tpr_all_color, ratio = compute_fp_tp(mask_img, predict_mask)
                predict_mask = extract_AO_color_ellipse(original_img, predict_mask_color)
                if predict_mask is not None:
                    # cv2.imwrite(f'Dataset/task1/color_ellipse/{name}_1.b_predicted_combine_mask.png', predict_mask)
                    # cv2.imwrite(f'Dataset/task1/color_ellipse_neo/{name}_1.b_predicted_combine_mask.png', predict_mask)
                    with open(file_paths["all_color_tpr"], "a") as file:
                        file.write(f"{best_tpr_all_color}\n")
                    with open(file_paths["all_color"], "a") as file:
                        file.write(f"{best_fpr_all_color}\n")

                # 2. kmeans + ellipse
                fpr_all_kmeans_arr = []
                tpr_all_kmeans_arr = []
                predict_mask_color = predicted_kmeans
                # save the segmentation
                predict_mask = extract_AO_k_means_ellipse(original_img, predict_mask_color)
                best_fpr_all_kmeans, best_tpr_all_kmeans, best_ratio_all_kmeans = compute_fp_tp(mask_img, predict_mask)
                if predict_mask is not None:
                    # cv2.imwrite(f'Dataset/task1/kmeans_ellipse/{name}_1.b_predicted_combine_mask.png', predict_mask)
                    # cv2.imwrite(f'Dataset/task1/kmeans_ellipse_neo/{name}_1.b_predicted_combine_mask.png', predict_mask)
                    with open(file_paths["all_kmeans"], "a") as file:
                        file.write(f"{best_fpr_all_kmeans}\n")
                    with open(file_paths["all_kmeans_tpr"], "a") as file:
                        file.write(f"{best_tpr_all_kmeans}\n")
    else:
        plot_roc(file_paths)