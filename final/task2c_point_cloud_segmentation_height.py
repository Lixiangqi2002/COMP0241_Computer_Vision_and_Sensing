import cv2
from task1 import extract_AO_Kmeans, extract_AO_k_means_ellipse
import numpy as np
import matplotlib.pyplot as plt


rgb = cv2.imread(f'Dataset/task2/2-c/2-c_pointCloud/RGB_db.jpg')
depth = cv2.imread(f'Dataset/task2/2-c/2-c_pointCloud/Depth_db.jpg')
rgb = cv2.resize(rgb, (640, 400))
depth = cv2.resize(depth, (640, 400))

hist = cv2.calcHist([depth], [0], None, [256], [0, 256])
plt.figure(figsize=(10, 6))
plt.plot(hist, color='blue')
plt.title("Depth Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.grid(True)
# plt.savefig("Dataset/task2/depth_histogram.png", dpi=300)
plt.show()

cv2.imshow('rgb', rgb)
# cv2.imwrite('Dataset/task2/rgb_resize.png', hist)
cv2.waitKey(0)
cv2.imshow('depth', depth)
# cv2.imwrite('Dataset/task2/depth_resize.png', hist)
cv2.waitKey(0)

predict_mask = extract_AO_Kmeans(rgb, threshold=100)
predict_mask = extract_AO_k_means_ellipse(rgb, predict_mask)
predict_mask = cv2.cvtColor(predict_mask, cv2.COLOR_BGR2GRAY)
cv2.imshow('predict_mask', predict_mask)
# cv2.imwrite('Dataset/task2/predict_mask.png', predict_mask)
cv2.waitKey(0)

depth_gray = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

depth_mask = depth_gray.copy()
depth_mask = cv2.bitwise_and(depth_gray, depth_gray, mask=predict_mask)
cv2.imshow('depth_mask', depth_mask)
# cv2.imwrite('Dataset/task2/depth_mask.png', depth_mask)
cv2.waitKey(0)

mask = depth_mask > 0  
depth_mask = depth_mask[mask]


plt.figure(figsize=(10, 6))
hist_values, bins, _ = plt.hist(depth_mask.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.7)
max_bin_index = np.argmax(hist_values)
max_bin_value = bins[max_bin_index]
plt.axvline(x=max_bin_value, color='red', linestyle='--', label=f'Highest Value: {max_bin_value}')
plt.title("Depth Distribution (Pixel Values)")
plt.xlabel("Depth Value (Pixel Intensity)")
plt.ylabel("Pixel Count")
plt.legend()
plt.grid(True)
# plt.savefig("Dataset/task2/depth_distribution.png", dpi=300)
plt.show()
