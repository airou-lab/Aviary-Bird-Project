import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paths to the images
image_path1 = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/For_Keon_1/frame_0615.jpg'
image_path2 = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/For_Keon_2/frame_0615.jpg'
image_path3 = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/For_Keon_3/frame_0615.jpg'

# Parameters for manual point selection
num_points_per_bird = 10
num_birds = 2

# Load the selected points
def load_selected_points(filename):
    points = {f'bird_{i+1}': [] for i in range(num_birds)}
    with open(filename, 'r') as f:
        for line in f:
            bird_id, x, y = line.strip().split(',')
            points[bird_id].append((int(x), int(y)))
    return points

# Load the manually selected points
selected_points_cam1 = load_selected_points('/selected_points_cam/selected_points_cam1.txt')
selected_points_cam2 = load_selected_points('/selected_points_cam/selected_points_cam2.txt')
selected_points_cam3 = load_selected_points('/selected_points_camselected_points_cam3.txt')

# Function to triangulate points
def triangulate_points(pts1, pts2, proj_matrix1, proj_matrix2):
    pts1 = np.array(pts1, dtype='float32').T
    pts2 = np.array(pts2, dtype='float32').T
    points_4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, pts1, pts2)
    points_3d = points_4d[:3] / points_4d[3]
    return points_3d.T

# Function to filter outliers
def filter_outliers(points_3d, x_range, y_range, z_range):
    mask = ((points_3d[:, 0] >= x_range[0]) & (points_3d[:, 0] <= x_range[1]) &
            (points_3d[:, 1] >= y_range[0]) & (points_3d[:, 1] <= y_range[1]) &
            (points_3d[:, 2] >= z_range[0]) & (points_3d[:, 2] <= z_range[1]))
    return points_3d[mask]

# Camera intrinsic parameters
K = np.array([[8.28013819e+02, 0.00000000e+00, 1.01014119e+03],
              [0.00000000e+00, 8.20546041e+02, 8.21208275e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
rvecs = [np.array([[-0.40848219], [-0.04476641], [-1.36582084]]),
         np.array([[-0.57560136], [0.06694504], [0.01289696]]),
         np.array([[-0.0139436], [1.02401999], [2.95480862]])]
tvecs = [np.array([[-2.64222845], [0.4595596], [1.91394887]]),
         np.array([[-1.20345239], [-2.50698575], [3.88573475]]),
         np.array([[1.15682178], [1.07354459], [1.30999783]])]

proj_matrix1 = K @ np.hstack((cv2.Rodrigues(rvecs[0])[0], tvecs[0]))
proj_matrix2 = K @ np.hstack((cv2.Rodrigues(rvecs[1])[0], tvecs[1]))
proj_matrix3 = K @ np.hstack((cv2.Rodrigues(rvecs[2])[0], tvecs[2]))

# Triangulate points using manually selected points
points_3d_bird1_cam1_cam2 = triangulate_points(selected_points_cam1['bird_1'], selected_points_cam2['bird_1'], proj_matrix1, proj_matrix2)
points_3d_bird1_cam2_cam3 = triangulate_points(selected_points_cam2['bird_1'], selected_points_cam3['bird_1'], proj_matrix2, proj_matrix3)
points_3d_bird1_cam1_cam3 = triangulate_points(selected_points_cam1['bird_1'], selected_points_cam3['bird_1'], proj_matrix1, proj_matrix3)

points_3d_bird2_cam1_cam2 = triangulate_points(selected_points_cam1['bird_2'], selected_points_cam2['bird_2'], proj_matrix1, proj_matrix2)
points_3d_bird2_cam2_cam3 = triangulate_points(selected_points_cam2['bird_2'], selected_points_cam3['bird_2'], proj_matrix2, proj_matrix3)
points_3d_bird2_cam1_cam3 = triangulate_points(selected_points_cam1['bird_2'], selected_points_cam3['bird_2'], proj_matrix1, proj_matrix3)

# Define the expected range of the aviary
x_range = (0, 2.44)
y_range = (0, 4.572)
z_range = (-0.5, 0.5)

# Filter outliers
points_3d_bird1_cam1_cam2 = filter_outliers(points_3d_bird1_cam1_cam2, x_range, y_range, z_range)
points_3d_bird1_cam2_cam3 = filter_outliers(points_3d_bird1_cam2_cam3, x_range, y_range, z_range)
points_3d_bird1_cam1_cam3 = filter_outliers(points_3d_bird1_cam1_cam3, x_range, y_range, z_range)

points_3d_bird2_cam1_cam2 = filter_outliers(points_3d_bird2_cam1_cam2, x_range, y_range, z_range)
points_3d_bird2_cam2_cam3 = filter_outliers(points_3d_bird2_cam2_cam3, x_range, y_range, z_range)
points_3d_bird2_cam1_cam3 = filter_outliers(points_3d_bird2_cam1_cam3, x_range, y_range, z_range)

# Function to compute the average 3D point
def compute_average_point(points_3d):
    return np.mean(points_3d, axis=0)

# Compute the average 3D points for each bird
avg_point_bird1_cam1_cam2 = compute_average_point(points_3d_bird1_cam1_cam2)
avg_point_bird1_cam2_cam3 = compute_average_point(points_3d_bird1_cam2_cam3)
avg_point_bird1_cam1_cam3 = compute_average_point(points_3d_bird1_cam1_cam3)

avg_point_bird2_cam1_cam2 = compute_average_point(points_3d_bird2_cam1_cam2)
avg_point_bird2_cam2_cam3 = compute_average_point(points_3d_bird2_cam2_cam3)
avg_point_bird2_cam1_cam3 = compute_average_point(points_3d_bird2_cam1_cam3)

# Visualize average 3D points from different camera perspectives
def visualize_avg_3d_points(avg_point_bird1, avg_point_bird2, rvecs, tvecs, filename_prefix):
    fig = plt.figure(figsize=(18, 6))
    
    # Plot from Camera 1 perspective
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(avg_point_bird1[0], avg_point_bird1[1], avg_point_bird1[2], c='r', marker='x', s=100, label='Bird 1 Avg')
    ax1.scatter(avg_point_bird2[0], avg_point_bird2[1], avg_point_bird2[2], c='b', marker='x', s=100, label='Bird 2 Avg')
    ax1.view_init(elev=-np.rad2deg(rvecs[0][0][0]), azim=np.rad2deg(rvecs[0][2][0]))
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_zlabel('Z (meters)')
    ax1.set_title('Camera 1 Perspective')
    ax1.legend()
    
    # Plot from Camera 2 perspective
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(avg_point_bird1[0], avg_point_bird1[1], avg_point_bird1[2], c='r', marker='x', s=100, label='Bird 1 Avg')
    ax2.scatter(avg_point_bird2[0], avg_point_bird2[1], avg_point_bird2[2], c='b', marker='x', s=100, label='Bird 2 Avg')
    ax2.view_init(elev=-np.rad2deg(rvecs[1][0][0]), azim=np.rad2deg(rvecs[1][2][0]))
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_zlabel('Z (meters)')
    ax2.set_title('Camera 2 Perspective')
    ax2.legend()
    
    # Plot from Camera 3 perspective
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(avg_point_bird1[0], avg_point_bird1[1], avg_point_bird1[2], c='r', marker='x', s=100, label='Bird 1 Avg')
    ax3.scatter(avg_point_bird2[0], avg_point_bird2[1], avg_point_bird2[2], c='b', marker='x', s=100, label='Bird 2 Avg')
    ax3.view_init(elev=-np.rad2deg(rvecs[2][0][0]), azim=np.rad2deg(rvecs[2][2][0]))
    ax3.set_xlabel('X (meters)')
    ax3.set_ylabel('Y (meters)')
    ax3.set_zlabel('Z (meters)')
    ax3.set_title('Camera 3 Perspective')
    ax3.legend()
    
    fig.suptitle('Average 3D Points from Camera Perspectives')
    plt.savefig(f'{filename_prefix}.png')
    plt.show()

# Visualize average 3D points from each camera perspective
visualize_avg_3d_points(avg_point_bird1_cam1_cam2, avg_point_bird2_cam1_cam2, rvecs, tvecs, 'avg_3d_points_perspective_cam1_cam2')
visualize_avg_3d_points(avg_point_bird1_cam2_cam3, avg_point_bird2_cam2_cam3, rvecs, tvecs, 'avg_3d_points_perspective_cam2_cam3')
visualize_avg_3d_points(avg_point_bird1_cam1_cam3, avg_point_bird2_cam1_cam3, rvecs, tvecs, 'avg_3d_points_perspective_cam1_cam3')

# Visualize birds in images
def visualize_birds_in_images(image_path, points, color):
    image = cv2.imread(image_path)
    for point in points:
        cv2.circle(image, point, 5, color, -1)
    return image

proj_image1_bird1 = visualize_birds_in_images(image_path1, selected_points_cam1['bird_1'], (0, 0, 255))
proj_image2_bird1 = visualize_birds_in_images(image_path2, selected_points_cam2['bird_1'], (0, 0, 255))
proj_image3_bird1 = visualize_birds_in_images(image_path3, selected_points_cam3['bird_1'], (0, 0, 255))

proj_image1_bird2 = visualize_birds_in_images(image_path1, selected_points_cam1['bird_2'], (255, 0, 0))
proj_image2_bird2 = visualize_birds_in_images(image_path2, selected_points_cam2['bird_2'], (255, 0, 0))
proj_image3_bird2 = visualize_birds_in_images(image_path3, selected_points_cam3['bird_2'], (255, 0, 0))

# Save and display projected points
cv2.imwrite('proj_image_1_bird1.png', proj_image1_bird1)
cv2.imwrite('proj_image_2_bird1.png', proj_image2_bird1)
cv2.imwrite('proj_image_3_bird1.png', proj_image3_bird1)

cv2.imwrite('proj_image_1_bird2.png', proj_image1_bird2)
cv2.imwrite('proj_image_2_bird2.png', proj_image2_bird2)
cv2.imwrite('proj_image_3_bird2.png', proj_image3_bird2)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(proj_image1_bird1, cv2.COLOR_BGR2RGB))
plt.title('Projected Bird 1 - Camera 1')
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(proj_image2_bird1, cv2.COLOR_BGR2RGB))
plt.title('Projected Bird 1 - Camera 2')
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(proj_image3_bird1, cv2.COLOR_BGR2RGB))
plt.title('Projected Bird 1 - Camera 3')
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(proj_image1_bird2, cv2.COLOR_BGR2RGB))
plt.title('Projected Bird 2 - Camera 1')
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(proj_image2_bird2, cv2.COLOR_BGR2RGB))
plt.title('Projected Bird 2 - Camera 2')
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(proj_image3_bird2, cv2.COLOR_BGR2RGB))
plt.title('Projected Bird 2 - Camera 3')
plt.show()
