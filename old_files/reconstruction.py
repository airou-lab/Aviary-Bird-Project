import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_images(image_dir):
    images = []
    filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
    for filename in filenames:
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)
    return images, filenames

def detect_keypoints(frame, xmin, ymin, xmax, ymax, nfeatures=500):
    orb = cv2.ORB_create(nfeatures)
    mask = np.zeros_like(frame[:,:,0])
    mask[ymin:ymax, xmin:xmax] = 255
    keypoints, descriptors = orb.detectAndCompute(frame, mask)
    if descriptors is None:
        descriptors = np.array([], dtype=np.float32).reshape(0, 32)
    return keypoints, descriptors

def match_keypoints(descriptors1, descriptors2):
    if descriptors1.size == 0 or descriptors2.size == 0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return sorted(matches, key=lambda x: x.distance)

def estimate_essential_matrix(pts1, pts2, camera_matrix):
    E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, cv2.RANSAC, 0.999, 1.0)
    return E, mask

def decompose_and_triangulate(pts1, pts2, camera_matrix, E):
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, camera_matrix)
    points_3D = cv2.triangulatePoints(np.eye(3, 4), np.hstack((R, t)), pts1.T, pts2.T)
    points_3D /= points_3D[3]
    return points_3D[:3].T

def plot_3d_points(points_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Different colors for different sets of points
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    for i, pts in enumerate(points_3d):
        ax.scatter(pts[:, 0] / 304.8, pts[:, 1] / 304.8, pts[:, 2] / 304.8, c=colors[i % len(colors)], label=f'Set {i+1}')
    ax.set_xlabel('X (feet)')
    ax.set_ylabel('Y (feet)')
    ax.set_zlabel('Z (feet)')
    ax.legend()
    plt.show()

def visualize_matches(img1, kp1, img2, kp2, matches):
    matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.title('Keypoint Matches')
    plt.show()

def process_images_and_detections(camera_dirs, csv_files, camera_matrix):
    all_points_3d = []  # Store all triangulated points
    for camera_index, (image_dir, csv_path) in enumerate(zip(camera_dirs, csv_files)):
        images, _ = load_images(image_dir)
        detections = pd.read_csv(csv_path)
        detections.columns = ['Frame', 'Path', 'Extra', 'Label', 'Confidence', 'xmin', 'ymin', 'xmax', 'ymax']

        if camera_index == 0:
            base_images = images
            base_keypoints, base_descriptors = zip(*[detect_keypoints(img, int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax)) for img, row in zip(images, detections.itertuples(index=False))])
        else:
            for frame_idx, (frame, row) in enumerate(zip(images, detections.itertuples(index=False))):
                if frame_idx < len(base_images):
                    keypoints, descriptors = detect_keypoints(frame, int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax))
                    matches = match_keypoints(base_descriptors[frame_idx], descriptors)
                    if len(matches) > 8:
                        pts1 = np.float32([base_keypoints[frame_idx][m.queryIdx].pt for m in matches])
                        pts2 = np.float32([keypoints[m.trainIdx].pt for m in matches])
                        E, mask = estimate_essential_matrix(pts1, pts2, camera_matrix)
                        points_3D = decompose_and_triangulate(pts1[mask.ravel() == 1], pts2[mask.ravel() == 1], camera_matrix, E)
                        all_points_3d.append(points_3D)
                        print(f"Triangulated {len(points_3D)} points between frame {frame_idx + 1} of camera 1 and camera {camera_index + 1}.")
                        # visualize_matches(base_images[frame_idx], base_keypoints[frame_idx], frame, keypoints, [matches[m] for m in mask.ravel().nonzero()[0]])

    # After processing all data, plot the 3D points
    if all_points_3d:
        plot_3d_points(all_points_3d)

# Camera calibration matrix
camera_matrix = np.array([[2.47313196e+03, 0.00000000e+00, 2.78379536e+03],
                          [0.00000000e+00, 2.45870179e+03, 2.41120292e+03],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Setup
camera_dirs = [
    '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/Camera1',
    '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/Camera2'
]
csv_files = [
    '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/csv/cleaned_output1.csv',
    '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/csv/cleaned_output2.csv'
]
process_images_and_detections(camera_dirs, csv_files, camera_matrix)
