import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

# Load the bounding boxes from a file
def read_boxes(file_path):
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            box = tuple(map(int, line.strip().split(',')))
            boxes.append(box)
    return boxes

# Apply CLAHE to enhance image contrast
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Extract features from fixed objects (feeder and water bowl)
def extract_fixed_object_features(image, roi_coords, method='sift', num_features=10000):
    (x1, y1, x2, y2) = roi_coords
    roi = image[y1:y2, x1:x2]
    if method == 'sift':
        detector = cv2.SIFT_create(nfeatures=num_features)
    elif method == 'orb':
        detector = cv2.ORB_create(nfeatures=num_features)
    elif method == 'brisk':
        detector = cv2.BRISK_create()
    elif method == 'akaze':
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    keypoints, descriptors = detector.detectAndCompute(roi, None)
    if keypoints is not None:
        for kp in keypoints:
            kp.pt = (kp.pt[0] + x1, kp.pt[1] + y1)
    return keypoints, descriptors

# Visualize keypoints on the image
def visualize_keypoints(image, keypoints):
    keypoint_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Keypoints', keypoint_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Visualize matches between two images
def visualize_matches(image1, keypoints1, image2, keypoints2, matches, title='Matches'):
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow(title, matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Match features using BFMatcher
def match_features(descriptors1, descriptors2, ratio_thresh=0.85):
    bf = cv2.BFMatcher()
    matches = []
    
    if descriptors1 is not None and descriptors2 is not None:
        raw_matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        for m_n in raw_matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio_thresh * n.distance:
                    matches.append(m)
    
    return matches

# Triangulate points
def triangulate_points(pts1, pts2, proj_matrix1, proj_matrix2):
    if pts1.shape[1] == 0 or pts2.shape[1] == 0:
        print("No points to triangulate")
        return np.zeros((3, 1, 3))
    
    points_hom = cv2.triangulatePoints(proj_matrix1, proj_matrix2, pts1[:2], pts2[:2])
    points_3d = cv2.convertPointsFromHomogeneous(points_hom.T)
    return points_3d

# Filter ghost points
def filter_ghost_points(points_3d):
    filtered_points = []
    for pt in points_3d:
        if np.linalg.norm(pt) > 1e-3:  # Threshold to filter out near-zero points
            filtered_points.append(pt)
    return np.array(filtered_points)

# Cluster points using DBSCAN
def cluster_points(points_3d, eps=0.2, min_samples=1):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points_3d)
    labels = db.labels_
    clusters = [points_3d[labels == i] for i in range(len(set(labels)) - (1 if -1 in labels else 0))]
    return clusters, labels

# Get cluster centers
def get_cluster_centers(clusters):
    centers = [np.mean(cluster, axis=0) for cluster in clusters]
    return centers

# Visualize point clouds
def visualize_point_cloud(points_3d, clusters, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=labels, cmap='viridis', s=50)

    for cluster in clusters:
        ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], s=100)

    ax.set_xlabel('X (feet)')
    ax.set_ylabel('Y (feet)')
    ax.set_zlabel('Z (feet)')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

# Main function
def main():
    # Paths to the images
    image_path1 = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/For_Keon_1/frame_0615.jpg'
    image_path2 = '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/For_Keon_2/frame_0615.jpg'
    
    # Load images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    
    if image1 is None or image2 is None:
        print("Error: One or more images could not be loaded.")
        return

    image1_clahe = apply_clahe(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY))
    image2_clahe = apply_clahe(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY))

    # Define approximate ROI coordinates for feeder in each image
    feeder_roi1 = (1078, 539, 1175, 614)
    feeder_roi2 = (894, 640, 973, 692)

    # Extract features from feeder
    keypoints_feeder1, descriptors_feeder1 = extract_fixed_object_features(image1, feeder_roi1, method='sift')
    keypoints_feeder2, descriptors_feeder2 = extract_fixed_object_features(image2, feeder_roi2, method='sift')

    # Match feeder features across images
    feeder_matches_12 = match_features(descriptors_feeder1, descriptors_feeder2)

    print(f"Number of feeder matches: {len(feeder_matches_12)}")

    # Visualize feeder matches
    visualize_matches(image1, keypoints_feeder1, image2, keypoints_feeder2, feeder_matches_12, title='Feeder Matches 1-2')

    # Filter matches with epipolar constraints
    if feeder_matches_12:
        pts1 = np.float32([keypoints_feeder1[m.queryIdx].pt for m in feeder_matches_12]).T
        pts2 = np.float32([keypoints_feeder2[m.trainIdx].pt for m in feeder_matches_12]).T
        
        print(f"Number of points in pts1: {pts1.shape[1]}")
        print(f"Number of points in pts2: {pts2.shape[1]}")

        if len(pts1.T) == len(pts2.T) and len(pts1.T) > 0:
            fundamental_matrix, mask = cv2.findFundamentalMat(pts1.T, pts2.T, cv2.FM_RANSAC)
            if mask is not None:
                matches_epipolar = [m for m, inlier in zip(feeder_matches_12, mask.ravel()) if inlier]
                print(f"Number of matches after epipolar filtering: {len(matches_epipolar)}")

                # Visualize matches after epipolar filtering
                visualize_matches(image1, keypoints_feeder1, image2, keypoints_feeder2, matches_epipolar, title='Epipolar Feeder Matches 1-2')

                # Simplified projection matrices for the cameras
                K = np.array([[1., 0., 960.],
                              [0., 1., 540.],
                              [0., 0., 1.]])
                
                proj_matrix1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))  # Projection matrix for camera 1
                R = np.eye(3)  # Identity rotation matrix
                T = np.array([[-0.2], [0], [0]])  # Translation vector
                proj_matrix2 = np.dot(K, np.hstack((R, T)))  # Projection matrix for camera 2

                pts1_hom = np.vstack([pts1, np.ones((1, pts1.shape[1]))])
                pts2_hom = np.vstack([pts2, np.ones((1, pts2.shape[1]))])

                print(f"Shape of pts1_hom: {pts1_hom.shape}")
                print(f"Shape of pts2_hom: {pts2_hom.shape}")

                points_3d = triangulate_points(pts1_hom, pts2_hom, proj_matrix1, proj_matrix2)
                print("Triangulated points:")
                print(points_3d)

                # Convert to feet (assuming the aviary dimensions were provided in meters)
                points_3d_feet = points_3d * 3.28084  # 1 meter = 3.28084 feet
                print("Triangulated points in feet:")
                print(points_3d_feet)

                # Filter ghost points
                filtered_points = filter_ghost_points(points_3d_feet)
                print("Filtered points:")
                print(filtered_points)

                # Reshape filtered_points to 2D
                filtered_points_2d = filtered_points.reshape(-1, 3)

                # Cluster points
                if len(filtered_points_2d) > 0:
                    clusters, labels = cluster_points(filtered_points_2d, eps=0.1, min_samples=2)
                    cluster_centers = get_cluster_centers(clusters)
                    print(f"Clusters: {len(clusters)}")
                    print("Cluster centers:")
                    print(cluster_centers)

                    # Visualize 3D point cloud and clusters
                    visualize_point_cloud(filtered_points_2d, clusters, labels)
                else:
                    print("No points left after filtering ghost points.")
            else:
                print("Epipolar filtering did not find enough inliers.")
        else:
            print("Mismatch in the number of points between pts1 and pts2.")
    else:
        print("No matches found.")

if __name__ == '__main__':
    main()
