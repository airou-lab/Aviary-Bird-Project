import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial

# Image paths
image_paths = [
    '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/For_Keon_1/frame_0615.jpg',
    '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/For_Keon_2/frame_0615.jpg',
    '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/frames/For_Keon_3/frame_0615.jpg'
]

# Camera calibration parameters
K = np.array([[8.28013819e+02, 0.00000000e+00, 1.01014119e+03],
              [0.00000000e+00, 8.20546041e+02, 8.21208275e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([-0.2600221, 0.12330861, -0.07062931, -0.04367669, -0.02240264])

# Load corner and security camera coordinates
def load_landmarks(file_path):
    landmarks = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(int, line.strip().split(','))
            landmarks.append((x, y))
    return landmarks

# Assuming these are the correct filenames for the landmarks
landmarks_1 = load_landmarks('corners_1.txt')
landmarks_2 = load_landmarks('corners_2.txt')
landmarks_3 = load_landmarks('corners_3.txt')

# Manually defined bounding boxes for feeder and water bowl
feeder_water_bowl_1 = [(1078, 539, 1175, 614), (1226, 590, 1314, 664)]  # feeder and water bowl
feeder_water_bowl_2 = [(894, 640, 973, 692), (833, 726, 926, 788)]
feeder_water_bowl_3 = [(959, 556, 1032, 597), (993, 496, 1069, 533)]

# Combine the manually defined landmarks and corner/security camera coordinates
def combine_landmarks(landmarks, feeder_water_bowl, corner_labels, camera_labels):
    combined_landmarks = []
    for i, landmark in enumerate(landmarks):
        combined_landmarks.append({'type': 'corner' if i < 2 else 'camera', 'coordinates': landmark, 'label': corner_labels[i] if i < 2 else camera_labels[i-2]})
    for i, bbox in enumerate(feeder_water_bowl):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        combined_landmarks.append({'type': 'feeder' if i == 0 else 'water', 'coordinates': (center_x, center_y), 'label': 'Feeder' if i == 0 else 'Water Bowl'})
    return combined_landmarks

# Define labels for each landmark type in each image
corner_labels_1 = ['Corner 2', 'Corner 3']
corner_labels_2 = ['Corner 1', 'Corner 2']
corner_labels_3 = ['Corner 3', 'Corner 4']

camera_labels = ['Security Cam 1', 'Security Cam 2']

combined_landmarks_1 = combine_landmarks(landmarks_1, feeder_water_bowl_1, corner_labels_1, camera_labels)
combined_landmarks_2 = combine_landmarks(landmarks_2, feeder_water_bowl_2, corner_labels_2, camera_labels)
combined_landmarks_3 = combine_landmarks(landmarks_3, feeder_water_bowl_3, corner_labels_3, camera_labels)

#### Step 1: Visualize Original Detections ####

# Function to draw bounding boxes and landmarks on an image
def draw_landmarks(image, landmarks):
    for landmark in landmarks:
        x, y = landmark['coordinates']
        color = (255, 0, 0) if landmark['type'] in ['corner', 'camera'] else (0, 0, 255)
        cv2.circle(image, (x, y), 5, color, -1)
        cv2.putText(image, landmark['label'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Draw original detections
image1_copy = cv2.imread(image_paths[0])
image2_copy = cv2.imread(image_paths[1])
image3_copy = cv2.imread(image_paths[2])
draw_landmarks(image1_copy, combined_landmarks_1)
draw_landmarks(image2_copy, combined_landmarks_2)
draw_landmarks(image3_copy, combined_landmarks_3)

# Plot images with detections
fig, axes = plt.subplots(1, 3, figsize=(20, 10))
axes[0].imshow(cv2.cvtColor(image1_copy, cv2.COLOR_BGR2RGB))
axes[0].set_title("Image 1 with Landmarks")
axes[1].imshow(cv2.cvtColor(image2_copy, cv2.COLOR_BGR2RGB))
axes[1].set_title("Image 2 with Landmarks")
axes[2].imshow(cv2.cvtColor(image3_copy, cv2.COLOR_BGR2RGB))
axes[2].set_title("Image 3 with Landmarks")
plt.show()

#### Step 2: Undistort the Images ####

# Function to undistort image
def undistort_image(image, K, dist):
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, K, dist, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]
    return undistorted_image

# Undistort images
undistorted_image1 = undistort_image(cv2.imread(image_paths[0]), K, dist)
undistorted_image2 = undistort_image(cv2.imread(image_paths[1]), K, dist)
undistorted_image3 = undistort_image(cv2.imread(image_paths[2]), K, dist)

# Plot undistorted images
fig, axes = plt.subplots(1, 3, figsize=(20, 10))
axes[0].imshow(cv2.cvtColor(undistorted_image1, cv2.COLOR_BGR2RGB))
axes[0].set_title("Undistorted Image 1")
axes[1].imshow(cv2.cvtColor(undistorted_image2, cv2.COLOR_BGR2RGB))
axes[1].set_title("Undistorted Image 2")
axes[2].imshow(cv2.cvtColor(undistorted_image3, cv2.COLOR_BGR2RGB))
axes[2].set_title("Undistorted Image 3")
plt.show()

#### Step 3: Generate Voronoi Cells Based on Landmarks ####

# Function to generate Voronoi cells based on landmarks
def generate_voronoi_landmarks(landmarks):
    points = np.array([landmark['coordinates'] for landmark in landmarks])
    vor = scipy.spatial.Voronoi(points)
    return vor

# Helper function to convert Voronoi to finite regions
def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)

# Generate Voronoi cells for each image based on landmarks
vor1 = generate_voronoi_landmarks(combined_landmarks_1)
vor2 = generate_voronoi_landmarks(combined_landmarks_2)
vor3 = generate_voronoi_landmarks(combined_landmarks_3)

regions1, vertices1 = voronoi_finite_polygons_2d(vor1)
regions2, vertices2 = voronoi_finite_polygons_2d(vor2)
regions3, vertices3 = voronoi_finite_polygons_2d(vor3)

# Function to draw Voronoi cells on an image
def draw_voronoi(image, regions, vertices, landmarks):
    for region in regions:
        polygon = vertices[region]
        cv2.polylines(image, [np.int32(polygon)], isClosed=True, color=(0, 255, 0), thickness=2)
    for landmark in landmarks:
        x, y = landmark['coordinates']
        color = (255, 0, 0) if landmark['type'] in ['corner', 'camera'] else (0, 0, 255)
        cv2.circle(image, (x, y), 5, color, -1)
        cv2.putText(image, landmark['label'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Draw Voronoi cells on undistorted images based on landmarks
undistorted_image1_voronoi_landmarks = undistorted_image1.copy()
undistorted_image2_voronoi_landmarks = undistorted_image2.copy()
undistorted_image3_voronoi_landmarks = undistorted_image3.copy()
draw_voronoi(undistorted_image1_voronoi_landmarks, regions1, vertices1, combined_landmarks_1)
draw_voronoi(undistorted_image2_voronoi_landmarks, regions2, vertices2, combined_landmarks_2)
draw_voronoi(undistorted_image3_voronoi_landmarks, regions3, vertices3, combined_landmarks_3)

# Plot images with Voronoi cells based on landmarks
fig, axes = plt.subplots(1, 3, figsize=(20, 10))
axes[0].imshow(cv2.cvtColor(undistorted_image1_voronoi_landmarks, cv2.COLOR_BGR2RGB))
axes[0].set_title("Undistorted Image 1 with Voronoi Cells (Landmarks)")
axes[1].imshow(cv2.cvtColor(undistorted_image2_voronoi_landmarks, cv2.COLOR_BGR2RGB))
axes[1].set_title("Undistorted Image 2 with Voronoi Cells (Landmarks)")
axes[2].imshow(cv2.cvtColor(undistorted_image3_voronoi_landmarks, cv2.COLOR_BGR2RGB))
axes[2].set_title("Undistorted Image 3 with Voronoi Cells (Landmarks)")
plt.show()
