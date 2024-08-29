import numpy as np
import re

def parse_matrix_from_lines(lines):
    """Extracts a matrix from a list of lines containing numerical data."""
    matrix_data = []
    for line in lines:
        # Updated regex to handle scientific notation properly
        numbers = [float(num) for num in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", line)]
        matrix_data.append(numbers)
    return np.array(matrix_data)

def parse_vector_data(data, start_index, end_index):
    """Parses vector data where each vector's components might be on the same line or split across multiple lines."""
    vectors = []
    buffer = []  # Temporary buffer to hold numbers as we parse them
    for i in range(start_index, end_index):
        # Extract all numbers from the current line
        numbers = [float(num) for num in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", data[i])]
        buffer.extend(numbers)
        # Assuming each vector should have 3 components, check if buffer has collected 3 numbers
        if len(buffer) >= 3:
            vectors.append(np.array(buffer[:3]))
            buffer = buffer[3:]  # Remove the first 3 components, keep the rest if any
    return vectors

def load_calibration(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.readlines()
        
        # Parse camera matrix and distortion coefficients
        camera_matrix = parse_matrix_from_lines(data[1:4])
        distortion_coeffs = parse_matrix_from_lines([data[5]])[0]

        # Parse rotation and translation vectors
        rotation_vectors = parse_vector_data(data, 8, 23)
        translation_vectors = parse_vector_data(data, 26, 41)

        return camera_matrix, distortion_coeffs, rotation_vectors, translation_vectors
    except Exception as e:
        print(f"Failed to load calibration data from {file_path}: {e}")
        return None, None, None, None

# Example usage
camera_matrix1, dist_coeffs1, rotation_vectors1, translation_vectors1 = load_calibration('cam_cal/Cam1Calibration.txt')
if camera_matrix1 is not None:
    print("Camera Matrix 1:", camera_matrix1)
else:
    print("Failed to load Camera 1 Calibration Data.")


