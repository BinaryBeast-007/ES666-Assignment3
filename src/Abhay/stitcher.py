import pdb
import glob
import cv2
import os
from src.ExampleModule import example_function
from src.ExampleModule.subdirectory import sub_function
from typing import List, Tuple, Optional
import numpy as np
import random

random.seed(1234)

def apply_perspective_transform(points, transform_matrix):

    shape_original = points.shape

    if points.ndim == 3:
        points = points.reshape(-1, 2)

    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    points_transformed = points_homogeneous @ transform_matrix.T

    # Normalize by the third (homogeneous) coordinate
    points_transformed = points_transformed[:, :2] / points_transformed[:, 2:]

    out_of_bounds = np.abs(points_transformed) > 1e10
    points_transformed[out_of_bounds] = 0

    if shape_original[-1] == 3:
        points_transformed = points_transformed.reshape(shape_original)

    return points_transformed

def perform_image_warping(image, transform_matrix, dimensions):

    width, height = dimensions
    output_image = np.zeros((height, width, image.shape[2]) if image.ndim == 3 else (height, width), dtype=image.dtype)
    input_height, input_width = image.shape[:2]
    grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    coords_homogeneous = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)
    all_points = coords_homogeneous.reshape(-1, 3)

    # Invert the transformation matrix
    transform_inv = np.linalg.inv(transform_matrix)
    all_points_transformed = all_points @ transform_inv.T

    all_points_transformed = all_points_transformed[:, :2] / all_points_transformed[:, 2:]
    all_points_transformed = all_points_transformed.reshape(height, width, 2)

    x_coords = all_points_transformed[:, :, 0]
    y_coords = all_points_transformed[:, :, 1]

    valid = (
        (x_coords >= 0) & (x_coords < input_width - 1) &
        (y_coords >= 0) & (y_coords < input_height - 1)
    )

    x_coords = x_coords.astype(np.int32)
    y_coords = y_coords.astype(np.int32)

    if image.ndim == 3:
        output_image[valid] = image[y_coords[valid], x_coords[valid]]
    else:
        output_image[valid] = image[y_coords[valid], x_coords[valid]]

    return output_image

def stitch_images(left_image, right_image):

    key_pts_left, desc_left, key_pts_right, desc_right = detect_keypoints(left_image, right_image)
    matches = find_matches(key_pts_left, key_pts_right, desc_left, desc_right)
    final_transform = compute_ransac(matches)

    # Define corners of the images
    dimensions_left = right_image.shape[:2]
    dimensions_right = left_image.shape[:2]
    corners_left = np.float32([[0, 0], [0, dimensions_left[0]], [dimensions_left[1], dimensions_left[0]], [dimensions_left[1], 0]]).reshape(-1, 1, 2)
    corners_right = np.float32([[0, 0], [0, dimensions_right[0]], [dimensions_right[1], dimensions_right[0]], [dimensions_right[1], 0]]).reshape(-1, 1, 2)
    corners_right_transformed = apply_perspective_transform(corners_right, final_transform)
    all_corners = np.concatenate((corners_left, corners_right_transformed), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    transform_translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]]) @ final_transform

    panorama = perform_image_warping(left_image, transform_translation, (xmax - xmin, ymax - ymin))
    panorama[-ymin:dimensions_left[0] - ymin, -xmin:dimensions_left[1] - xmin] = right_image
    return panorama, final_transform

def detect_keypoints(left_image, right_image):
    # Detection and description of keypoints in both images
    keypoint_detector = cv2.SIFT_create()
    keypoints_left, descriptors_left = keypoint_detector.detectAndCompute(left_image, None)
    keypoints_right, descriptors_right = keypoint_detector.detectAndCompute(right_image, None)
    return keypoints_left, descriptors_left, keypoints_right, descriptors_right

def find_matches(keypoints_left, keypoints_right, descriptors_left, descriptors_right):
    # Finding good matches between keypoints in both images
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = matcher.knnMatch(descriptors_left, descriptors_right, k=2)

    good_matches = []
    for first, second in matches:
        if first.distance < 0.75 * second.distance:
            good_matches.append([
                keypoints_left[first.queryIdx].pt[0], keypoints_left[first.queryIdx].pt[1],
                keypoints_right[first.trainIdx].pt[0], keypoints_right[first.trainIdx].pt[1]
            ])
    return good_matches

def compute_ransac(matches):
    # Applying RANSAC to estimate a robust homography
    best_inliers = []
    best_homography = None
    threshold = 5
    for _ in range(500):
        sampled_matches = random.sample(matches, 4)
        homography_matrix = estimate_homography(sampled_matches)
        inliers = []

        for match in matches:
            point_left = np.array([match[0], match[1], 1]).reshape(3, 1)
            point_right = np.array([match[2], match[3], 1]).reshape(3, 1)
            projected_point = np.dot(homography_matrix, point_left)
            projected_point /= projected_point[2]
            distance = np.linalg.norm(point_right - projected_point)

            if distance < threshold:
                inliers.append(match)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_homography = homography_matrix

    return best_homography

def estimate_homography(points):
    # Estimating a homography matrix from point correspondences
    matrix = []
    for point in points:
        x, y = point[0], point[1]
        X, Y = point[2], point[3]
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])

    A = np.array(matrix)
    _, _, vh = np.linalg.svd(A)
    homography = vh[-1].reshape(3, 3)
    return homography / homography[2, 2]
