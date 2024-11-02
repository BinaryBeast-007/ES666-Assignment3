import pdb
import glob
import cv2
import os
from src.JohnDoe import some_function
from src.JohnDoe.some_folder import folder_func
from typing import List, Tuple, Optional
import numpy as np
import random

random.seed(1234)

def perspective_transform(points, transformation_matrix):
    """
    Apply perspective transformation to a set of 2D points.

    Parameters:
    points: numpy array of shape (N, 2) or (N, 1, 2) containing 2D points
    transformation_matrix: 3x3 perspective transformation matrix

    Returns:
    numpy.ndarray: Transformed points in same shape as input
    """
    original_shape = points.shape

    if len(points.shape) == 3:
        points = points.reshape(-1, 2)

    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])

    transformed_points = homogeneous_points @ transformation_matrix.T

    # Convert back from homogeneous coordinates
    # Divide by the homogeneous coordinate (z)
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]

    mask = np.abs(transformed_points) > 1e10
    transformed_points[mask] = 0

    if len(original_shape) == 3:
        transformed_points = transformed_points.reshape(original_shape)

    return transformed_points

def warp_perspective(img, transformation_matrix, output_size):
    """
    Apply perspective transformation to an image.

    Parameters:
    img (numpy.ndarray): Input image
    transformation_matrix (numpy.ndarray): 3x3 transformation matrix
    output_size (tuple): (width, height) of output image

    Returns:
    numpy.ndarray: Transformed image
    """
    width, height = output_size
    output = np.zeros((height, width, 3) if len(img.shape) == 3 else (height, width), dtype=img.dtype)
    input_height, input_width = img.shape[:2]
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    homogeneous_coords = np.stack([x, y, np.ones_like(x)], axis=-1)
    points = homogeneous_coords.reshape(-1, 3)

    # Apply inverse transformation
    H_inv = np.linalg.inv(transformation_matrix)
    transformed_points = points @ H_inv.T

    transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]
    transformed_points = transformed_points.reshape(height, width, 2)

    src_x = transformed_points[:, :, 0]
    src_y = transformed_points[:, :, 1]

    valid_mask = (
        (src_x >= 0) & (src_x < input_width - 1) &
        (src_y >= 0) & (src_y < input_height - 1)
    )

    src_x = src_x.astype(np.int32)
    src_y = src_y.astype(np.int32)

    if len(img.shape) == 3:
        output[valid_mask] = img[src_y[valid_mask], src_x[valid_mask]]
    else:
        output[valid_mask] = img[src_y[valid_mask], src_x[valid_mask]]

    return output

def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """

    key_points1, descriptor1, key_points2, descriptor2 = get_keypoint(left_img, right_img)
    good_matches = match_keypoint(key_points1, key_points2, descriptor1, descriptor2)
    final_H = ransac(good_matches)

    rows1, cols1 = right_img.shape[:2]
    rows2, cols2 = left_img.shape[:2]

    points1 =  np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    points  =  np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    points2 =  perspective_transform(points, final_H)
    list_of_points = np.concatenate((points1,points2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    H_translation = (np.array([[1, 0, (-x_min)], [0, 1, (-y_min)], [0, 0, 1]])).dot(final_H)

    output_img = warp_perspective(left_img, H_translation, (x_max-x_min, y_max-y_min))
    output_img[(-y_min):rows1+(-y_min), (-x_min):cols1+(-x_min)] = right_img
    result_img = output_img
    return result_img, final_H

    
def get_keypoint(left_img, right_img):
    kp1, des1 = cv2.SIFT_create().detectAndCompute(left_img, None)
    kp2, des2 = cv2.SIFT_create().detectAndCompute(right_img, None)
    return kp1, des1, kp2, des2

def match_keypoint(key_points1, key_points2, descriptor1, descriptor2):
    #k-Nearest neighbours between each descriptor
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)

    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            left_pt = key_points1[m.queryIdx].pt
            right_pt = key_points2[m.trainIdx].pt
            good_matches.append([left_pt[0], left_pt[1], right_pt[0], right_pt[1]])

    return good_matches

def homography(points):
    A = []
    for pt in points:
      x, y = pt[0], pt[1]
      X, Y = pt[2], pt[3]
      A.append([x, y, 1, 0, 0, 0, -1 * X * x, -1 * X * y, -1 * X])
      A.append([0, 0, 0, x, y, 1, -1 * Y * x, -1 * Y * y, -1 * Y])

    A = np.array(A)
    u, s, vh = np.linalg.svd(A)
    H = (vh[-1, :].reshape(3, 3))
    H = H/ H[2, 2]
    return H

def ransac(good_pts):
    best_inliers = []
    final_H = []
    t=5
    for i in range(500):
        random_pts = random.choices(good_pts, k=4)
        H = homography(random_pts)
        inliers = []
        for pt in good_pts:
            p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
            p_1 = np.array([pt[2], pt[3], 1]).reshape(3, 1)
            Hp = np.dot(H, p)
            Hp = Hp / Hp[2]
            dist = np.linalg.norm(p_1 - Hp)

            if dist < t: inliers.append(pt)

        if len(inliers) > len(best_inliers):
            best_inliers,final_H = inliers,H
    return final_H








class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        # Collect all homographies calculated for pair of images and return
        homography_matrix_list =[]

        if len(all_images) < 2:
            print('Need atleast 2 images to stitch')
            return None
        else:
            resized_size = 0.25 if len(all_images) >= 6 else 0.6
            print('Image size is reduced to:', resized_size, "times the original size")
            result_img = cv2.resize(cv2.imread(all_images[0]), (0,0), fx=resized_size, fy=resized_size)
            for i in range(1, 5):
                left_img = result_img
                right_img = cv2.resize(cv2.imread(all_images[i]), (0,0), fx=resized_size, fy=resized_size)
                result_img, final_H = solution(left_img, right_img)
                homography_matrix_list.append(final_H)
                print('Stitching Done for Image {} with panorama'.format(i))
        stitched_image = result_img
        
        return stitched_image, homography_matrix_list
