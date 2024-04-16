import os
import numpy as np
import cv2 as cv

from typing import List, Tuple, Union


def _extract_features(
    img: np.ndarray, method: str = "orb", mask: Union[np.ndarray, None] = None, **kwargs
) -> Tuple[List[cv.KeyPoint], np.ndarray]:
    """
    Extract features from an image using a given method.
    Currently supports SIFT and ORB.

    Args:
        img (np.ndarray): Image to get features

    Returns:
        Tuple[List[cv.KeyPoint], np.ndarray]: Tuple containing the keypoints and descriptors
    """
    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Create the extractor
    if method == "sift":
        extractor = cv.SIFT_create(**kwargs)
    elif method == "orb":
        extractor = cv.ORB_create(**kwargs)
    else:
        raise ValueError("Invalid method")

    # Extract features
    kp, des = extractor.detectAndCompute(gray, mask)

    return kp, des


def _match_features(query_ds, train_ds, method="flann", **kwargs):
    """
    Match features between two sets of keypoints and descriptors.
    Currently supports brute-force and FLANN.

    Args:
        query_ds (np.ndarray): Descriptors of query image
        train_ds (np.ndarray): Descriptors of train image
        method (str, optional): Matching method. Defaults to "bf".
        **kwargs: Additional arguments for the matcher

    Returns:
        Tuple[List[cv.KeyPoint], np.ndarray, List[cv.KeyPoint], np.ndarray, List[cv.DMatch]]: Tuple containing the keypoints and descriptors
    """
    # Create the matcher
    if method == "bf":
        matcher = cv.BFMatcher(**kwargs)
    elif method == "flann":
        matcher = cv.FlannBasedMatcher(**kwargs)
    else:
        raise ValueError("Invalid method")

    # Match the descriptors
    matches = matcher.knnMatch(query_ds, train_ds, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append([m])

    return good_matches


def _transform_image(
    query: np.ndarray, train: np.ndarray, query_kp: List, train_kp: List, matches: List
) -> np.ndarray:
    """
    Aligns two images using a homography matrix estimated from keypoint matches.

    Args:
        query (np.ndarray): The query image
        train (np.ndarray): The train image
        query_kp (List): Keypoints of the query image
        train_kp (List): Keypoints of the train image
        matches (List): List of matches

    Returns:
        Transformed train image
    """
    # Find keypoints
    query_pts = np.float32([query_kp[m[0].queryIdx].pt for m in matches]).reshape(
        -1, 1, 2
    )  # query
    train_pts = np.float32([train_kp[m[0].trainIdx].pt for m in matches]).reshape(
        -1, 1, 2
    )  # train

    # Estimate homography matrix
    M, _ = cv.findHomography(train_pts, query_pts, cv.RANSAC, 5.0)

    # Find the perspective transformation
    h, w = query.shape[:2]
    transformed_train = cv.warpPerspective(train, M, (w, h))[0:h, 0:w]

    return transformed_train


def align_images(
    query: np.ndarray,
    train: np.ndarray,
    mask: Union[np.ndarray, None] = None,
    extract_method: str = "orb",
    match_method: str = "bf",
    extract_kwargs: dict = {},
    match_kwargs: dict = {},
    transform_kwargs: dict = {},
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns the train image (complex scence, expected to include the query)
    image using a pipeline of feature extraction, matching and homography.

    For our dataset, the train image is the digital image and the query image
    is the film image.

    Args:
        query (np.ndarray): Query image
        train (np.ndarray): Train image
        mask (np.ndarray, optional): Mask for feature extraction
        extract_method (str, optional): Feature extraction method. Defaults to "orb".
        match_method (str, optional): Feature matching method. Defaults to "bf".
        extract_kwargs (dict, optional): Additional arguments for feature extraction. Defaults to {}.
        match_kwargs (dict, optional): Additional arguments for feature matching. Defaults to {}.

    Returns:
        Tuple of aligned images (query, aligned_train)
    """
    # Extract features
    query_kp, query_ds = _extract_features(
        query, method=extract_method, **extract_kwargs
    )
    train_kp, train_ds = _extract_features(
        train, method=extract_method, mask=mask, **extract_kwargs
    )

    # Match features
    matches = _match_features(query_ds, train_ds, method=match_method, **match_kwargs)

    # Align images
    aligned_train = _transform_image(
        query, train, query_kp, train_kp, matches, **transform_kwargs
    )

    return query, aligned_train
