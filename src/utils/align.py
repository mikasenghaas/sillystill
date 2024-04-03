import numpy as np
import cv2 as cv

from typing import List, Tuple, Union


def extract_features(
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


def match_features(query_ds, train_ds, method="flann", **kwargs):
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


def transform_image(query, train, query_kp, train_kp, matches) -> np.ndarray:
    """
    Aligns two images using ORB feature detection and homography matrix.

    Args:
        img1: Image 1
        img2: Image 2

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
    h, w = query.shape[:2]  # query
    aligned_train = cv.warpPerspective(train, M, (w, h))[0:h, 0:w]

    return aligned_train


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
        train (np.ndarray): Train image
        query (np.ndarray): Query image

    Returns:
        Tuple of aligned images
    """
    # Extract features
    query_kp, query_ds = extract_features(
        query, method=extract_method, **extract_kwargs
    )
    train_kp, train_ds = extract_features(
        train, method=extract_method, mask=mask, **extract_kwargs
    )

    # Match features
    matches = match_features(query_ds, train_ds, method=match_method, **match_kwargs)

    # Align images
    aligned_train = transform_image(
        query, train, query_kp, train_kp, matches, **transform_kwargs
    )

    return aligned_train, query
