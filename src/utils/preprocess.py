import os
from typing import List, Tuple, Union

import cv2 as cv
import numpy as np
from skimage.exposure import cumulative_distribution


def _extract_features(
    img: np.ndarray, method: str = "orb", mask: Union[np.ndarray, None] = None, **kwargs
) -> Tuple[List[cv.KeyPoint], np.ndarray]:
    """Extract features from an image using a given method. Currently supports SIFT and ORB.

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
    """Match features between two sets of keypoints and descriptors. Currently supports brute-force
    and FLANN.

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
    """Aligns two images using a homography matrix estimated from keypoint matches.

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


def _cdf(channel: np.ndarray):
    """Computes the CDF of an image.

    Args:
        channel (np.ndarray): An image channel

    Returns:
        np.ndarray: The CDF of the image channel
    """
    # Compute the CDF and the bin centres
    cdf, b = cumulative_distribution(channel)

    # Pad the CDF to have values between 0 and 1
    cdf = np.insert(cdf, 0, [0] * b[0])
    cdf = np.append(cdf, [1] * (255 - b[-1]))

    return cdf


def _histogram_matching(
    template_cdf: np.ndarray, source_cdf: np.ndarray, channel: np.ndarray
) -> np.ndarray:
    """Matches the histogram of a channel to the histogram of another channel.

    Args:
        template_cdf (np.ndarray): The CDF of the template image
        source_cdf (np.ndarray): The CDF of the source image
        channel (np.ndarray): The channel to match (of source image)

    Returns:
        np.ndarray: The channel with the matched histogram
    """
    pixels = np.arange(256)
    # find closest pixel-matches corresponding to the CDF of the input image, given the value of the CDF H of
    # the template image at the corresponding pixels, s.t. c_t = H(pixels) <=> pixels = H-1(c_t)
    new_pixels = np.interp(source_cdf, template_cdf, pixels)
    new_channel = (np.reshape(new_pixels[channel.ravel()], channel.shape)).astype(np.uint8)

    return new_channel


def keypoint_align(
    query: np.ndarray,
    train: np.ndarray,
    mask: Union[np.ndarray, None] = None,
    extract_method: str = "orb",
    match_method: str = "bf",
    extract_kwargs: dict = {},
    match_kwargs: dict = {},
    transform_kwargs: dict = {},
) -> Tuple[np.ndarray, np.ndarray]:
    """Aligns the train image (complex scene, expected to include the query) image using a pipeline
    of feature extraction, matching and homography.

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
    query_kp, query_ds = _extract_features(query, method=extract_method, **extract_kwargs)
    train_kp, train_ds = _extract_features(
        train, method=extract_method, mask=mask, **extract_kwargs
    )

    # Match features
    matches = _match_features(query_ds, train_ds, method=match_method, **match_kwargs)

    # Align images
    aligned_train = _transform_image(query, train, query_kp, train_kp, matches, **transform_kwargs)

    return query, aligned_train


def luminance_align(template: np.ndarray, source: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Matches the luminance of the source image to the template image.

    Args:
        template (np.ndarray): The template image (RGB)
        source (np.ndarray): The source image (RGB)

    Returns:
        Tuple[np.ndarray, np.ndarray]: The source and template images
                                    with matched luminance
    """
    # Convert images from RGB to LAB
    source_lab = cv.cvtColor(source, cv.COLOR_RGB2LAB)
    template_lab = cv.cvtColor(template, cv.COLOR_RGB2LAB)

    # Split the image channels
    source_l, source_a, source_b = cv.split(source_lab)
    template_l, _, _ = cv.split(template_lab)

    # Compute the CDF of the images
    source_cdf = _cdf(source_l)
    template_cdf = _cdf(template_l)

    # Match the histograms
    matched_source_l = _histogram_matching(template_cdf, source_cdf, source_l)

    # Merge the new L channel with the original A and B channels
    source_lab = cv.merge((matched_source_l, source_a, source_b))

    # Convert back to RBG and then return result
    source = cv.cvtColor(source_lab, cv.COLOR_LAB2RGB)
    template = cv.cvtColor(template_lab, cv.COLOR_LAB2RGB)

    return template, source
