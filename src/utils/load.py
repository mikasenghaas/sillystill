import os
from typing import Dict, List, Literal, Optional, Tuple, Union, get_args

import numpy as np
import pandas as pd
import yaml
from PIL import Image as PIL
from PIL.Image import Image

CameraType = Literal["film", "digital"]
ProcessingStateType = Literal["raw", "processed"]
ImageFormatType = Literal["tif", "png", "jpeg", "jpg"]
ColorSpaceType = Literal["RGB"]


def _assert_literal(value, expected_type) -> None:
    assert value in get_args(expected_type), f"Expected {expected_type}, got {value}"


def _load_image_from_path(path: str, as_array: bool = False) -> Image:
    """
    Load an image from a file into a PIL Image. If specified, return the image
    as an array.

    Args:
        path (str): The path to the image file.
        as_array (bool, optional): Whether to return the image as an array.

    Returns:
        Image: The loaded image.
    """
    # Assertions
    assert os.path.exists(path), f"Image file {path} does not exist."

    # Load the image
    img = PIL.open(path).convert("RGB")

    # Convert to array if specified
    if as_array:
        return np.array(img)
    return img


def _save_image_to_path(path: str, image: Union[np.ndarray, Image]) -> None:
    """Save an image to a file (can be numpy array or PIL Image)

    Args:
        path (str): The path to save the image.
        image (Image): The image to save.
    """
    # Convert to PIL image if necessary
    if isinstance(image, np.ndarray):
        image = PIL.fromarray(image)

    # Ensure save directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the image
    image.save(path)


def load_image(
    idx: int,
    processing_state: ProcessingStateType,
    camera: CameraType,
    as_array: bool = False,
) -> Image:
    """Load an image of a specific type and format for a given index of our image-pairs dataset.

    Args:
        idx (int): The index of the image pair.
        processing_state (str): The processing state of the image (raw/processed).
        camera (str): The camera type of the image (film/digital).
        as_array (bool, optional): Whether to return the image as an array.

    Returns:
        Image: The loaded image.
    """
    # Assertions
    _assert_literal(processing_state, ProcessingStateType)
    _assert_literal(camera, CameraType)

    # Set base directory
    data_dir = os.path.join("data", processing_state, camera)

    # Match over files to search independently of image format (.tif, .png, etc.)
    filenames = [f for f in os.listdir(data_dir) if f"{idx:03d}" in f]
    assert len(filenames) == 1, f"Expected 1 image file, found {len(filenames)}."

    return _load_image_from_path(
        os.path.join(data_dir, filenames[0]), as_array=as_array
    )


def save_image(
    idx: int,
    image: Image,
    processing_state: ProcessingStateType,
    camera: CameraType,
    image_format: ImageFormatType = "jpeg",
):
    """Save an image of a specific type and format for a given index of our image-pairs dataset.

    Args:
        idx (int): The index of the image pair
        processing_state (ProcessingStateType): The processing state of the image (raw/processed)
        camera (CameraType): The camera type of the image (film/digital)
        image_format (ImageFormatType): The format of the image to save
        image (Image): The image to save
    """
    # Assertions
    _assert_literal(processing_state, ProcessingStateType)
    _assert_literal(camera, CameraType)
    _assert_literal(image_format, ImageFormatType)

    # Set path to save the image
    data_dir = os.path.join("data", processing_state, camera)
    path = os.path.join(data_dir, f"{idx:03d}.{image_format}")

    # Save the image
    _save_image_to_path(path, image)


def load_metadata(
    idx: Optional[int] = None, as_df: bool = False
) -> Union[Dict, pd.DataFrame]:
    """Load the metadata for a specific image pair or all image pairs.

    Args:
        idx (int, optional): The index of the image pair.
        as_df (bool, optional): Whether to return the metadata as a DataFrame.

    Returns:
        dict: The metadata for the image pair.
    """
    # Set path to metadata file
    path = os.path.join("data", "meta.yaml")

    # Load metadata from YAML
    assert os.path.exists(path), f"Metadata file {path} does not exist."
    with open(path) as f:
        meta = yaml.safe_load(f)

    # Get the metadata for a specific image
    if idx is not None:
        assert idx in meta.keys(), f"Index {idx} not found in metadata."
        return meta[idx]

    # Turn metadata into a DataFrame, if specified
    if as_df:
        records = [{"id": key, **value} for key, value in meta.items()]
        meta = pd.DataFrame(records).set_index("id")

    # Return all of the metadata
    return meta


def load_image_pair(
    idx: int, processing_state: ProcessingStateType, as_array: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray, Dict], Tuple[Image, Image, Dict]]:
    """Load a pair of images and their metadata. Each image pair is stored in the digital and film
    directories with the same filename.

    Args:
        idx (int): The index of the image pair to load.
        processing_state (ProcessingStateType): The processing state of the images.
        as_array (bool, optional): Whether to return the images as arrays.

    Returns:
        Tuple of digital image, film image and metadata.
    """
    # Assertions
    _assert_literal(processing_state, ProcessingStateType)

    # Load the images and metadata
    meta = load_metadata(idx)
    digital = load_image(
        idx, camera="digital", processing_state=processing_state, as_array=as_array
    )
    film = load_image(
        idx, camera="film", processing_state=processing_state, as_array=as_array
    )

    return film, digital, meta


def save_image_pair(
    idx: int, film: Image, digital: Image, image_format: ImageFormatType = "jpeg"
):
    """Save a pair of images to the processed directory for a given index.

    Args:
        idx (int): The index of the image pair.
        film (Image): The film image.
        digital (Image): The digital image.
    """
    # Assertions
    _assert_literal(image_format, ImageFormatType)

    # Save images
    save_image(
        idx,
        film,
        camera="film",
        processing_state="processed",
        image_format=image_format,
    )
    save_image(
        idx,
        digital,
        camera="digital",
        processing_state="processed",
        image_format=image_format,
    )
