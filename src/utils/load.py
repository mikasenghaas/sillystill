import os
import yaml

from PIL import Image as PIL
from PIL.Image import Image
import pandas as pd

from typing import Literal, Tuple, List, Dict, Optional, Union

ImageType = Literal["digital", "film"]
ImageFormat = Literal["raw", "processed"]


def _assert_image_format(image_format: str) -> None:
    assert image_format in ["raw", "processed"], f"Invalid image format: {image_format}"


def _assert_image_type(image_type: str) -> None:
    assert image_type in ["digital", "film"], f"Invalid image type: {image_type}"


def _assert_inputs(idx: int, image_type: str, image_format: str) -> None:
    assert isinstance(idx, int), f"Index must be an integer, not {type(idx)}"
    _assert_image_type(image_type)
    _assert_image_format(image_format)


def _load_image_from_path(path: str) -> Image:
    """
    Load an image from a file into a PIL Image.

    Args:
        path (str): The path to the image file.

    Returns:
        Image: The loaded image.
    """
    # Assertions
    assert os.path.exists(path), f"Image file {path} does not exist."

    # Load the image
    return PIL.open(path).convert("RGB")


def load_image(idx: int, image_type: ImageType, image_format: ImageFormat) -> Image:
    """
    Load an image of a specific type and format for a given index
    of our image-pairs dataset.

    Args:
        idx (int): The index of the image pair.
        image_type (ImageType): The type of image to load.
        image_format (ImageFormat): The format of the image to load.

    Returns:
        Image: The loaded image.
    """
    # Assertions
    _assert_inputs(idx, image_type, image_format)

    # Set base directory
    data_dir = os.path.join("data", image_format, image_type)

    # Match over files to search indepently of image format (.tif, .png, etc.)
    filenames = [f for f in os.listdir(data_dir) if f"{idx:03d}" in f]
    assert len(filenames) == 1, f"Expected 1 image file, found {len(filenames)}."

    return _load_image_from_path(os.path.join(data_dir, filenames[0]))


def load_metadata(
    idx: Optional[int] = None, as_df: bool = False
) -> Union[Dict, pd.DataFrame]:
    """
    Load the metadata for a specific image pair or all image pairs.

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
    idx: int, image_format: Literal["raw", "processed"]
) -> Tuple[Image, Image, Dict]:
    """
    Load a pair of images and their metadata. Each image pair is stored in the
    digital and film directories with the same filename.

    Args:
        idx (int): The index of the image pair to load.
        format (str): The format of the images to load.

    Returns:
        Tuple of digital image, film image and metadata.
    """
    # Assertions
    _assert_image_format(image_format)

    # Load the images and metadata
    meta = load_metadata(idx)
    digital = load_image(idx, image_type="digital", image_format=image_format)
    film = load_image(idx, image_type="film", image_format=image_format)

    return digital, film, meta
