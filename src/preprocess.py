import cv2 as cv
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.load import load_image_pair, load_metadata, save_image_pair
from src.utils.preprocess import keypoint_align, luminance_align


def main():
    """Preprocesses all the data inside the `data/raw` folder."""
    # Set OpenCV seed
    cv.setRNGSeed(42)

    # Load all metadata
    meta = load_metadata()
    image_pair_idxs = list(meta.keys())

    pbar = tqdm(image_pair_idxs, total=len(image_pair_idxs))
    for idx in pbar:
        # Load raw image pair
        film, digital, _ = load_image_pair(idx, processing_state="raw", as_array=True)

        # 1) Keypoint alignment
        try:
            pbar.set_description("Keypoint Alignment")
            film, digital = keypoint_align(
                film,
                digital,
                extract_method="sift",
                match_method="flann",
            )
        except Exception as e:
            print(f"[ERROR] Failed to align image pair {idx}: {e}")
            continue

        # 2) Luminance alignment
        pbar.set_description("Luminance Alignment")
        digital, film = luminance_align(template=digital, source=film)

        # Save processed image pair
        save_image_pair(idx, film, digital)

    print(f"[DONE] Processed {len(image_pair_idxs)} image pairs.")


if __name__ == "__main__":
    main()
