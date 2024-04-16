import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.load import load_metadata, load_image_pair, save_image_pair
from src.utils.preprocess import align_images


def main():
    """
    Preprocesses all the data inside the `data/raw` folder.
    """
    # Load all metadata
    meta = load_metadata()
    image_pair_idxs = list(meta.keys())

    pbar = tqdm(total=len(image_pair_idxs))
    for idx in pbar:
        # Load raw image pair
        film, digital, _ = load_image_pair(idx, processing_state="raw", as_array=True)

        # 1) Keypoint alignment
        try:
            pbar.set_description(f"Keypoint Alignment")
            film, digital = align_images(
                film, digital, extract_method="sift", match_method="flann"
            )
        except Exception as e:
            print(f"[ERROR] Failed to align image pair {idx}: {e}")
            continue

        # 2) Luminance alignment (TODO)

        # Save images
        save_image_pair(idx, film, digital)

    print(f"[DONE] Processed {len(image_pair_idxs)})")


if __name__ == "__main__":
    main()
