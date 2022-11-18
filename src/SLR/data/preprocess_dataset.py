import pandas as pd
import cv2
import pickle
from pathlib import Path
import importlib
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Dict, Tuple


def _load_processor(name: str):
    """Tool function for loading dataset processor
    located in 'src/SLR/data/dataset_processors/'

    Parameters
    ----------
    name : str
        Dataset name
    """
    module = importlib.import_module(f"src.SLR.data.dataset_processors.{name}")
    processor_class = getattr(module, name)
    return processor_class


def _update_gloss_dict(annotation: pd.DataFrame, gloss_dict: Dict) -> Dict:
    """Tool function for updating `gloss_dict` accoring to `annotation`

    Parameters
    ----------
    annotation : pd.DataFrame
        Annotation dataframe
    gloss_dict : Dict
        Gloss dictionary

    Returns
    -------
    Dict
        Update gloss dictionary
    """
    for _, row in annotation.iterrows():
        words = row["label"].split()
        for gloss in words:
            gloss_dict[gloss] += 1
    return gloss_dict


def _save_groundtruth(annotation: pd.DataFrame, save_path: Path):
    # TODO refactor
    df = annotation[["fileid", "signer", "label"]]
    df["a"] = 1
    df["b"] = 0.0
    df["c"] = "1.79769e+308"

    df[["fileid", "a", "signer", "b", "c", "label"]].to_csv(
        save_path, header=False, index=False, sep=" "
    )


def _resize_img(
    img_dir: Path,
    pattern: str,
    save_dir: Path,
    target_size: Tuple[int, int] = (210, 260),
):
    """Tool function for resizing images from a single video

    Parameters
    ----------
    img_dir : Path
        Image dir, should be like 'DATA_PATH/MODE/VIDEO_NAME'
    pattern : str
        Image pattern, should be like '**/*.png'
    save_dir : Path
        Save directory, should be like 'DATA_PATH/MODE/VIDEO_NAME'
    target_size : Tuple[int, int], optional
        Target size, by default (210, 260)
    """
    imgs = img_dir.glob(pattern)
    save_dir.mkdir(parents=True, exist_ok=True)
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(save_dir / img_path.name), img)


def preprocess_dataset(
    dataset_name: str,
    raw_info_dir: str,
    raw_img_dir: str,
    processed_info_dir: str,
    processed_img_dir: str,
    target_size: Tuple[int, int],
    n_jobs: int,
    resize_img: bool,
):
    """Preprocess dataset

    Parameters
    ----------
    dataset_name : str
        Dataset name
    raw_info_dir : str
        Raw annotation dir
    raw_img_dir : str
        Raw image dir
    processed_info_dir : str
        Processed annotation dir
    processed_img_dir : str
        Processed image dir
    target_size : Tuple[int, int]
        Target resize size
    n_jobs : int
        Number of CPU cores to be used
    resize_img : bool
        Whether to resize images
    """
    raw_info_dir = Path(raw_info_dir)
    raw_img_dir = Path(raw_img_dir)
    processed_info_dir = Path(processed_info_dir)
    processed_img_dir = Path(processed_img_dir)

    processed_info_dir.mkdir(parents=True, exist_ok=True)
    processed_img_dir.mkdir(parents=True, exist_ok=True)

    gloss_dict = defaultdict(int)
    for mode in ["train", "dev", "test"]:
        print(f"{mode}")

        # * ------------------------ Preprocess Annotations ------------------------
        print(f"Preprocessing annotations ...", end="\t")
        # * Get processor class
        processor = _load_processor(dataset_name)()
        # * Load annotations
        annotation = processor.load_annotation(raw_info_dir, mode)
        # * Parse annotations
        annotation = processor.parse_annotation(annotation, raw_img_dir, mode)
        # * Save annotations
        annotation.to_csv(processed_info_dir / f"{mode}_info.csv", index=None)
        print("Done")

        # * ------------------------ Update Gloss Dictionary ------------------------

        gloss_dict = _update_gloss_dict(annotation, gloss_dict)

        # * --------------------------- Save Groundtruth ----------------------------
        print(f"Saving groundtruth ...", end="\t")
        _save_groundtruth(annotation, processed_info_dir / f"groundtruth-{mode}.stm")
        print("Done")
        # * ---------------------------- Resize Images ------------------------------

        if resize_img:
            parallel = Parallel(n_jobs=n_jobs, backend="multiprocessing")
            parallel(
                delayed(_resize_img)(
                    raw_img_dir / mode / row["fileid"],
                    row["pattern"],
                    processed_img_dir / mode / row["fileid"],
                    target_size,
                )
                for _, row in tqdm(
                    annotation.iterrows(),
                    total=annotation.shape[0],
                    desc=f"Resizing {mode} set imgs",
                )
            )
        print("")

    # * Save gloss dict
    gloss_dict = sorted(gloss_dict.items(), key=lambda d: d[0])
    save_dict = {}
    for idx, (key, value) in enumerate(gloss_dict):
        save_dict[key] = [idx + 1, value]  # * [encode_number, count]
    pickle.dump(save_dict, open(processed_info_dir / "gloss_dict.pkl", "wb"))
