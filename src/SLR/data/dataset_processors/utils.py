from pathlib import Path


def count_imgs(img_dir: Path, pattern: str) -> int:
    """Tool function for counting number of imgs in target folder

    Parameters
    ----------
    img_dir : Path
        Target folder
    pattern : str
        Search pattern, should be like '*.png'

    Returns
    -------
    int
        Number of images
    """
    return len(list(Path(img_dir).glob(pattern)))
