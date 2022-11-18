from pathlib import Path
import pandas as pd
from collections import defaultdict
from typing import Dict

from .utils import count_imgs
from .base_processor import BaseProcessor


class Phoenix2014(BaseProcessor):
    def load_annotation(self, info_dir: Path, mode: str) -> pd.DataFrame:
        """Load phoenix2014 annotations

        Parameters
        ----------
        info_dir : Path
            Info directory
        mode : str
            Dataset mode, should be one of ['train','dev','test']

        Returns
        -------
        pd.DataFrame
            Phoenix2014 annotations
        """
        df = pd.read_csv(info_dir / f"{mode}.corpus.csv", sep="|")
        return df

    def parse_annotation(
        self, df: pd.DataFrame, img_dir: Path, mode: str
    ) -> pd.DataFrame:
        """Parse phoenix2014 dataset info

        Parameters
        ----------
        df : pd.DataFrame
            Dataset info
        img_dir : str
            Root dir of storing imgs
        mode : str
            Dataset type, should be one of ['train','dev','test']

        Returns
        -------
        pd.DataFrame
            Processed dataset info, should at least
                contain ['fileid', 'label', 'signer', 'pattern', 'num_frames']
        """
        df.rename(columns={"id": "fileid", "annotation": "label"}, inplace=True)

        folder = img_dir / mode
        df["num_frames"] = df.apply(lambda x: count_imgs(folder, x["folder"]), axis=1)
        df["pattern"] = df["folder"].apply(lambda x: "**/" + x.split("/")[-1])
        return df
