from pathlib import Path
import pandas as pd
from collections import defaultdict
from typing import Dict

from .utils import count_imgs
from .base_processor import BaseProcessor


class Phoenix2014T(BaseProcessor):
    def load_annotation(self, info_dir: Path, mode: str) -> pd.DataFrame:
        """Load phoenix2014T annotations

        Parameters
        ----------
        info_dir : Path
            Info directory
        mode : str
            Dataset mode, should be one of ['train','dev','test']

        Returns
        -------
        pd.DataFrame
            Phoenix2014T annotations
        """
        df = pd.read_csv(info_dir / f"PHOENIX-2014-T.{mode}.corpus.csv", sep="|")
        return df

    def parse_annotation(
        self, df: pd.DataFrame, img_dir: Path, mode: str
    ) -> pd.DataFrame:
        """Parse phoenix2014T dataset info

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
        df.rename(
            columns={
                "name": "fileid",
                "orth": "label",
                "speaker": "signer",
            },
            inplace=True,
        )

        df["pattern"] = df["video"].apply(lambda x: "**/" + x.split("/")[-1])
        df["num_frames"] = df.apply(
            lambda x: count_imgs(img_dir / mode / x["fileid"], x["pattern"]), axis=1
        )
        return df
