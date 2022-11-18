from abc import ABC, abstractmethod
from typing import Dict
from pathlib import Path
import pandas as pd


class BaseProcessor(ABC):
    @abstractmethod
    def load_annotation(self, info_dir: Path, mode: str) -> pd.DataFrame:
        """Load dataset annotations

        Parameters
        ----------
        info_dir : Path
            Info directory
        mode : str
            Dataset mode, should be one of ['train','dev','test']

        Returns
        -------
        pd.DataFrame
            Annotation dataframe
        """
        pass

    @abstractmethod
    def parse_annotation(
        self, df: pd.DataFrame, img_dir: Path, mode: str
    ) -> pd.DataFrame:
        """Parse dataset annotations

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
            Parsed annotation dataframe
        """
        pass
