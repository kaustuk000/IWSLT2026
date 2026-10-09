import os
import re
import numpy as np
import pandas as pd

try:
    from helper.preprocessor import preprocess
except ImportError:
    from preprocessor import preprocess


class MTDatasetBuilder:
    """Preprocesses, filters, deduplicates, and splits a Hindi–Bhojpuri MT dataset."""

    REASON_MAP = {
        "_flag_hin_too_short":  "hin_too_short",
        "_flag_bho_too_short":  "bho_too_short",
        "_flag_ratio_low":      "ratio_too_low",
        "_flag_ratio_high":     "ratio_too_high",
        "_flag_true_duplicate": "true_duplicate",
    }

    def __init__(self, df: pd.DataFrame, out_dir: str = "data/MT_data"):
        self.df = df.copy()
        self.out_dir = out_dir

    def build(self) -> pd.DataFrame:
        df = self.df.copy()   # copy so self.df stays pristine
        before = len(df)

        df["hin"] = df["hin"].apply(preprocess)
        df["bho"] = df["bho"].apply(preprocess)

        df = self._apply_flags(df)
        flag_cols = [c for c in df.columns if c.startswith("_flag_")]
        kept, removed = self._remove_flagged(df, flag_cols)

        self._save_removed(removed, flag_cols)
        self._print_report(before, kept, removed, flag_cols, df)

        return self._shuffle_and_split(kept, flag_cols)

    @staticmethod
    def normalize_for_dedup(text: str) -> str:
        text = re.sub(r"[।?,!.\-\'\"]+", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def _apply_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        df["_hin_norm"] = df["hin"].apply(self.normalize_for_dedup)
        df["_bho_norm"] = df["bho"].apply(self.normalize_for_dedup)

        ratio = df["hin"].str.len() / (df["bho"].str.len() + 1)

        df["_flag_hin_too_short"]  = df["hin"].str.len() <= 3
        df["_flag_bho_too_short"]  = df["bho"].str.len() <= 3
        df["_flag_ratio_low"]      = ratio < 0.33
        df["_flag_ratio_high"]     = ratio > 3.0
        df["_flag_true_duplicate"] = df.duplicated(
            subset=["_bho_norm", "_hin_norm"], keep="first"
        )
        return df

    @staticmethod
    def _remove_flagged(
        df: pd.DataFrame, flag_cols: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        is_removed = df[flag_cols].any(axis=1)
        return df[~is_removed].copy(), df[is_removed].copy()

    def _save_removed(self, removed: pd.DataFrame, flag_cols: list[str]) -> None:
        removed["removal_reasons"] = removed[flag_cols].apply(
            lambda row: ", ".join(self.REASON_MAP[c] for c in flag_cols if row[c]),
            axis=1,
        )
        os.makedirs(self.out_dir, exist_ok=True)
        removed[["bho", "hin", "removal_reasons"]].to_csv(
            f"{self.out_dir}/removed.csv", index=False
        )

    def _print_report(
        self,
        before: int,
        kept: pd.DataFrame,
        removed: pd.DataFrame,
        flag_cols: list[str],
        df_full: pd.DataFrame,
    ) -> None:
        print(f"\nRows before : {before:,}")
        print(f"Rows kept   : {len(kept):,}")
        print(f"Rows removed: {len(removed):,}  → {self.out_dir}/removed.csv")
        print("\nRemoval breakdown:")
        for col, label in self.REASON_MAP.items():
            n = int(df_full[col].sum())
            if n:
                print(f"  {label:<22} {n:>6,}")

    @staticmethod
    def _shuffle_and_split(kept: pd.DataFrame, flag_cols: list[str]) -> pd.DataFrame:
        norm_cols = ["_hin_norm", "_bho_norm"]
        to_drop = [c for c in flag_cols + norm_cols if c in kept.columns]
        kept = kept.drop(columns=to_drop).reset_index(drop=True)
        kept = kept.sample(frac=1, random_state=42).reset_index(drop=True)

        n = len(kept)
        kept["split"] = np.where(
            np.arange(n) < 0.90 * n, "train",
            np.where(np.arange(n) < 0.95 * n, "dev", "test"),
        )
        print(f"\nSplit counts:\n{kept['split'].value_counts().to_string()}")
        return kept


class SplitSaver:
    """Writes train/dev/test splits from a built DataFrame to disk."""

    def __init__(self, df: pd.DataFrame, out_dir: str = "data/MT_data"):
        self.df = df
        self.out_dir = out_dir

    def save(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        for split in ("train", "dev", "test"):
            self._save_split(split)

    def _save_split(self, split: str) -> None:
        mask = self.df["split"] == split
        path = f"{self.out_dir}/{split}.csv"
        self.df[mask][["bho", "hin"]].to_csv(path, index=False)
        print(f"Saved {split}: {mask.sum():,} rows → {path}")


if __name__ == "__main__":
    OUT_DIR = "data/MT_data/noisy_data"
    raw_df = pd.read_csv("data/MT_data/noisy_data/hi_bho_noisy.csv")   

    built_df = MTDatasetBuilder(raw_df, out_dir=OUT_DIR).build()
    SplitSaver(built_df, out_dir=OUT_DIR).save()
