import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ------------------------------------------------------------------
# STAGE 1: Imputasi berbasis median grup
# ------------------------------------------------------------------
class GroupedImputerStage(BaseEstimator, TransformerMixin):
    """Tahap 1: Imputasi berbasis median grup.

    - nilai_agunan_kini & usia_bangunan_tahun diimputasi berdasarkan
    wilayah_agunan
    - pendapatan_bulanan diimputasi berdasarkan status_pekerjaan
    """

    def __init__(self):
        self.group_specs = {
            "nilai_agunan_kini": "wilayah_agunan",
            "usia_bangunan_tahun": "wilayah_agunan",
            "pendapatan_bulanan": "status_pekerjaan",
        }

    def fit(self, X, y=None):
        self.group_medians_ = {}
        self.global_medians_ = {}
        for target_col, group_col in self.group_specs.items():
            self.group_medians_[target_col] = X.groupby(group_col)[
                target_col
            ].median()
            self.global_medians_[target_col] = X[target_col].median()
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        for target_col, group_col in self.group_specs.items():
            missing = X[target_col].isna()
            if missing.any():
                filled = X.loc[missing, group_col].map(
                    self.group_medians_[target_col]
                )
                filled = filled.fillna(self.global_medians_[target_col])
                X.loc[missing, target_col] = filled
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


# ------------------------------------------------------------------
# STAGE 2: Feature Engineering Rasio Finansial
# ------------------------------------------------------------------
class RatioFeatureEngineer(BaseEstimator, TransformerMixin):
    """Tahap 2: Feature Engineering Rasio Finansial (LTV, DTI, &

    persentase_penurunan).
    """

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        X["ltv_awal"] = X["plafon_pinjaman"] / X["nilai_agunan_awal"]
        X["ltv_kini"] = X["plafon_pinjaman"] / X["nilai_agunan_kini"]
        X["dti_ratio"] = (
            X["plafon_pinjaman"] / X["tenor_bulan"]
        ) / X["pendapatan_bulanan"]

        # Menambahkan persentase penurunan
        selisih_agunan = X["nilai_agunan_awal"] - X["nilai_agunan_kini"]
        X["persentase_penurunan"] = selisih_agunan / X["nilai_agunan_awal"]

        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(
            self.feature_names_in_.tolist()
            + ["ltv_awal", "ltv_kini", "dti_ratio", "persentase_penurunan"]
        )