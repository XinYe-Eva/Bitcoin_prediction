import numpy as np
from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


class DataProcessor(object):
    def __init__(self, symbol: str, feature_builder: Optional[Callable] = None):
        """Load the dataframe of a given symbol"""
        df_ob = pd.read_pickle(f"Data/{symbol}_Orderbook.pkl", compression="gzip")
        self._df_klines = pd.read_pickle(f"Data/{symbol}_Kline.pkl", compression="gzip")
        self._df_klines["Return"] *= 100
        self._labels = self._df_klines["Return"].resample("30T").std().shift(-1)
        self._symbol = symbol
        if feature_builder is not None:
            df_ob = feature_builder(df_ob)
        features_index = pd.date_range(df_ob.index[0], df_ob.index[-1], freq="30T")
        features_columns = list()
        for i in range(60):
            features_columns.extend([f"{col}_{i}" for col in df_ob.columns])
        self._df_features = pd.DataFrame(
            df_ob.values.reshape(len(features_index), -1),
            index=features_index,
            columns=features_columns,
        )
        self._df_features.index = features_index
        self._labels = self._labels[self._labels.index.isin(self._df_features.index)]

    def plot(self, figsize: Tuple[float] = (10, 10), pad: float = 3):
        """Plot some properties of the market data"""
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize)
        self._df_klines["Close"].plot(
            grid=True, ax=axes[0], title=f"{self._symbol} price"
        )
        self._labels.plot(grid=True, ax=axes[1], title=f"{self._symbol} Volatility")
        hist_volaility = self._labels.hist(bins=100, ax=axes[2])
        hist_volaility.set_xlim([-0.01, 0.01])
        hist_volaility.set_xticks(np.arange(0, 0.3, step=0.02))
        hist_volaility.set_title(f"Distribution of {self._symbol} Volatility")
        fig.tight_layout(pad=pad)

    def split_train_val_test(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        shuffle: bool = False,
    ) -> Tuple[pd.DataFrame]:
        """Split the data into train, validation and test data"""
        if train_ratio + val_ratio + test_ratio != 1:
            raise ValueError("The sum of the splitting ratio should be 1")
        df_features = self._df_features.copy()
        df_features["Labels"] = self._labels
        if shuffle:
            df_features = df_features.sample(frac=1)
        train_sample_size = int(len(df_features) * train_ratio)
        val_sample_size = int(len(df_features) * val_ratio)
        df_train = df_features.iloc[:train_sample_size]
        df_val = df_features.iloc[
            train_sample_size : train_sample_size + val_sample_size
        ]
        df_test = df_features.iloc[train_sample_size + val_sample_size :]
        features = [col for col in df_train.columns[:-1] if col != "Labels"]
        return (
            df_train[features],
            df_train[["Labels"]],
            df_val[features],
            df_val[["Labels"]],
            df_test[features],
            df_test[["Labels"]],
        )

    @staticmethod
    def evaluate(
        y_true: pd.Series,
        y_pred: Union[np.array, pd.Series],
        figsize: Tuple[float] = (10, 5),
    ):
        y_true = y_true.iloc[len(y_true) - len(y_pred) :]
        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred.flatten(), index=y_true.index, name="Pred")
        df_plot = pd.merge(y_pred, y_true, left_index=True, right_index=True)
        df_plot.columns = ["Pred", "Labels"]
        df_plot.plot(figsize=figsize, grid=True)
        return {
            "MSE": mean_squared_error(y_true, y_pred),
            "R^2": r2_score(y_true, y_pred),
        }

    def __repr__(self):
        return f"Data Processor for {self._symbol}"
