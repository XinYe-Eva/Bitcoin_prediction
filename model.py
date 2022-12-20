from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
import tqdm


torch.manual_seed(0)


class TimeSeriesNeuroNetwork(nn.Module, BaseEstimator):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        model_type: str = "lstm",
        num_layers: int = 1,
        batch_size: int = 32,
        dropout: float = 0,
        loss_fun: Any = nn.MSELoss(reduce="mean"),
        device: Optional[str] = None,
        batch_normalize: bool = False,
    ):
        """A timeseries neuro network"""
        super().__init__()
        self._input_size = input_size
        self._loss_fun = loss_fun
        self._batch_size = batch_size
        self._batch_normalize = batch_normalize
        kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": True,
            "dropout": dropout,
        }
        if model_type == "lstm":
            self._ts_layer = nn.LSTM(**kwargs)
        elif model_type == "rnn":
            self._ts_layer = nn.RNN(**kwargs)
        elif model_type == "gru":
            self._ts_layer = nn.GRU(**kwargs)
        else:
            raise ValueError(
                f"Argument model_type only support lstm/rnn/gru, given {model_type}"
            )

        if device is None:
            self._device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(device)
        self._output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )
        self.to(self._device)
        self.train_loss = list()
        self.val_loss = list()

    def fit(
        self,
        train_X: Union[np.array, pd.DataFrame],
        train_y: Union[np.array, pd.DataFrame],
        val_X: Union[np.array, pd.DataFrame],
        val_y: Union[np.array, pd.DataFrame],
        optimizer: Optimizer,
        epochs: int = 5,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        class TimeSeriesDataset(Dataset):
            def __init__(
                self,
                X: np.array,
                y: np.array,
                input_size: int,
                device: Optional[torch.device] = None,
                batch_normalize: bool = True,
            ):
                X = X.reshape(-1, 60, input_size)
                if batch_normalize:
                    X = (X - X[:, [0]]) / X.std(axis=1).reshape(-1, 1, input_size)
                self.X = torch.from_numpy(X).float().to(device)
                self.y = torch.from_numpy(y).float().to(device)

            def __len__(self):
                return len(self.y)

            def __getitem__(self, index: int) -> Tuple[Tensor]:
                return self.X[index], self.y[index]

        if isinstance(train_X, pd.DataFrame):
            train_X = train_X.values
        if isinstance(train_y, pd.DataFrame):
            train_y = train_y.values
        if isinstance(val_X, pd.DataFrame):
            val_X = val_X.values
        if isinstance(val_y, pd.DataFrame):
            val_y = val_y.values
        ds_train = TimeSeriesDataset(
            train_X, train_y, self._input_size, self._device, self._batch_normalize
        )
        ds_val = TimeSeriesDataset(
            val_X, val_y, self._input_size, self._device, self._batch_normalize
        )
        for epoch in tqdm.tqdm(range(1, epochs + 1)) if verbose else range(epochs):
            # Training
            train_batch_loss = list()
            self.train()
            for X, y in DataLoader(
                ds_train, batch_size=self._batch_size, shuffle=False
            ):
                loss = self._loss_fun(y, self(X))
                train_batch_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            self.train_loss.append(np.mean(train_batch_loss))
            # Validation
            self.eval()
            val_batch_loss = list()
            with torch.no_grad():
                for X, y in DataLoader(
                    ds_val, batch_size=self._batch_size, shuffle=False
                ):
                    val_loss = self._loss_fun(y, self(X))
                    val_batch_loss.append(val_loss.item())
            self.val_loss.append(np.mean(val_batch_loss))
            # Output
            if verbose:
                print(
                    f"[{epoch}/{epochs}] Training loss: {round(self.train_loss[-1], 5)}\t"
                    f"Validation loss: {round(self.val_loss[-1], 5)}"
                )

    def plot_loss(self, **kwargs):
        plt.figure()
        df_loss = pd.DataFrame([self.train_loss, self.val_loss]).T
        df_loss.columns = ["Training loss", "Validation Loss"]
        df_loss.plot(grid=True, title="Losses", **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        return self._output_layer(self._ts_layer(X)[0][:, -1, :])

    def predict(self, X: pd.DataFrame) -> np.array:
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = X.reshape(-1, 60, self._input_size)
        if self._batch_normalize:
            X = (X - X[:, [0]]) / X.std(axis=1).reshape(-1, 1, self._input_size)
        X = torch.from_numpy(X)
        return self(X.float().to(self._device)).detach().cpu().numpy().flatten()
