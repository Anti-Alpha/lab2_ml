import logging
import pickle
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd


def train_test_split(
    df: pd.DataFrame,
    test_size: Union[float, int] = 0.25,
    random_state: Union[int, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if random_state is not None:
        np.random.seed(random_state)

    total = len(df)
    n = int(total * test_size) if isinstance(test_size, float) else test_size

    idx = np.random.permutation(total)
    return df.iloc[idx[n:]], df.iloc[idx[:n]]


def unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def add_batches(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["batch_name"] = "unset"
    num = cfg["n_batches"]
    size = len(out) // num
    col = out.columns.get_loc("batch_name")

    for i in range(num):
        end = None if i == num - 1 else size * (i + 1)
        out.iloc[i * size:end, col] = str(i)

    return out


def filter_batches(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    return df[df["batch_name"].isin(cfg["batch_names_select"])].copy()


def process_data(dir_path: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = [f"{dir_path}/cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)]
    test_file = f"{dir_path}/cifar-10-batches-py/test_batch"

    X, y = [], []
    for file in parts:
        data = unpickle(file)
        X.append(data[b"data"])
        y += data[b"labels"]

    X_train = np.vstack(X).reshape(-1, 3, 32, 32).astype("float32") / 255.0
    y_train = np.array(y)

    test = unpickle(test_file)
    X_test = test[b"data"].reshape(-1, 3, 32, 32).astype("float32") / 255.0
    y_test = np.array(test[b"labels"])

    df_train = pd.DataFrame({"image": list(X_train), "label": y_train})
    df_test = pd.DataFrame({"image": list(X_test), "label": y_test})

    df_train = add_batches(df_train, cfg)
    logging.info(f"Batches created: {cfg['n_batches']}")

    df_train = filter_batches(df_train, cfg)
    logging.info(f"Selected batches: {cfg['batch_names_select']}")

    df_train, df_val = train_test_split(df_train, test_size=cfg.get("val_size", 0.2), random_state=cfg.get("random_state", 42))
    logging.info(f"Split sizes - train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

    return df_train, df_val, df_test