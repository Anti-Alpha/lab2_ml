import logging
import pickle
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd

def train_test_split(data: pd.DataFrame, test_size: Union[float, int] = 0.25, random_state: Union[int, None] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if random_state is not None:
        np.random.seed(random_state)
    n = len(data)
    if isinstance(test_size, float):
        test_size = int(n * test_size)
    idx = np.random.permutation(n)
    return data.iloc[idx[test_size:]], data.iloc[idx[:test_size]]

def unpickle(file):
    with open(file, "rb") as f:
        return pickle.load(f, encoding="bytes")

def assign_batches(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df_ = df.copy(deep=True)
    df_["batch_name"] = "not_set"
    n = cfg["n_batches"]
    s = len(df_) // n
    i = 0
    for b in range(n):
        loc = df_.columns.get_loc("batch_name")
        if isinstance(loc, int):
            if b == n - 1:
                df_.iloc[i:, loc] = str(b)
            else:
                df_.iloc[i:i+s, loc] = str(b)
        else:
            raise TypeError("Expected one 'batch_name' column")
        i += s
    return df_

def select_batches(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    return df[df["batch_name"].isin(cfg["batch_names_select"])].copy(deep=True)

def process_data(dir: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    batches = [f"{dir}/cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)]
    test_path = f"{dir}/cifar-10-batches-py/test_batch"
    data, labels = [], []
    for b in batches:
        d = unpickle(b)
        data.append(d[b"data"])
        labels.extend(d[b"labels"])
    x = np.vstack(data).reshape(-1, 3, 32, 32).astype("float32") / 255.0
    y = np.array(labels)
    test_d = unpickle(test_path)
    xt = test_d[b"data"].reshape(-1, 3, 32, 32).astype("float32") / 255.0
    yt = np.array(test_d[b"labels"])
    df_train = pd.DataFrame({"image": list(x), "label": y})
    df_test = pd.DataFrame({"image": list(xt), "label": yt})
    df_train = assign_batches(df_train, cfg)
    logging.info(f"Split train dataset in {cfg['n_batches']} batches")
    df_train = select_batches(df_train, cfg)
    logging.info(f"Batches {cfg['batch_names_select']} selected from train dataset")
    df_train, df_val = train_test_split(df_train, test_size=cfg.get("val_size", 0.2), random_state=cfg.get("random_state", 42))
    logging.info(f"Prepared 3 data splits: train, size: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
    return df_train, df_val, df_test