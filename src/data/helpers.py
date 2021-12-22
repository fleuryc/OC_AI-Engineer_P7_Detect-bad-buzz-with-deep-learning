"""Helper functions, not project specific."""

import io
import logging
import os
import zipfile

import numpy as np
import pandas as pd
import requests


def download_extract_zip(
    zip_file_url: str,
    files_names: tuple[str],
    target_path: str,
) -> None:
    """
    Download Zip from url and extract content files to local path.

    - Check if content files already exist.
        - If they all exist, return.
        - If not, download zip file and extract content files.

    Args:
        zip_file_url: Url of zip file to download.
        files_names: List of file names to extract from zip.
        target_path: Path to extract zip contents to.

    Returns:
        None
    """
    # We must NOT download and extract zip file by default.
    must_download: bool = False

    for file in files_names:
        # Check if content files exist
        file_path = os.path.join(target_path, file)
        if not os.path.exists(file_path):
            # If at least one file does not exist, we must download zip file
            must_download = True
            break

    # If all files already exist, return
    if not must_download:
        logging.info("All files already exist in %s", target_path)
        return

    # Download zip file
    r = requests.get(zip_file_url)
    if r.status_code != 200:
        raise ValueError(f"Failed to download {zip_file_url}")

    # Check if zip file is OK
    z = zipfile.ZipFile(io.BytesIO(r.content))
    if z.testzip() is not None:
        raise ValueError(f"Failed to extract {zip_file_url}")

    # Check if content path exists
    if not os.path.exists(target_path):
        logging.info("Creating %s", target_path)
        os.makedirs(target_path)

    # Extract files from zip
    logging.info("Extracting %s to %s", zip_file_url, target_path)
    z.extractall(target_path)
    logging.info("Extracted %s to %s", zip_file_url, target_path)


def reduce_dataframe_memory_usage(
    df: pd.DataFrame,
    high_precision: bool = False,
) -> pd.DataFrame:
    """
    Iterate through all the columns of a dataframe and modify the data type to
    reduce memory usage.

    Args:
        df (pd.DataFrame): dataframe to reduce memory usage.
        high_precision (bool): If True, use 64-bit floats instead of 32-bit

    Returns:
        pd.DataFrame: dataframe with reduced memory usage.
    """
    start_mem = round(df.memory_usage().sum() / 1024 ** 2, 2)
    logging.info("Memory usage of dataframe is %d MB", start_mem)

    # Iterate through columns
    for col in df.columns:
        if df[col].dtype == "object":
            # "object" dtype
            if df[col].nunique() < max(100, df.shape[0] / 100):
                # If number of unique values is less than max(100, 1%)
                df[col] = df[col].astype("category")
            else:
                # If number of unique values is greater than max(100, 1%)
                df[col] = df[col].astype("string")

        elif str(df[col].dtype)[:3] == "int":
            # "int" dtype
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                df[col] = df[col].astype("UInt8")
            elif c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype("Int8")
            elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                df[col] = df[col].astype("UInt16")
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype("Int16")
            elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                df[col] = df[col].astype("UInt32")
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype("Int32")
            elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                df[col] = df[col].astype("UInt64")
            else:
                df[col] = df[col].astype("Int64")

        elif str(df[col].dtype)[:5] == "float":
            # "float" dtype
            c_min = df[col].min()
            c_max = df[col].max()
            if (
                not high_precision
                and c_min > np.finfo(np.float32).min
                and c_max < np.finfo(np.float32).max
            ):
                df[col] = df[col].astype("float32")
            else:
                df[col] = df[col].astype("float64")

    end_mem = round(df.memory_usage().sum() / 1024 ** 2, 2)
    logging.info("Memory usage after optimization is %d MB", end_mem)
    if start_mem > 0:
        logging.info(
            "Decreased by %d %%", round(100 * (start_mem - end_mem) / start_mem)
        )

    return df
