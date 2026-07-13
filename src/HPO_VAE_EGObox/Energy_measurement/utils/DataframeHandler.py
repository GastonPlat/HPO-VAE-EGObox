import pandas as pd
import numpy as np

def get_df(full_path_csv: str, n_header: int | bool) -> pd.DataFrame:
    df = pd.read_csv(filepath_or_buffer=full_path_csv, header=n_header)
    return(df)

def add_column_df_csv(csv: str, loc, columnheader, values) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df.insert(loc=loc, column=columnheader, value=values)

def merge_dfs_row(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    try: 
        return pd.concat([df1, df2], axis = 0)
    except:
        print("missing good format to concatenate your dataframes")

def merge_dfs_col(df1: pd.DataFrame, df2: pd.DataFrame)-> pd.DataFrame:
    try: 
        return pd.concat([df1, df2], axis = 1)
    except:
        print("missing good format to concatenate your dataframes")

def convert_to_dataframe():
    pass

def get_last_line(df: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        last_row = df.tail(n=1)
        return last_row
    elif isinstance(df, np.ndarray):
        last_row = df[-1]
        return last_row
    else:
        return f"Your input data {df} is not a pandas.DataFrame or a numpy.ndarray"
    
def save_dataframe_to_csv(df: pd.DataFrame, filename: str, index: bool = False):
    """
    Save a pandas DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    filename : str
        The output CSV filename (e.g., 'data.csv').
    index : bool, optional
        Whether to write row indices. Default is False.
    """
    df.to_csv(filename, index=index)




