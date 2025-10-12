import pandas as pd

def chk_types(df):
    """
    """
    dtypes = df.dtypes
    n_unique = df.nunique()
    return pd.DataFrame({'dtypes': dtypes, 'n_unique': n_unique})