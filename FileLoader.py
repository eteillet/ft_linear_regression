import pandas as pd

class FileLoader:

    def load(self, path):
        """
        displays a message specifying the dimensions of the dataset
        Return:
            - the dataset loaded as a pandas.DataFrame
        """
        try:
            df = pd.read_csv(path)
            print(f"Loading dataset of dimensions {df.shape[0]} rows x {df.shape[1]} columns")
        except Exception as e:
            print(e)
            return None
        return df

    def display(self, df, n):
        """
        display:
            - the first n rows of the dataset df if n is positive
            - the last n rows of the dataset df if n is negative
        """
        if not isinstance(df, pd.DataFrame) or not isinstance(n, int):
            print("Invalid argument type in following function: display(dataframe, integer)")
            return

        if n >= 0:
            print(df[:n])
        elif n < 0:
            print(df[n:])