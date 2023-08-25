def load_dataset():
    import pandas as pd
    df = pd.read_csv(r"diabetes_prediction_dataset.csv")
    return df


