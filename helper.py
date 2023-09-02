import os.path
import pandas as pd


def load_merged_data(only_turned_off=False, only_standard_refuelling=True, drop_na=True):
    base_path = os.path.abspath("")
    data_directory = os.path.join(base_path, "data")
    # read as pickle
    df = pd.read_pickle(os.path.join(data_directory, 'merged.pkl'))

    if only_standard_refuelling:
        df = df[df['refuellingChargingDegreeActual'] >= 14.5]

    if only_turned_off:
        df = df[df['Ausgeschaltet'] == 1]

    if drop_na:
        df = df.dropna()

    return df


def split_train_val_test(df, train, val):

    df['day'] = df['date'].dt.day

    # split by day

    df = df.sort_values(by=['day'])
    df = df.drop(columns=['day'])

    n = len(df)
    train_df = df.iloc[:int(n * train)]
    val_df = df.iloc[int(n * train):int(n * (train + val))]
    test_df = df.iloc[int(n * (train + val)):]

    return train_df, val_df, test_df


def split_xy(df, input_columns, sample_size=None):
    if sample_size is not None:
        df = df.sample(sample_size)

    x = df[input_columns]
    y = df['refuellingTimePointActual']
    return x, y
