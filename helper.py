import os.path
import pandas as pd


def load_merged_data(only_turned_off=False, drop_na=True):
    base_path = os.path.abspath("")
    data_directory = os.path.join(base_path, "data")
    df = pd.read_csv(os.path.join(data_directory, 'merged.csv'), sep=',', parse_dates=['date', 'refuellingDateActual'])
    df = df[df['refuellingChargingDegreeActual'] >= 14.5]

    df['day'] = df['date'].dt.day

    if only_turned_off:
        df = df[df['Ausgeschaltet'] == 1]

    if drop_na:
        df = df.dropna()

    return df


def split_train_val_test(df, train, val):
    df = df.drop(columns=['date', 'operationPhase', 'operationPhaseNumeric', 'refuellingDateActual', 'refuellingChargingDegreeActual'])
    df = df.sample(frac=1).reset_index(drop=True)
    train_val_critical_day = int(30 * train)
    val_test_critical_day = int(30 * (train + val))

    train_df = df[df['day'] <= train_val_critical_day]
    val_df = df[(df['day'] > train_val_critical_day) & (df['day'] <= val_test_critical_day)]
    test_df = df[df['day'] > val_test_critical_day]

    train_df = train_df.drop(columns=['day'])
    val_df = val_df.drop(columns=['day'])
    test_df = test_df.drop(columns=['day'])
    return train_df, val_df, test_df


def split_xy(df, sample_size=None):
    df = df.dropna()

    if sample_size is not None:
        df = df.sample(sample_size)

    x = df.drop(columns=['refuellingTimePointActual'])
    y = df['refuellingTimePointActual']
    return x, y