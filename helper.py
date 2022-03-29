import pandas as pd


def find_gaps(df, Date_col_name="DateTime"):
    from datetime import datetime, timedelta
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    deltas = df['DateTime'].diff()[1:]
    gaps = deltas[deltas > timedelta(minutes=30)]
    # Print results
    print(f'{len(gaps)} gaps with median gap duration: {gaps.median()}')
    print(gaps)
