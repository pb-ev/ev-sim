import pandas as pd


def convert_time(time_str):
    time = pd.to_datetime(time_str, format="%I:%M %p").strftime("%H:%M")
    return int(time.split(":")[0])
