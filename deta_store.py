import streamlit as st
import pandas as pd
from deta import Deta

deta = Deta(st.secrets["API"]["DETA_KEY"])
db = deta.Base("topical_q_a")


def fetch_all_items(db):
    res = db.fetch()
    all_items = res.items

    # fetch until last is 'None'
    while res.last:
        res = db.fetch(last=res.last)
        all_items += res.items
        print(res)

    return all_items


def check_dataframe(data):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    return data


def convert_timestamp_to_time(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def app():
    st.set_page_config(layout="wide")
    st.title("Data Table")
    data = fetch_all_items(db)
    data = check_dataframe(data)
    data = data.drop(columns=["key"])
    data = convert_timestamp_to_time(data)
    if data is not None:
        st.dataframe(data)
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="q_data.csv",
            mime="text/csv",
        )
    else:
        st.warning("No data to display.")


app()
