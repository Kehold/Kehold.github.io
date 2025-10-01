import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Petrophysics Demo App")

# Upload LAS or CSV data
uploaded_file = st.file_uploader("Upload Well Log Data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # Select column to plot
    log = st.selectbox("Select log to plot:", df.columns)

    fig, ax = plt.subplots()
    ax.plot(df[log], df["Depth"])
    ax.set_xlabel(log)
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    st.pyplot(fig)
