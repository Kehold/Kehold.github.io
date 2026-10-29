"""Production Intelligence Agents · Streamlit entry point.

Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Production Intelligence Agents", page_icon="🛢️", layout="wide")

pages = [
    st.Page("views/run.py", title="Run analysis", icon="▶️", default=True),
    st.Page("views/spatial.py", title="1 · Spatial and 3D", icon="📍"),
    st.Page("views/analogs.py", title="2 · Analog selection", icon="🔎"),
    st.Page("views/decline.py", title="3 · Decline curves", icon="📉"),
    st.Page("views/transformer.py", title="4 · Transformer forecast", icon="🧠"),
    st.Page("views/ensemble.py", title="5 · Ensemble and uncertainty", icon="🎯"),
    st.Page("views/report.py", title="Report and download", icon="📦"),
    st.Page("views/field.py", title="Field data and method", icon="🗂️"),
]
st.navigation(pages).run()
