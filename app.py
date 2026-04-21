import streamlit as st
import pandas as pd
import subprocess
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="COVID-19 Forecasting", layout="centered")

st.title("🦠 COVID-19 Case Forecasting")
st.write("Deep Learning Ensemble Model")

st.sidebar.header("Controls")
run_pipeline = st.sidebar.button("Run Forecast")

if run_pipeline:
    with st.spinner("Running forecasting pipeline..."):
        subprocess.run(
            ["python", "main_pipeline.py"],
            check=True
        )
    st.success("Pipeline execution completed!")

results_path = "results/model_results.csv"

if os.path.exists(results_path):
    df = pd.read_csv(results_path)

    st.subheader("📊 Model Performance")
    st.dataframe(df)

    st.subheader("📉 RMSE Comparison")
    fig, ax = plt.subplots()
    ax.bar(df["model"], df["rmse"])
    ax.set_ylabel("RMSE")
    st.pyplot(fig)
else:
    st.info("Run the forecast to generate results.")
