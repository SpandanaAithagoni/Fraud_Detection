import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(
    "attention_model.keras"
)

st.title("Fraud Detection Dashboard")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(
        uploaded_file
    )

    st.write(df.head())

    X = df.drop(
        "Class",
        axis=1,
        errors="ignore"
    )

    X = X.values

    seq_len = 5

    sequences = []

    for i in range(
        len(X)-seq_len
    ):
        sequences.append(
            X[i:i+seq_len]
        )

    sequences = np.array(
        sequences
    )

    pred = model.predict(
        sequences
    )

    st.subheader(
        "Fraud Probability"
    )

    st.write(pred)

    high_risk = np.where(
        pred.flatten()>0.5
    )[0]

    st.subheader(
        "High Risk Transactions"
    )

    st.write(high_risk)