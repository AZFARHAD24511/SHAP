import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

st.title("📊 SHAP Analysis - Direct UCI Dataset Load")

def load_data():
    bank_marketing = fetch_ucirepo(id=222)
    X = pd.DataFrame(bank_marketing.data.features, columns=bank_marketing.variables['name'][:-1])
    y = bank_marketing.data.targets.iloc[:, 0]  # درست دسترسی به ستون 'y'
    
    df = pd.concat([X, y], axis=1)
    # df = df.dropna()
    return df


df = load_data()
st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
st.write(df.head())

sample_n = st.slider("Select sample size", min_value=100, max_value=len(df), value=500, step=100)

if st.button("Run SHAP Analysis"):
    with st.spinner("Running analysis..."):
        if 'y' not in df.columns:
            st.error("❌ Column 'y' not found.")
            st.stop()

        X = df.drop(columns=['y'])
        y = df['y'].map({'no': 0, 'yes': 1})

        X_enc = X.copy()
        for col in X_enc.select_dtypes(include=['object']).columns:
            X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))

        mi = mutual_info_classif(X_enc, y, discrete_features='auto', random_state=42)
        mi_series = pd.Series(mi, index=X_enc.columns).sort_values(ascending=False)

        st.subheader("Top 10 Features by Mutual Information")
        st.write(mi_series.head(10))

        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_enc, y)

        explainer = shap.Explainer(rf, X_enc)
        X_small = X_enc.sample(n=sample_n, random_state=42)
        shap_result = explainer(X_small)

        st.subheader("SHAP Summary Plot (Bar)")
        shap.summary_plot(shap_result.values[..., 1], X_small, plot_type='bar')
        st.pyplot(plt.gcf())

        st.subheader("SHAP Summary Plot (Beeswarm)")
        shap.summary_plot(shap_result.values[..., 1], X_small)
        st.pyplot(plt.gcf())

        st.subheader("SHAP Dependence Plot for 'duration'")
        if 'duration' in X_small.columns:
            shap.dependence_plot('duration', shap_result.values[..., 1], X_small)
            st.pyplot(plt.gcf())
        else:
            st.warning("'duration' column not found.")

st.markdown("""
---
<p style='text-align:center; font-size:small;'>
This software was developed by Dr. A.Farhadi, PhD in Econometrics and Data Science.<br>
Dataset: Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306<br>
For any personal or commercial use, please cite the author.
</p>
""", unsafe_allow_html=True)
