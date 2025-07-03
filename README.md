# SHAP
## What is SHAP?

**SHAP (SHapley Additive exPlanations)** is a method to explain the output of any machine learning model. It is based on concepts from cooperative game theory, particularly the Shapley values, which were originally developed to fairly distribute payout among players depending on their contribution to the total gain.

### How SHAP Works

- SHAP assigns each feature an importance value for a particular prediction.
- It quantifies the contribution of each feature to pushing the prediction from the average prediction (baseline) to the actual prediction for a single instance.
- By aggregating these values across the dataset, SHAP helps to explain how each feature influences the model’s output both locally (for individual predictions) and globally (overall feature importance).

### Why Use SHAP?

- It provides **consistent and locally accurate explanations**.
- Works with any model (tree-based, linear, neural networks, etc.).
- Offers detailed insights into feature interactions and the direction and magnitude of effects.
- Helps build **trust** in machine learning models by making them interpretable.

### In This Project

I used SHAP to analyze the Bank Marketing model predictions. The plots included in this repository:

- **SHAP Summary Plot:** Shows overall feature importance and how feature values relate to their impact on the prediction.
- **SHAP Dependence Plot:** Displays how the effect of one feature depends on the value of another.
- **SHAP Value Sort Plot:** Visualizes individual prediction explanations sorted by their SHAP values.

These visualizations help users understand why the model makes certain predictions and which factors are the most influential.

---

If you want to learn more about SHAP, visit the official documentation:  
[https://shap.readthedocs.io/en/latest/](https://shap.readthedocs.io/en/latest/)




# Bank Marketing Dataset and Visualization App

I have used the **Bank Marketing** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) for my project.

Based on this dataset, I developed a user-friendly web application that allows users to explore the data and visualize outputs through interactive charts.

You can access the app here:  
[https://shapbanking.streamlit.app/](https://shapbanking.streamlit.app/)

---

## Project Files

- `SHAP_bank_marketing_app.py` — The main Streamlit application script for data loading, modeling, and visualization.
- `SHAP_Dependence_Plot.png` — A SHAP dependence plot showing feature interactions and effects.
- `SHAP_Summary_Plot.png` — A SHAP summary plot demonstrating feature importance across the dataset.
- `SHAP_value_sort.png` — A plot showing sorted SHAP values for model interpretability.

---

## Visualization Examples

### SHAP Dependence Plot
![SHAP Dependence Plot](SHAP_Dependence_Plot.png)

### SHAP Summary Plot
![SHAP Summary Plot](SHAP_Summary_Plot.png)

### SHAP Value Sort Plot
![SHAP Value Sort](SHAP_value_sort.png)

---

**Dataset Citation:**  
Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

---

Feel free to try the app and explore the data interactively!

