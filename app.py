import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load model
model = joblib.load("fluorescence_model.pkl")

# Original dataset
intensity_data = np.array([
3644, 3662, 3735, 3847, 3963, 4201, 4302, 4469, 4797, 5296, 5379, 5845, 6314,
3642, 3661, 3738, 3846, 3967, 4203, 4398, 4468, 4795, 5299, 5375, 5849, 6318,
3641, 3663, 3734, 3848, 3965, 4209, 4308, 4473, 4798, 5292, 5382, 5851, 6322
])

concentration_data = np.array([
0, 0.1, 1, 20, 40, 60, 80, 100, 150, 200, 250, 300, 333,
0, 0.1, 1, 20, 40, 60, 80, 100, 150, 200, 250, 300, 333,
0, 0.1, 1, 20, 40, 60, 80, 100, 150, 200, 250, 300, 333
])

# Page config
st.set_page_config(page_title="Fluorescence Analyzer", layout="wide")

# ---------- HEADER ----------
st.markdown("""
<div style='text-align:center; padding:20px; background-color:#f5f7ff; border-radius:12px;'>
    <h1 style='color:#4F46E5;'>📷 Fluorescence Analyzer</h1>
<p style='font-size:18px; color:gray;'>Machine Learning-based concentration prediction tool</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# Model parameters
slope = model.coef_[0]
intercept = model.intercept_

# Layout columns
col1, col2 = st.columns([1,1])

# ---------- LEFT PANEL ----------
with col1:
    st.markdown("""
    <div style='background-color:#f9fafb; padding:20px; border-radius:12px;'>
    <h4>📊 Input Data</h4>
    </div>
    """, unsafe_allow_html=True)

    user_input = st.text_input("Enter intensity values (comma separated, e.g., 4000,5000,5500)")

    st.write("")

    predict_btn = st.button("🚀 Predict Concentration")

# ---------- RIGHT PANEL ----------
with col2:
    st.markdown("""
    <div style='background-color:#f9fafb; padding:20px; border-radius:12px;'>
    <h4>⚡ Prediction Output</h4>
    </div>
    """, unsafe_allow_html=True)

    if predict_btn:
        try:
            values = [float(i) for i in user_input.split(",")]
            predictions = model.predict(np.array(values).reshape(-1,1))

            st.success("Predictions:")
            for i, p in zip(values, predictions):
                st.write(f"Intensity {i} → {p:.2f} µM")

            # Equation
            st.info(f"Calibration Equation: C = {slope:.4f} × I + {intercept:.2f}")

            # R2
            y_pred_full = model.predict(intensity_data.reshape(-1,1))
            r2 = r2_score(concentration_data, y_pred_full)
            st.info(f"R²: {r2:.4f}")

        except:
            st.error("Please enter valid numbers separated by commas")

# ---------- FULL WIDTH GRAPH ----------
st.write("---")

import io

import io

import io

st.markdown("### 📈 Calibration Curve")

fig, ax = plt.subplots()

# Plot data
ax.scatter(intensity_data, concentration_data)

sorted_idx = np.argsort(intensity_data)
ax.plot(intensity_data[sorted_idx],
        model.predict(intensity_data.reshape(-1,1))[sorted_idx])

# Axis labels (UPDATED)
ax.set_xlabel("Fluorescence Intensity", fontsize=18, fontweight='bold', family='Times New Roman')
ax.set_ylabel("Concentration (ppm)", fontsize=18, fontweight='bold', family='Times New Roman')

# Title (optional)
ax.set_title("Calibration Curve", fontsize=16, fontweight='bold', family='Times New Roman')

# Tick styling (UPDATED)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(16)
    label.set_fontname('Times New Roman')
    label.set_fontweight('bold')

# No legend (as you requested)

st.pyplot(fig)

# Save graph (high quality)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300)
buf.seek(0)

st.download_button(
    label="📥 Download Calibration Graph",
    data=buf,
    file_name="calibration_curve.png",
    mime="image/png"
)