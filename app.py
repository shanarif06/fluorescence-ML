import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import io

# Load BOTH models
lr_model = joblib.load("fluorescence_model.pkl")
rf_model = joblib.load("rf_model.pkl")

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

# Header
st.markdown("""
<div style='text-align:center; padding:20px; background-color:#f5f7ff; border-radius:12px;'>
    <h1 style='color:#4F46E5;'>📷 Fluorescence Analyzer</h1>
    <p style='font-size:18px; color:gray;'>Machine Learning-based concentration prediction tool</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# Layout
col1, col2 = st.columns([1,1])

# ---------- LEFT PANEL ----------
with col1:
    st.markdown("""
    <div style='background-color:#f9fafb; padding:20px; border-radius:12px;'>
    <h4>📊 Input Data</h4>
    </div>
    """, unsafe_allow_html=True)

    user_input = st.text_input("Enter intensity values (comma separated, e.g., 4000,5000,5500)")

    model_choice = st.selectbox(
        "Select Model",
        ["Linear Regression", "Random Forest"]
    )

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

            if model_choice == "Linear Regression":
                predictions = lr_model.predict(np.array(values).reshape(-1,1))
            else:
                predictions = rf_model.predict(np.array(values).reshape(-1,1))

            st.success("Predictions:")
            for i, p in zip(values, predictions):
                st.write(f"Intensity {i} → {p:.2f} ppm")

        except:
            st.error("Please enter valid numbers separated by commas")

# ---------- GRAPH MODE SELECTOR ----------
graph_mode = st.selectbox(
    "Select Visualization Mode",
    ["Linear Regression", "Random Forest", "Compare Both"]
)

# ---------- MODEL COMPARISON ----------
st.write("---")
st.markdown("### 📊 Model Comparison")

# Predictions
y_lr = lr_model.predict(intensity_data.reshape(-1,1))
y_rf = rf_model.predict(intensity_data.reshape(-1,1))

# Metrics
r2_lr = r2_score(concentration_data, y_lr)
r2_rf = r2_score(concentration_data, y_rf)

rmse_lr = np.sqrt(mean_squared_error(concentration_data, y_lr))
rmse_rf = np.sqrt(mean_squared_error(concentration_data, y_rf))

mae_lr = mean_absolute_error(concentration_data, y_lr)
mae_rf = mean_absolute_error(concentration_data, y_rf)

# Conditional metrics display
if graph_mode == "Linear Regression":
    st.write(f"🔵 Linear Regression → R²: {r2_lr:.4f}, RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.2f}")

elif graph_mode == "Random Forest":
    st.write(f"🟠 Random Forest → R²: {r2_rf:.4f}, RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}")

else:
    st.write(f"🔵 Linear Regression → R²: {r2_lr:.4f}, RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.2f}")
    st.write(f"🟠 Random Forest → R²: {r2_rf:.4f}, RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}")

# ---------- GRAPH ----------
st.write("---")
st.markdown("### 📈 Model Visualization")

fig, ax = plt.subplots()
sorted_idx = np.argsort(intensity_data)

# Experimental data
ax.scatter(intensity_data, concentration_data, s=50, label="Experimental Data")

if graph_mode == "Linear Regression":
    ax.plot(intensity_data[sorted_idx], y_lr[sorted_idx], linewidth=2, label="Linear Regression")

elif graph_mode == "Random Forest":
    ax.scatter(intensity_data, y_rf, marker='x', s=80, label="Random Forest")

else:
    ax.plot(intensity_data[sorted_idx], y_lr[sorted_idx], linewidth=2, label="Linear Regression")
    ax.scatter(intensity_data, y_rf, marker='x', s=80, label="Random Forest")

# Labels
ax.set_xlabel("Fluorescence Intensity", fontsize=18, fontweight='bold', family='Times New Roman')
ax.set_ylabel("Concentration (ppm)", fontsize=18, fontweight='bold', family='Times New Roman')
ax.set_title("Model Visualization", fontsize=16, fontweight='bold', family='Times New Roman')

ax.grid(alpha=0.3)
ax.legend(frameon=False)

st.pyplot(fig)

# Download comparison graph
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
buf.seek(0)

st.download_button(
    label="📥 Download Visualization Graph",
    data=buf,
    file_name="model_visualization.png",
    mime="image/png"
)

# ---------- STEP 1: NEW SECTION ----------
st.write("---")
st.markdown("### 📊 Model Evaluation (Actual vs Predicted)")

# ---------- STEP 2: EVALUATION GRAPH ----------
fig2, ax2 = plt.subplots()

min_val = min(concentration_data)
max_val = max(concentration_data)

# Perfect line
ax2.plot([min_val, max_val],
         [min_val, max_val],
         'k--',
         linewidth=2,
         label="Perfect Prediction")

if graph_mode == "Linear Regression":
    ax2.scatter(concentration_data, y_lr, s=60, label="Linear Regression")

elif graph_mode == "Random Forest":
    ax2.scatter(concentration_data, y_rf, marker='x', s=80, label="Random Forest")

else:
    ax2.scatter(concentration_data, y_lr, s=60, label="Linear Regression")
    ax2.scatter(concentration_data, y_rf, marker='x', s=80, label="Random Forest")

# Labels
ax2.set_xlabel("Actual Concentration", fontsize=18, fontweight='bold', family='Times New Roman')
ax2.set_ylabel("Predicted Concentration", fontsize=18, fontweight='bold', family='Times New Roman')
ax2.set_title("Actual vs Predicted", fontsize=16, fontweight='bold', family='Times New Roman')
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Times New Roman')
    label.set_fontsize(16)
    label.set_fontweight('bold')
# Styling
for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
    label.set_fontsize(16)
    label.set_fontname('Times New Roman')
    label.set_fontweight('bold')

ax2.grid(alpha=0.3)
ax2.legend(frameon=False)

st.pyplot(fig2)

# ---------- STEP 3: DOWNLOAD ----------
buf2 = io.BytesIO()
fig2.savefig(buf2, format="png", dpi=300, bbox_inches='tight')
buf2.seek(0)

st.download_button(
    label="📥 Download Evaluation Graph",
    data=buf2,
    file_name="actual_vs_predicted.png",
    mime="image/png"
) 
