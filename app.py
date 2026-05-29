import streamlit as st
import numpy as np
import joblib
import json

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="centered",
)

# ── Load model artifacts ─────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("model/mlp_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    with open("model/metrics.json") as f:
        meta = json.load(f)
    return model, scaler, meta

model, scaler, meta = load_artifacts()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🌸 Iris Species Classifier")
st.markdown(
    f"MLP Neural Network · **{meta['hidden_layers']}** hidden layers · "
    f"Test accuracy **{meta['accuracy']*100:.1f}%**"
)
st.divider()

st.subheader("Enter flower measurements")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width  = st.slider("Sepal width (cm)",  2.0, 4.5, 3.0, 0.1)
with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0, 0.1)
    petal_width  = st.slider("Petal width (cm)",  0.1, 2.5, 1.2, 0.1)

st.divider()

if st.button("🔍 Predict Species", use_container_width=True, type="primary"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)

    prediction   = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    species = meta["target_names"][prediction]
    emoji_map = {"setosa": "🌼", "versicolor": "💐", "virginica": "🌺"}

    st.success(f"### {emoji_map.get(species, '🌸')} Predicted: **{species.capitalize()}**")

    st.subheader("Confidence scores")
    for name, prob in zip(meta["target_names"], probabilities):
        st.progress(float(prob), text=f"{name.capitalize()}: {prob*100:.1f}%")

    with st.expander("ℹ️ Input features"):
        st.json({
            "sepal_length": sepal_length,
            "sepal_width":  sepal_width,
            "petal_length": petal_length,
            "petal_width":  petal_width,
        })

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Model: MLPClassifier · Trained on {meta['train_size']} samples · "
    f"Validated on {meta['test_size']} samples"
)
