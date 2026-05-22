import streamlit as st
import numpy as np
import joblib
import os
from PIL import Image

BASE_DIR = os.path.dirname(__file__)

st.set_page_config(
    page_title="Braille to Text",
    page_icon="⠃",
    layout="wide"
)

BRAILLE_MAP = {
    'a':'⠁','b':'⠃','c':'⠉','d':'⠙','e':'⠑','f':'⠋','g':'⠛','h':'⠓',
    'i':'⠊','j':'⠚','k':'⠅','l':'⠇','m':'⠍','n':'⠝','o':'⠕','p':'⠏',
    'q':'⠟','r':'⠗','s':'⠎','t':'⠞','u':'⠥','v':'⠧','w':'⠺','x':'⠭',
    'y':'⠽','z':'⠵'
}

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_all_models():
    svm = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
    rf  = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    le  = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    classes = np.load(os.path.join(MODELS_DIR, "classes.npy"), allow_pickle=True)

    from tensorflow.keras.models import load_model
    cnn = load_model(os.path.join(MODELS_DIR, "cnn_model.keras"))

    return svm, rf, cnn, le, classes

def preprocess(uploaded_file):
    img = Image.open(uploaded_file).convert("L")   # grayscale
    img = img.resize((64, 64))
    arr = np.array(img) / 255.0
    flat     = arr.reshape(1, -1)                  # for SVM / RF
    cnn_inp  = arr.reshape(1, 64, 64, 1)           # for CNN
    preview  = arr                                  # for display
    return flat, cnn_inp, preview


def predict(svm, rf, cnn, le, flat, cnn_inp):
    # SVM
    svm_char = svm.predict(flat)[0]

    # RF
    rf_char  = rf.predict(flat)[0]

    # CNN
    probs      = cnn.predict(cnn_inp, verbose=0)[0]
    cnn_idx    = int(np.argmax(probs))
    cnn_char   = le.inverse_transform([cnn_idx])[0]
    cnn_conf   = float(probs[cnn_idx])

    return svm_char, rf_char, cnn_char, cnn_conf, probs


def segment_grid(image_array, cell_size=64):
    """
    Grid-based segmentation — works perfectly on synthetic/clean Braille.
    image_array: grayscale numpy array, any size
    Returns: list of 64x64 cell arrays and list of (row, col) positions
    """
    h, w = image_array.shape
    cells = []
    positions = []
    
    rows = h // cell_size
    cols = w // cell_size
    
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r*cell_size, (r+1)*cell_size
            x1, x2 = c*cell_size, (c+1)*cell_size
            cell = image_array[y1:y2, x1:x2]
            
            # Skip blank cells
            if cell.std() > 5:  # has variation = has content
                cells.append(cell)
                positions.append((r, c))
    
    return cells, positions


# ── SIDEBAR ──────────────────────────────────────────────────
st.sidebar.title("⠃ Braille to Text")
st.sidebar.caption("Data Science Project")
page = st.sidebar.radio(
    "Navigate",
    ["🔤 Predict Character", "📊 Model Comparison", "📄 Document", "ℹ️ About"]
)
st.sidebar.divider()
st.sidebar.markdown("""
**Models trained on:**  
Braille Character Dataset  
1560 images · 26 classes  

**Best model:** CNN (83.7%)
""")

# ── LOAD MODELS ──────────────────────────────────────────────
try:
    svm, rf, cnn, le, classes = load_all_models()
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.info("Make sure the `models/` folder is next to `app.py` and contains all 5 files.")
    st.stop()

# ════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ════════════════════════════════════════════════════════════
if page == "🔤 Predict Character":
    st.title("Braille Character Recognition")
    st.caption("Upload a Braille character image — all 3 models will predict it.")

    uploaded = st.file_uploader(
        "Choose an image", type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        key="single"
    )

    if uploaded:
        flat, cnn_inp, preview = preprocess(uploaded)
        svm_char, rf_char, cnn_char, cnn_conf, probs = predict(
            svm, rf, cnn, le, flat, cnn_inp
        )

        col_img, col_results = st.columns([1, 2], gap="large")

        with col_img:
            st.subheader("Input image")
            st.image(preview, clamp=True, width='stretch')
            st.caption("Resized to 64×64 grayscale")

        with col_results:
            st.subheader("Predictions")

            # Majority vote
            all_preds = [str(svm_char).lower(), str(rf_char).lower(), str(cnn_char).lower()]
            from collections import Counter
            vote_counts = Counter(all_preds)
            majority_char, majority_count = vote_counts.most_common(1)[0]
            braille_sym = BRAILLE_MAP.get(majority_char, "?")

            if majority_count == 3:
                label = "✅ All 3 models agree"
                color = "#f0fff4"
                border = "#38a169"
                text_color = "#1a4731"
            elif majority_count == 2:
                agreeing = [m for m, p in zip(["SVM", "RF", "CNN"], all_preds) if p == majority_char]
                label = f"2/3 models agree ({' + '.join(agreeing)})"
                color = "#fffbeb"
                border = "#d97706"
                text_color = "#78350f"
            else:
                majority_char = str(cnn_char).lower()
                braille_sym = BRAILLE_MAP.get(majority_char, "?")
                label = "⚠️ All models disagree — showing CNN (most accurate)"
                color = "#fff5f5"
                border = "#e53e3e"
                text_color = "#742a2a"

            st.markdown(
                f"""
                <div style="background:{color};border:1.5px solid {border};
                border-radius:12px;padding:20px 24px;margin-bottom:16px">
                <div style="font-size:13px;color:{text_color};font-weight:500;
                margin-bottom:4px">{label}</div>
                <div style="font-size:52px;font-weight:700;color:{text_color};
                line-height:1">{majority_char.upper()}
                <span style="font-size:36px;margin-left:12px">{braille_sym}</span>
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(cnn_conf, text=f"CNN confidence: {cnn_conf*100:.1f}%")

            # SVM and RF side by side
            c1, c2 = st.columns(2)
            with c1:
                sym = BRAILLE_MAP.get(str(svm_char).lower(), "?")
                st.metric("SVM prediction", f"{str(svm_char).upper()}  {sym}")
                st.caption("Accuracy: 61.9%")
            with c2:
                sym = BRAILLE_MAP.get(str(rf_char).lower(), "?")
                st.metric("Random Forest prediction", f"{str(rf_char).upper()}  {sym}")
                st.caption("Accuracy: 69.9%")

        # Top-5 CNN probabilities
        st.divider()
        st.subheader("CNN — top 5 predictions")
        top5_idx  = np.argsort(probs)[::-1][:5]
        top5_chars = le.inverse_transform(top5_idx)
        top5_probs = probs[top5_idx]

        for char, prob in zip(top5_chars, top5_probs):
            sym = BRAILLE_MAP.get(str(char).lower(), "?")
            st.progress(float(prob),
                text=f"{str(char).upper()} {sym}  —  {prob*100:.1f}%")

    else:
        st.info("👆 Upload a Braille character image to get started.")
        st.markdown("""
        **Tips for best results:**
        - Use images from the same dataset you trained on
        - Single character per image, not a full word
        - Clear, well-lit image works best
        """)

# ════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("Model Comparison")
    st.caption("Evaluation metrics across all 3 trained models.")

    # Metrics table
    st.subheader("Overall metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### SVM")
        st.metric("Accuracy",  "61.9%")
        st.metric("Precision", "69.9%")
        st.metric("Recall",    "61.9%")
        st.metric("F1 Score",  "63.3%")

    with col2:
        st.markdown("### Random Forest")
        st.metric("Accuracy",  "69.9%")
        st.metric("Precision", "73.5%")
        st.metric("Recall",    "69.9%")
        st.metric("F1 Score",  "69.7%")

    with col3:
        st.markdown("### CNN ⭐")
        st.metric("Accuracy",  "83.7%")
        st.metric("Precision", "86.0%")
        st.metric("Recall",    "83.7%")
        st.metric("F1 Score",  "84.0%")

    st.divider()

    # Bar chart using streamlit native
    st.subheader("Visual comparison")
    import pandas as pd

    df = pd.DataFrame({
        "Metric":  ["Accuracy", "Precision", "Recall", "F1 Score"] * 3,
        "Score":   [
            0.619, 0.699, 0.619, 0.633,   # SVM
            0.699, 0.735, 0.699, 0.697,   # RF
            0.837, 0.860, 0.837, 0.840,   # CNN
        ],
        "Model": ["SVM"]*4 + ["Random Forest"]*4 + ["CNN"]*4
    })

    import altair as alt
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Model:N", axis=alt.Axis(labelAngle=0), title=None),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
        color=alt.Color("Model:N", scale=alt.Scale(
            domain=["SVM", "Random Forest", "CNN"],
            range=["#85B7EB", "#5DCAA5", "#534AB7"]
        )),
        column=alt.Column("Metric:N", title=None),
        tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".3f")]
    ).properties(width=120, height=300)

    st.altair_chart(chart)

    st.divider()
    st.subheader("Why CNN wins")
    st.markdown("""
    - **SVM** flattens the image to a 4096-dimensional vector — loses all spatial structure
    - **Random Forest** also works on flat pixels — no understanding of shape or position  
    - **CNN** uses convolutional filters that detect edges, curves, and dot patterns — exactly what Braille needs
    - Augmentation (rotation, zoom, shift) during CNN training added robustness, pushing accuracy from 76% → 83.7%

    **Limitations:**  
    Dataset has only 60 images per class. Real-world performance may be lower on photos taken 
    with different lighting, angles, or paper types. Future work: collect real scanned Braille 
    images to improve generalization.
    """)

    # Confusion matrix images if they exist
    cm_path = os.path.join(os.path.dirname(__file__), "confusion_matrices.png")
    comp_path = os.path.join(os.path.dirname(__file__), "model_comparison.png")

    if os.path.exists(comp_path):
        st.divider()
        st.subheader("Comparison chart (from training)")
        st.image(comp_path, width='stretch')

    if os.path.exists(cm_path):
        st.divider()
        st.subheader("Confusion matrices (from training)")
        st.image(cm_path, width='stretch')

# ════════════════════════════════════════════════════════════
# PAGE 3 — DOCUMENT
# ════════════════════════════════════════════════════════════
elif page == "📄 Document":
    st.title("Full Document Conversion")
    st.caption("Upload a clean Braille image with multiple characters. Each cell is classified individually.")

    st.info("💡 Try uploading one of the test images: `test_hello.png`, `test_braille.png`, or `test_abcde.png`")

    uploaded = st.file_uploader("Upload Braille document image", type=["png","jpg","jpeg"], key="doc")

    if uploaded:
        img = Image.open(uploaded).convert('L')
        img_array = np.array(img)

        st.image(img, caption="Uploaded image", use_container_width=True)

        cells, positions = segment_grid(img_array, cell_size=64)

        if len(cells) == 0:
            st.error("No cells detected. Make sure the image is made of 64×64 aligned Braille cells.")
        else:
            st.info(f"Detected {len(cells)} character cells")

            predicted_chars = []
            confidences = []

            for cell in cells:
                cell_norm = cell.astype('float32') / 255.0
                cell_input = cell_norm.reshape(1, 64, 64, 1)
                probs = cnn.predict(cell_input, verbose=0)[0]
                pred_idx = np.argmax(probs)
                pred_char = le.inverse_transform([pred_idx])[0]
                predicted_chars.append(str(pred_char).lower())
                confidences.append(float(probs[pred_idx]))

            result_text = ''.join(predicted_chars)

            st.subheader("Decoded text")
            braille_str = ' '.join(BRAILLE_MAP.get(c, '?') for c in result_text)
            st.markdown(f"## `{result_text.upper()}`")
            st.caption(f"Braille symbols: {braille_str}")

            avg_conf = np.mean(confidences) * 100
            st.metric("Average confidence", f"{avg_conf:.1f}%")

            st.subheader("Character breakdown")
            cols_per_row = 10
            for i in range(0, len(cells), cols_per_row):
                chunk_cells  = cells[i:i+cols_per_row]
                chunk_preds  = predicted_chars[i:i+cols_per_row]
                chunk_confs  = confidences[i:i+cols_per_row]

                cols = st.columns(len(chunk_cells))
                for j, (cell, pred, conf) in enumerate(zip(chunk_cells, chunk_preds, chunk_confs)):
                    with cols[j]:
                        st.image(cell, width=60)
                        color = "green" if conf > 0.8 else "orange" if conf > 0.5 else "red"
                        sym = BRAILLE_MAP.get(pred, '?')
                        st.markdown(
                            f"<p style='text-align:center;color:{color};font-weight:bold;font-size:13px'>"
                            f"{pred.upper()} {sym}</p>",
                            unsafe_allow_html=True
                        )
    else:
        st.markdown("""
        **How it works:**
        1. Upload a Braille image containing multiple characters side by side
        2. The image is split into 64×64 cells (one per character)
        3. Each cell is fed into the CNN classifier
        4. Results are shown character by character with confidence scores

        **Best results with:** synthetic/clean images generated from the same dataset.
        """)

# ════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("About this project")

    st.markdown("""
    ## Braille Character Recognition
    **A data science project comparing classical ML and deep learning for Braille OCR.**

    ---

    ### Dataset
    - **Source:** Braille Character Dataset (Kaggle — shanks0465)
    - **Size:** 1560 images, 26 classes (a–z)
    - **Images:** 64×64 grayscale, 60 images per class
    - **Split:** 80% train / 20% test, random_state=42

    ---

    ### Models trained
    | Model | Approach | Accuracy |
    |---|---|---|
    | SVM (RBF kernel) | Classical ML on flattened pixels | 61.9% |
    | Random Forest (100 trees) | Ensemble on flattened pixels | 69.9% |
    | CNN | Deep learning with augmentation | 83.7% |

    ---

    ### Pipeline
    1. Load images → resize to 64×64 → normalize to [0,1]
    2. EDA: class distribution, sample grid, pixel stats
    3. Train SVM and Random Forest on flattened pixel vectors
    4. Train CNN with data augmentation (rotation, zoom, shift)
    5. Evaluate all 3: accuracy, precision, recall, F1, confusion matrix
    6. Deploy best model (CNN) in this Streamlit app

    ---

    ### Tech stack
    `Python` · `TensorFlow/Keras` · `scikit-learn` · `OpenCV` · `Streamlit`

    ---

    ### Limitations & future work
    - 60 images per class is a small dataset — model may not generalize to real photos
    - Future: collect real scanned Braille images, try transfer learning (MobileNetV2)
    - Future: extend to full words and sentences, not just single characters
    """)