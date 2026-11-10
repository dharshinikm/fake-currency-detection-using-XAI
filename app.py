import os
import sys

# Ensure script directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import cv2
import numpy as np
import json
import base64
import io
import wave
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import pipeline module
import pipeline


# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Indian Currency Authentication (XAI)",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Academic & Professional Presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .metric-title {
        font-size: 0.82rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0f172a;
    }
    .stage-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 6px;
    }
    .badge-pass {
        background-color: #dcfce7;
        color: #15803d;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-fail {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def decode_base64_image(b64_str):
    """Decodes a base64 string or data-URI into a BGR numpy array."""
    if not b64_str:
        return None
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    try:
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def generate_audio_siren():
    """Generates a WAV audio byte stream for the counterfeit alarm."""
    try:
        wave_data, sr = pipeline.counterfeit_alarm(sr=22050, seconds=2.2)
        pcm = (np.clip(wave_data, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()
    except Exception:
        return None


def clean_for_json(data):
    """Prepares pipeline result for JSON serialization (removes large arrays)."""
    serializable = {}
    for k, v in data.items():
        if k in ("preprocessing", "gradcam", "anomaly"):
            continue
        serializable[k] = v

    if "gradcam" in data and isinstance(data["gradcam"], dict):
        serializable["gradcam_layer"] = data["gradcam"].get("layer_used", "")

    if "anomaly" in data and isinstance(data["anomaly"], dict):
        an_dict = dict(data["anomaly"])
        an_dict.pop("annotated", None)
        serializable["anomaly"] = an_dict

    return serializable


# =============================================================================
# SIDEBAR NAVIGATION & INFO
# =============================================================================

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/ee/Indian_Rupee_symbol.svg", width=55)
    st.title("Currency XAI")
    st.caption("Explainable AI Indian Currency Authentication")
    st.divider()

    st.markdown("### 📋 Pipeline Stages")
    st.markdown("""
    1. **Banknote Input & Acquisition**
    2. **Image Quality Gate Check**
    3. **Computer Vision Preprocessing**
    4. **Denomination CNN Detection**
    5. **Authentication CNN & Intaglio Analysis**
    5b. **Novelty & Play-Money OCR Check**
    6. **Security Feature Verification**
    6b. **XAI Baseline Comparison**
    7. **Suspicious Region Localisation**
    8. **Grad-CAM Attention Mapping**
    9. **Overall Review & Risk Synthesis**
    10. **Counterfeit Siren Alarm**
    11. **Downloadable XAI Report**
    """)
    st.divider()

    st.markdown("### 🤖 Trained Models")
    st.success("✅ Denomination CNN: `6 Classes`")
    st.success("✅ Authentication CNN: `Binary (Real/Fake)`")
    st.divider()

    auto_siren = st.checkbox("🔊 Auto-play Siren on Suspect Notes", value=True)
    st.caption("Screening aid demonstrating multi-modal explainable AI.")


# =============================================================================
# MAIN HEADER
# =============================================================================

st.markdown('<div class="main-title">💵 Indian Currency Authentication with Explainable AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Complete stage-by-stage inspection showing graphs, preprocessing transforms, deep-learning feature attention, and downloadable audit reports.</div>', unsafe_allow_html=True)


# =============================================================================
# STAGE 1: IMAGE ACQUISITION & INPUT
# =============================================================================

st.markdown('<div class="stage-header">Stage 1 · Banknote Image Acquisition</div>', unsafe_allow_html=True)

input_method = st.radio(
    "Choose Input Method:",
    ["📁 Upload Image File", "📷 Live Camera Capture", "🧪 Load Demo / Sample Note"],
    horizontal=True
)

uploaded_image = None

if input_method == "📁 Upload Image File":
    file = st.file_uploader("Select Indian Currency Image", type=["jpg", "jpeg", "png"])
    if file is not None:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif input_method == "📷 Live Camera Capture":
    cam_file = st.camera_input("Capture Currency Note from Camera")
    if cam_file is not None:
        file_bytes = np.asarray(bytearray(cam_file.read()), dtype=np.uint8)
        uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif input_method == "🧪 Load Demo / Sample Note":
    if os.path.exists("outputs/currency_result.json"):
        st.info("💡 A previously processed note is cached. Click below to load its full multi-stage analysis.")
        if st.button("Load Cached Demo Analysis Result", type="secondary"):
            with open("outputs/currency_result.json", "r", encoding="utf-8") as f:
                demo_data = json.load(f)
            # Reconstruct demo result format
            st.session_state["cached_demo_result"] = demo_data
            st.success("Loaded demo results! Scroll down to inspect all stage outputs.")
    else:
        st.warning("No previous demo result file found. Please upload an image.")

# Allow user to trigger analysis if an image was uploaded
if uploaded_image is not None:
    col_img1, col_img2 = st.columns([1, 2])
    with col_img1:
        st.image(cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB), caption="Input Currency Image", use_container_width=True)
    with col_img2:
        h, w, c = uploaded_image.shape
        st.write(f"**Resolution:** `{w} x {h} px`")
        st.write(f"**Color Channels:** `{c} (BGR format)`")
        st.write(f"**Aspect Ratio:** `{w / float(h):.2f}`")
        if st.button("🚀 Run Complete XAI Pipeline Analysis", type="primary", use_container_width=True):
            with st.spinner("Executing 11-stage Explainable AI Pipeline (Quality Gate, Denomination CNN, Authentication, Security Audit, Grad-CAM)..."):
                try:
                    result = pipeline.run_pipeline(uploaded_image)
                    st.session_state["analysis_result"] = result
                    st.session_state["analyzed_image"] = uploaded_image
                    st.session_state.pop("cached_demo_result", None)
                    st.success("Pipeline analysis completed successfully!")
                except Exception as e:
                    st.error(f"Pipeline Execution Error: {e}")
                    st.stop()


# Retrieve active result
active_result = None
is_cached_demo = False

if "analysis_result" in st.session_state:
    active_result = st.session_state["analysis_result"]
elif "cached_demo_result" in st.session_state:
    active_result = st.session_state["cached_demo_result"]
    is_cached_demo = True

if active_result is None:
    st.info("👆 Please upload an Indian currency note image or click 'Load Demo / Sample Note' above to begin.")
    st.stop()


# =============================================================================
# STAGE 10 ALERT BANNER & AUDIO SIREN (HIGH PRIORITY VERDICT)
# =============================================================================

st.divider()
verdict = str(active_result.get("verdict", "UNKNOWN")).upper()
status_color = active_result.get("status_color", "#ef4444" if "COUNTERFEIT" in verdict or "SUSPECT" in verdict else "#10b981")
denom_name = active_result.get("denomination", {}).get("denomination", "Indian Currency")
risk_score = active_result.get("risk_score", active_result.get("risk", {}).get("risk_score", 0))
confidence = active_result.get("confidence", active_result.get("risk", {}).get("confidence", 0))

is_suspect_or_fake = ("SUSPECT" in verdict) or ("COUNTERFEIT" in verdict) or active_result.get("trigger_alarm", False)

if is_suspect_or_fake:
    st.markdown(f"""
    <div style="background-color: #7f1d1d; color: white; padding: 20px 24px; border-radius: 10px; margin-bottom: 20px;">
        <div style="font-size: 26px; font-weight: 800; display: flex; align-items: center; gap: 10px;">
            ⚠️ {verdict} ALERT
        </div>
        <div style="margin-top: 8px; font-size: 16px; opacity: 0.95;">
            Denomination: <b>{denom_name}</b> &nbsp;|&nbsp; Composite Risk Score: <b>{risk_score}%</b> &nbsp;|&nbsp; System Confidence: <b>{confidence}%</b>
        </div>
        <div style="margin-top: 6px; font-size: 14px; color: #fca5a5;">
            <b>Advisory:</b> Do not accept this note. Critical security feature failures or counterfeit signatures detected.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if auto_siren:
        siren_bytes = generate_audio_siren()
        if siren_bytes:
            st.audio(siren_bytes, format="audio/wav", autoplay=True)
else:
    st.markdown(f"""
    <div style="background-color: #064e3b; color: white; padding: 20px 24px; border-radius: 10px; margin-bottom: 20px;">
        <div style="font-size: 26px; font-weight: 800; display: flex; align-items: center; gap: 10px;">
            ✔ {verdict} NOTE
        </div>
        <div style="margin-top: 8px; font-size: 16px; opacity: 0.95;">
            Denomination: <b>{denom_name}</b> &nbsp;|&nbsp; Composite Risk Score: <b>{risk_score}%</b> &nbsp;|&nbsp; System Confidence: <b>{confidence}%</b>
        </div>
        <div style="margin-top: 6px; font-size: 14px; color: #a7f3d0;">
            <b>Status:</b> All major security features and neural network authenticity checks verified.
        </div>
    </div>
    """, unsafe_allow_html=True)


# Key Summary KPI Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Final Verdict", str(active_result.get("verdict", "N/A")))
with col2:
    st.metric("Denomination", str(denom_name), f"Conf: {active_result.get('denomination', {}).get('confidence', 0)}%")
with col3:
    st.metric("Risk Score", f"{risk_score}%")
with col4:
    st.metric("System Confidence", f"{confidence}%")


# =============================================================================
# STAGE 2: IMAGE QUALITY GATE CHECK (Cell 26-27)
# =============================================================================

st.markdown('<div class="stage-header">Stage 2 · Image-Quality Gate & Clarity Check</div>', unsafe_allow_html=True)
q = active_result.get("quality", {})

q_col1, q_col2, q_col3, q_col4, q_col5 = st.columns(5)
with q_col1:
    q_pass = q.get("passed", False)
    st.metric("Quality Score", f"{q.get('quality_score', 0)}%", "PASS" if q_pass else "FAIL", delta_color="normal" if q_pass else "inverse")
with q_col2:
    st.metric("Blur Variance", str(q.get("blur_variance", "N/A")), "Target ≥ 80.0")
with q_col3:
    st.metric("Brightness", f"{q.get('brightness_mean', 0)} / 255")
with q_col4:
    st.metric("Glare Ratio", f"{float(q.get('glare_ratio', 0)) * 100:.2f}%")
with q_col5:
    res = q.get("resolution", [0, 0])
    st.metric("Resolution", f"{res[0]} x {res[1]} px")

warnings = q.get("warnings", [])
if warnings:
    st.warning("⚠️ **Quality Warnings:**\n" + "\n".join([f"- {w}" for w in warnings]))
else:
    st.success("✔ **No quality warnings.** The photo is clear and passes the image-quality gate.")


# =============================================================================
# STAGE 3: PREPROCESSING PIPELINE (Cell 27)
# =============================================================================

st.markdown('<div class="stage-header">Stage 3 · Computer Vision Preprocessing Strip</div>', unsafe_allow_html=True)
st.caption("Pipeline steps: Perspective warp & deskew, noise reduction, standard 256x128 resize, and CLAHE contrast enhancement.")

prep = active_result.get("preprocessing", {})

if prep and "original" in prep and prep["original"] is not None:
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    with p_col1:
        st.image(cv2.cvtColor(prep["original"], cv2.COLOR_BGR2RGB), caption="1. Original Scan", use_container_width=True)
    with p_col2:
        aligned_status = prep.get("aligned_success", True)
        st.image(cv2.cvtColor(prep["aligned"], cv2.COLOR_BGR2RGB), caption=f"2. Aligned Note (Detected: {aligned_status})", use_container_width=True)
    with p_col3:
        st.image(cv2.cvtColor(prep["denoised_bgr"], cv2.COLOR_BGR2RGB), caption="3. Denoised 256x128", use_container_width=True)
    with p_col4:
        st.image(prep["clahe_gray"], caption="4. CLAHE Contrast Map", use_container_width=True, clamp=True)

    # Matplotlib 4-panel strip figure identical to notebook Cell 27
    with st.expander("📊 View Preprocessing 4-Panel Subplot (Notebook Cell 27 format)", expanded=False):
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.2))
        panels = [
            ("Original", cv2.cvtColor(prep["original"], cv2.COLOR_BGR2RGB)),
            (f"Aligned (auto-detected: {aligned_status})", cv2.cvtColor(prep["aligned"], cv2.COLOR_BGR2RGB)),
            ("Denoised 256x128", cv2.cvtColor(prep["denoised_bgr"], cv2.COLOR_BGR2RGB)),
            ("CLAHE contrast", prep["clahe_gray"])
        ]
        for a, (t, im) in zip(ax, panels):
            a.imshow(im, cmap="gray" if im.ndim == 2 else None)
            a.set_title(t, fontsize=10)
            a.axis("off")
        plt.suptitle("Stage 3 - Preprocessing Transforms", fontsize=12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info("Preprocessing image strip is displayed in the final XAI report.")


# =============================================================================
# STAGE 4: DENOMINATION DETECTION (Cell 28-29)
# =============================================================================

st.markdown('<div class="stage-header">Stage 4 · Denomination Detection (6-Class CNN)</div>', unsafe_allow_html=True)
d = active_result.get("denomination", {})

d_col1, d_col2 = st.columns([1, 2])

with d_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Predicted Denomination</div>
        <div class="metric-value">{d.get("denomination", "Unknown")}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Prediction Confidence</div>
        <div class="metric-value">{d.get("confidence", 0)}%</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Colour Profile</div>
        <div style="font-size:1.1rem;font-weight:600;color:#334155;">{d.get("color_profile", "N/A")}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Model Used</div>
        <div style="font-size:0.95rem;color:#64748b;font-family:monospace;">{d.get("model_used", "denomination_cnn.keras")}</div>
    </div>
    """, unsafe_allow_html=True)

with d_col2:
    probs = d.get("all_probabilities") or {}
    if probs:
        names = list(probs.keys())
        vals = [float(v) for v in probs.values()]
        top_idx = int(np.argmax(vals))

        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        bars = ax.bar(names, vals, color="#94a3b8")
        bars[top_idx].set_color("#2563eb")

        ax.set_ylabel("Probability (%)", fontsize=10)
        ax.set_ylim(0, 108)
        ax.set_title("Denomination CNN - Class Probabilities", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 2.0, f"{v:.1f}%", ha="center", fontsize=9, fontweight="600")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# =============================================================================
# STAGE 5: AUTHENTICATION ANALYSIS & INTAGLIO (Cell 30-31)
# =============================================================================

st.markdown('<div class="stage-header">Stage 5 · Authentication Analysis & Intaglio Microstructure</div>', unsafe_allow_html=True)
a = active_result.get("auth", {})

a_col1, a_col2 = st.columns([1, 1.4])

with a_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Authentication CNN Verdict</div>
        <div class="metric-value" style="color:{'#15803d' if a.get('is_real') else '#b91c1c'};">{a.get("status", "N/A")}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">P(Genuine) vs P(Counterfeit)</div>
        <div style="font-size:1.15rem;font-weight:700;">
            Genuine: <span style="color:#15803d;">{a.get("real_probability", 0)}%</span> &nbsp;|&nbsp; 
            Fake: <span style="color:#b91c1c;">{a.get("fake_probability", 0)}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Intaglio Details
    it = a.get("intaglio", {})
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Intaglio Microstructure & Tactile Relief</div>
        <div style="font-size:0.95rem;margin-top:4px;">
            <b>Score:</b> {it.get("intaglio_score", "N/A")}% &nbsp;&middot;&nbsp; 
            <b>Laplacian Var:</b> {it.get("laplacian_var", "N/A")} &nbsp;&middot;&nbsp; 
            <b>Edge Energy:</b> {it.get("edge_energy", "N/A")}<br>
            <b>Flat Print-Like Surface:</b> <span style="color:{'#b91c1c' if it.get('is_flat_print') else '#15803d'};font-weight:700;">{'YES (Suspicious)' if it.get('is_flat_print') else 'NO (Raised Texture Found)'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with a_col2:
    # Horizontal Stacked Bar Chart for Real vs Fake Probability (Notebook Cell 31)
    real_p = float(a.get("real_probability", 50))
    fake_p = float(a.get("fake_probability", 50))

    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.barh(["Authentication"], [real_p], color="#10b981", label="P(Genuine)")
    ax.barh(["Authentication"], [fake_p], left=[real_p], color="#ef4444", label="P(Counterfeit)")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", fontsize=10)
    ax.set_title("Authentication CNN Binary Classifier Output", fontsize=11, fontweight="bold")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.4), frameon=True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =============================================================================
# STAGE 5b: NOVELTY & PLAY-MONEY OCR (Cell 32-33)
# =============================================================================

st.markdown('<div class="stage-header">Stage 5b · Novelty / Play-Money OCR Text Verification</div>', unsafe_allow_html=True)
nov = active_result.get("novelty", {})

if not nov.get("available", True):
    st.info("ℹ️ OCR module was skipped or not available on this scan.")
elif nov.get("is_novelty", False):
    matched = ", ".join([p.upper() for p in nov.get("matched_phrases", [])])
    st.error(f"🚨 **NOVELTY / PLAY-MONEY NOTE DETECTED!** Matched phrases: `{matched}`. This overrides neural network confidence!")
else:
    st.success("✔ **No novelty or play-money text detected** on this banknote.")

rbi_status = "FOUND" if nov.get("rbi_text_found", False) else "NOT FOUND"
st.write(f"**'RESERVE BANK OF INDIA' Heading Text:** `{rbi_status}`")

with st.expander("📄 View Extracted OCR Text from Banknote"):
    ocr_text = nov.get("ocr_text", "(No text detected)")
    st.text(ocr_text[:600] if ocr_text else "(No text detected)")


# =============================================================================
# STAGE 6: SECURITY-FEATURE ANALYSIS (Cell 34-35)
# =============================================================================

st.markdown('<div class="stage-header">Stage 6 · Security-Feature Verification Audit</div>', unsafe_allow_html=True)
sec = active_result.get("security", {})

st.write(f"**Verification Ratio:** `{sec.get('passed_features', 0)} / {sec.get('total_features', 5)}` features verified (`{sec.get('pass_ratio', 0)}%` pass ratio)")

# Horizontal Bar Chart for Security Features (Notebook Cell 35)
features_dict = sec.get("feature_details", {})
if features_dict:
    names = [f["name"] for f in features_dict.values()]
    scores = [float(f["score"]) for f in features_dict.values()]
    colors = ["#10b981" if f["passed"] else "#ef4444" for f in features_dict.values()]

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    bars = ax.barh(names, scores, color=colors)
    ax.axvline(45, ls="--", color="#334155", lw=1.2)
    ax.text(46.5, -0.4, "pass threshold 45%", fontsize=8.5, color="#334155", fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Feature Score (%)", fontsize=10)
    ax.set_title("Security-Feature Verification (Pass ≥ 45%)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    for bar, score in zip(bars, scores):
        ax.text(score + 1.5, bar.get_y() + bar.get_height() / 2, f"{score:.1f}%", va="center", fontsize=8.5)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Feature Audit Table with Crops
st.markdown("#### Feature Region Inspection Table")
for key, feat in features_dict.items():
    f_cols = st.columns([1.5, 1, 1, 2.5, 1.5])
    with f_cols[0]:
        st.markdown(f"**{feat.get('name', key)}**<br><span style='font-size:11px;color:#64748b;'>{feat.get('description', '')}</span>", unsafe_allow_html=True)
    with f_cols[1]:
        ok = feat.get("passed", False)
        badge = f"<span class='badge-pass'>✔ Verified</span>" if ok else f"<span class='badge-fail'>✖ {feat.get('status', 'Failed')}</span>"
        st.markdown(badge, unsafe_allow_html=True)
    with f_cols[2]:
        st.write(f"**{feat.get('score', 0)}%**")
    with f_cols[3]:
        st.caption(feat.get("details", ""))
    with f_cols[4]:
        crop_img = decode_base64_image(feat.get("crop_base64", ""))
        if crop_img is not None:
            st.image(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB), caption=feat.get("name", "Crop"), use_container_width=True)


# =============================================================================
# STAGE 6b: WHY — HOW THIS NOTE DIFFERS FROM GENUINE BASELINE (Cell 36-37)
# =============================================================================

st.markdown('<div class="stage-header">Stage 6b · XAI: Comparison with Genuine Baseline</div>', unsafe_allow_html=True)
diffs = active_result.get("xai_report", {}).get("differences", [])

if not diffs:
    st.success("✔ **No measurable deviation from a genuine note** — every security feature fell inside its expected genuine range.")
else:
    st.warning(f"⚠️ **{len(diffs)} security feature(s) deviate from the genuine baseline:**")
    for d_item in diffs:
        with st.container():
            st.markdown(f"""
            <div style="background:#fef2f2;border-left:4px solid #ef4444;padding:10px 14px;border-radius:6px;margin-bottom:10px;">
                <div style="font-weight:700;color:#991b1b;font-size:1.05rem;">{d_item.get('feature', 'Feature')} &mdash; <span style="font-weight:400;font-size:0.9rem;color:#64748b;">{d_item.get('metric', '')}</span></div>
                <div style="margin-top:4px;font-size:0.95rem;">
                    <b>A genuine note reads:</b> <span style="color:#15803d;">{d_item.get('genuine', '')}</span><br>
                    <b>This note reads:</b> <span style="color:#b91c1c;">{d_item.get('this_note', '')}</span><br>
                    <b>Physical difference:</b> {d_item.get('difference', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# STAGE 7: SUSPICIOUS-REGION LOCALISATION (Cell 38-39)
# =============================================================================

st.markdown('<div class="stage-header">Stage 7 · Suspicious-Region Localisation</div>', unsafe_allow_html=True)
an = active_result.get("anomaly", {})

if an.get("has_anomalies", False):
    st.error(f"🚨 **{an.get('anomaly_count', 0)} SUSPICIOUS REGION(S) LOCALIZED ON BANKNOTE**")
    for s in an.get("suspicious_regions", []):
        x, y, w, h = s.get("bbox", [0, 0, 0, 0])
        st.markdown(f"- **[{s.get('severity', 'WARNING')}] {s.get('feature', 'Region')}**: {s.get('reason', '')} *(Bounding box: x={x}, y={y}, w={w}, h={h})*")
else:
    st.success("✔ **No suspicious regions localized.** All security feature areas conform to standard coordinates.")

an_col1, an_col2 = st.columns(2)

annotated_img = None
if "annotated" in an and an["annotated"] is not None:
    annotated_img = an["annotated"]
elif "annotated_base64" in an and an["annotated_base64"]:
    annotated_img = decode_base64_image(an["annotated_base64"])

diff_map_img = decode_base64_image(an.get("diff_map_base64", ""))

with an_col1:
    if annotated_img is not None:
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Localised Suspicious Regions (Bounding Boxes)", use_container_width=True)
with an_col2:
    if diff_map_img is not None:
        st.image(cv2.cvtColor(diff_map_img, cv2.COLOR_BGR2RGB), caption="Texture-Residual Anomaly Map", use_container_width=True)


# =============================================================================
# STAGE 8: GRAD-CAM EXPLAINABILITY (Cell 40-41)
# =============================================================================

st.markdown('<div class="stage-header">Stage 8 · Grad-CAM Explainability Attention Map</div>', unsafe_allow_html=True)
gc = active_result.get("gradcam", {})
layer_name = gc.get("layer_used", active_result.get("gradcam_layer", "Convolutional Layer"))
st.caption(f"Explainability source: **{layer_name}**. Warm colors (yellow/red) indicate regions that heavily influenced the neural network's classification.")

gc_col1, gc_col2, gc_col3 = st.columns(3)

# 1. Aligned Note
with gc_col1:
    orig_aligned = None
    if prep and "resized_bgr" in prep and prep["resized_bgr"] is not None:
        orig_aligned = prep["resized_bgr"]
    elif prep and "aligned" in prep and prep["aligned"] is not None:
        orig_aligned = prep["aligned"]
    elif "analyzed_image" in st.session_state:
        orig_aligned = cv2.resize(st.session_state["analyzed_image"], (256, 128))

    if orig_aligned is not None:
        st.image(cv2.cvtColor(orig_aligned, cv2.COLOR_BGR2RGB), caption="1. Original Banknote (Aligned)", use_container_width=True)

# 2. Raw Heatmap
with gc_col2:
    if "heatmap_raw" in gc and gc["heatmap_raw"] is not None:
        fig_hm, ax_hm = plt.subplots(figsize=(5, 2.5))
        ax_hm.imshow(gc["heatmap_raw"], cmap="jet")
        ax_hm.axis("off")
        ax_hm.set_title("2. Activation Heatmap", fontsize=10)
        fig_hm.tight_layout()
        st.pyplot(fig_hm)
        plt.close(fig_hm)
    else:
        st.caption("Activation heatmap available in report.")

# 3. Superimposed Grad-CAM
with gc_col3:
    super_img = None
    if "superimposed" in gc and gc["superimposed"] is not None:
        super_img = gc["superimposed"]
    elif "superimposed_base64" in gc and gc["superimposed_base64"]:
        super_img = decode_base64_image(gc["superimposed_base64"])

    if super_img is not None:
        st.image(cv2.cvtColor(super_img, cv2.COLOR_BGR2RGB), caption="3. Grad-CAM Overlay", use_container_width=True)

# Grad-CAM Narrative Explanation
xai_narrative = active_result.get("xai_report", {}).get("xai_explanation", "")
if xai_narrative:
    st.info(f"🧠 **Model Attention Explanation:**\n\n{xai_narrative}")


# =============================================================================
# STAGE 9: OVERALL REVIEW & RISK COMPOSITION (Cell 42-43)
# =============================================================================

st.markdown('<div class="stage-header">Stage 9 · Overall Review & Risk Composition</div>', unsafe_allow_html=True)
r = active_result.get("risk", {})
b = r.get("breakdown", {})

rev_col1, rev_col2 = st.columns([1, 1.2])

with rev_col1:
    st.markdown(f"""
    <div style="border: 2px solid {status_color}; border-radius: 8px; padding: 14px 18px; background: #ffffff;">
        <div style="font-size: 22px; font-weight: 800; color: {status_color};">{verdict}</div>
        <div style="color: #64748b; font-size: 13px; margin-bottom: 12px;">{denom_name} &nbsp;&middot;&nbsp; Analysed: {active_result.get('xai_report', {}).get('timestamp', 'Recent')}</div>
        <table style="width: 100%; font-size: 13.5px; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 4px 0;">Denomination Confidence</td><td style="text-align:right;"><b>{d.get('confidence', 0)}%</b></td></tr>
            <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 4px 0;">CNN Real / Fake Probability</td><td style="text-align:right;"><b>{a.get('real_probability', 0)}% / {a.get('fake_probability', 0)}%</b></td></tr>
            <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 4px 0;">Security Features Passed</td><td style="text-align:right;"><b>{sec.get('passed_features', 0)}/{sec.get('total_features', 5)} ({sec.get('pass_ratio', 0)}%)</b></td></tr>
            <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 4px 0;">Suspicious Regions Found</td><td style="text-align:right;"><b>{an.get('anomaly_count', 0)}</b></td></tr>
            <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 4px 0;">Image Quality Score</td><td style="text-align:right;"><b>{q.get('quality_score', 0)}%</b></td></tr>
            <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 4px 0; color:{status_color};"><b>Composite Risk Score</b></td><td style="text-align:right; color:{status_color};"><b>{risk_score}%</b></td></tr>
            <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 4px 0;">System Confidence</td><td style="text-align:right;"><b>{confidence}%</b></td></tr>
            <tr><td style="padding: 4px 0;">Print-Forgery Pattern</td><td style="text-align:right;"><b>{'YES (Detected)' if r.get('is_print_forgery') else 'No'}</b></td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with rev_col2:
    # Risk Composition Bar Chart (Notebook Cell 43)
    labels = ["CNN Fake Risk\n(35% weight)", "Security Features\n(45% weight)", "Anomaly Penalty", "Quality Penalty"]
    vals = [
        float(b.get("fake_classifier_risk", 0)),
        float(b.get("security_feature_risk", 0)),
        float(b.get("anomaly_penalty", 0)),
        float(b.get("quality_penalty", 0))
    ]

    fig_rc, ax_rc = plt.subplots(figsize=(7.5, 3.4))
    rc_bars = ax_rc.bar(labels, vals, color=["#6366f1", "#ef4444", "#f97316", "#f59e0b"])
    ax_rc.set_ylabel("Risk Points", fontsize=10)
    ax_rc.set_title(f"Risk Composition Breakdown — Total Risk: {risk_score}%", fontsize=11, fontweight="bold")
    ax_rc.grid(axis="y", linestyle=":", alpha=0.5)

    for bar, val in zip(rc_bars, vals):
        ax_rc.text(bar.get_x() + bar.get_width() / 2, val + 0.6, f"{val:.1f}", ha="center", fontsize=9, fontweight="600")

    fig_rc.tight_layout()
    st.pyplot(fig_rc)
    plt.close(fig_rc)


# =============================================================================
# STAGE 11: DOWNLOADABLE XAI REPORT AT THE END (Cell 46, 47, 49)
# =============================================================================

st.markdown('<div class="stage-header">Stage 11 · Downloadable Explainable-AI (XAI) Report</div>', unsafe_allow_html=True)
st.write("Generate and download the complete, self-contained Explainable-AI forensic report containing all visual overlays, audit tables, and narrative rationales.")

# Build HTML report
try:
    html_report = pipeline.generate_html_report(active_result)
except Exception:
    # Fallback to reading saved report if available
    if os.path.exists("outputs/currency_xai_report.html"):
        with open("outputs/currency_xai_report.html", "r", encoding="utf-8") as f:
            html_report = f.read()
    else:
        html_report = "<html><body><h1>Report not generated</h1></body></html>"

# Clean JSON representation
clean_json_str = json.dumps(clean_for_json(active_result), indent=2, ensure_ascii=False)

dl_col1, dl_col2 = st.columns(2)

with dl_col1:
    st.download_button(
        label="📥 Download XAI Report (HTML)",
        data=html_report,
        file_name=f"Currency_XAI_Report_{denom_name.replace(' ', '_')}.html",
        mime="text/html",
        type="primary",
        use_container_width=True
    )

with dl_col2:
    st.download_button(
        label="📥 Download Analysis Data (JSON)",
        data=clean_json_str,
        file_name=f"Currency_Result_{denom_name.replace(' ', '_')}.json",
        mime="application/json",
        use_container_width=True
    )

# Report Interactive Preview
with st.expander("👁️ Preview Generated XAI Audit Report Inline", expanded=False):
    st.components.v1.html(html_report, height=750, scrolling=True)

st.markdown("---")
st.caption("Indian Fake Currency Detection with Explainable AI · Academic Project · Designed for faculty review & evaluation.")
