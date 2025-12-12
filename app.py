import streamlit as st
import pandas as pd
import joblib
import time
import base64
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ---------- SETTINGS ----------
HIGH_RISK_THRESHOLD = 0.70
MODERATE_RISK_THRESHOLD = 0.40
ALERT_SOUND_FILE = "alert_beep.mp3"       # put this in same folder
LOGO_FILE = "elantrix_logo.png"          # put this in same folder

# ---------- HELPERS ----------
@st.cache_resource
def load_model():
    # Train the model directly from the CSV so it matches the sklearn version in the cloud
    df = pd.read_csv("incart_arrhythmia.csv")

    # Same preprocessing as train_model.py
    df["label"] = (df["type"] != "N").astype(int)
    df = df.dropna()

    drop_cols = ["record", "type"]
    feature_cols = [c for c in df.columns if c not in drop_cols + ["label"]]

    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return model, feature_cols


def play_alert_sound():
    sound_path = Path(ALERT_SOUND_FILE)
    if sound_path.exists():
        data = sound_path.read_bytes()
        b64 = base64.b64encode(data).decode()
        md = f"""
        <audio autoplay style="display:none">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
        st.markdown(md, unsafe_allow_html=True)

def risk_level(avg_risk: float) -> str:
    if avg_risk > HIGH_RISK_THRESHOLD:
        return "high"
    elif avg_risk > MODERATE_RISK_THRESHOLD:
        return "moderate"
    return "low"


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Elantrix – Arrhythmia Risk Demo",
    layout="wide",
)
st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

# ---------- CUSTOM CSS ----------
# ---------- GLOBAL DARK THEME (MOBILE-FRIENDLY) ----------
st.markdown(
    """
    <style>
    /* Dark background for whole app */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #020617 !important;
        color: #e5e7eb !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Main text */
    h1, h2, h3, h4, h5, h6,
    p, span, label, li, div {
        color: #e5e7eb !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #020617 !important;
        color: #e5e7eb !important;
    }

    /* Inputs (Dad, phone, etc.) */
    input, textarea {
        background-color: #020617 !important;
        color: #e5e7eb !important;
        border-radius: 8px !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #9ca3af !important;  /* visible light grey */
        opacity: 1 !important;
    }

    /* HEADER FIX – make top bar dark and icons light */
    [data-testid="stHeader"] {
        background-color: #020617 !important;
        color: #e5e7eb !important;
        box-shadow: none !important;
    }
    [data-testid="stHeader"] * {
        color: #e5e7eb !important;
        fill: #e5e7eb !important;
    }

    /* FILE UPLOADER FIX – text + button visible */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #020617 !important;
        border: 1px dashed #4b5563 !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #020617 !important;
    }

    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span {
        color: #e5e7eb !important;   /* “Drag and drop file here”, size text, etc. */
    }

    [data-testid="stFileUploader"] button {
        background-color: #f97316 !important;  /* orange Browse button */
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- MODEL ----------
model, features = load_model()

# ---------- LAYOUT TOP BAR ----------
col_logo, col_title = st.columns([1, 4])

with col_logo:
    logo_path = Path(LOGO_FILE)
    if logo_path.exists():
        st.image(str(logo_path), width=90)

with col_title:
    st.markdown("### Elantrix")
    st.markdown(
        "<h1 style='margin-top:0;'>Arrhythmia & Early Heart Attack Risk Demo</h1>",
        unsafe_allow_html=True,
    )
    st.write(
        "Simulated smartwatch engine that analyses ECG-beat features in real time, "
        "detects dangerous arrhythmias, and triggers alerts to family and hospitals."
    )

st.markdown("---")

# ---------- SIDEBAR: CONTACT DETAILS ----------
st.sidebar.header("Alert Recipients")

family_name = st.sidebar.text_input("Family Member Name", "Dad")
family_phone = st.sidebar.text_input("Family Phone Number", "+91-9XXXXXXXXX")

hospital_name = st.sidebar.text_input("Nearest Hospital Name", "City Heart Institute")
hospital_phone = st.sidebar.text_input("Hospital Emergency Number", "108")

st.sidebar.info(
    "These contacts will be shown as recipients when a **high-risk alert** is triggered."
)

# ---------- FILE UPLOAD ----------
st.subheader("1. Upload ECG Feature Data")
uploaded = st.file_uploader(
    "Upload arrhythmia ECG segment (CSV) – this simulates data streaming from a smartwatch.",
    type=["csv"],
)

if not uploaded:
    st.info("Upload a CSV file (e.g. `normal_segment.csv` or `arrhythmia_segment.csv`) to start.")
    st.stop()

data = pd.read_csv(uploaded)

# keep only required features (ignore extra columns if present)
X = data[features]
probs = model.predict_proba(X)[:, 1]
avg_risk = float(probs.mean())
current_risk = risk_level(avg_risk)

# ---------- TABS: BATCH ANALYSIS / SIMULATED STREAM ----------
tab1, tab2 = st.tabs(["📊 Batch Analysis", "⌚ Simulated Smartwatch Stream"])

# ----- TAB 1: Batch Analysis -----
with tab1:
    st.subheader("2. Batch Analysis Result")

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Beats Analysed", len(X))
    with colB:
        st.metric("Average Arrhythmia Probability", f"{avg_risk:.2f}")
    with colC:
        if current_risk == "high":
            st.metric("Risk Level", "HIGH", "+ ALERT")
        elif current_risk == "moderate":
            st.metric("Risk Level", "MODERATE")
        else:
            st.metric("Risk Level", "LOW")

    # Alert box
    if current_risk == "high":
        st.error("🚨 HIGH RISK – Alert Triggered! (Demo)")
        play_alert_sound()
        st.markdown(
            f"""
            **Notifications sent to:**
            - 👨‍👩‍👧 Family: **{family_name}** ({family_phone})
            - 🏥 Hospital: **{hospital_name}** ({hospital_phone})
            """)
    elif current_risk == "moderate":
        st.warning("⚠️ MODERATE RISK – Irregularities detected. Recommend medical review.")
    else:
        st.success("✅ Normal Rhythm – No critical arrhythmia detected.")
    st.markdown("#### Arrhythmia Probability Over Time")
    st.line_chart(probs)

# ----- TAB 2: Simulated Smartwatch Stream -----
with tab2:
    st.subheader("2. Real-Time Streaming Simulation")

    st.write(
        "This simulates how our engine would behave on a smartwatch or phone – "
        "processing each heartbeat one-by-one and triggering alerts instantly."
    )

    # placeholders for live update
    status_placeholder = st.empty()
    chart_placeholder = st.empty()
    alert_placeholder = st.empty()

    # duration per beat (seconds)
    speed = st.slider("Playback speed (seconds per beat)", 0.02, 0.5, 0.10, 0.01)

    if st.button("▶ Start Simulation"):
        play_alert = False
        high_alert_triggered = False
        probs_so_far = []

        for idx, p in enumerate(probs):
            probs_so_far.append(p)

            with status_placeholder.container():
                st.write(f"Beat **{idx+1} / {len(probs)}**")
                st.progress(min(p, 0.999))

            with chart_placeholder.container():
                st.line_chart(probs_so_far)

            # Alert logic
            alert_placeholder.empty()
            if p > HIGH_RISK_THRESHOLD:
                high_alert_triggered = True
                with alert_placeholder.container():
                    st.error(
                        f"🚨 HIGH-RISK BEAT DETECTED at beat {idx+1}! "
                        "Immediate notification triggered. (Demo)"
                    )
                    st.markdown(
                        f"""
                        **Notifying:**
                        - 👨‍👩‍👧 Family: **{family_name}** ({family_phone})  
                        - 🏥 Hospital: **{hospital_name}** ({hospital_phone})
                        """
                    )
                play_alert = True

            time.sleep(speed)

        if high_alert_triggered and play_alert:
            play_alert_sound()
        elif not high_alert_triggered:
            with alert_placeholder.container():
                st.success("Simulation finished. No high-risk beats detected.")
# ---------- TEAM FOOTER ----------
st.set_page_config(page_title="Team Footer", layout="wide")

def img_to_base64(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    ext = path.suffix.lower().replace('.', '')
    mime = "jpeg" if ext in ("jpg","jpeg") else "png" if ext=="png" else ext
    return f"data:image/{mime};base64,{b64}"

def render_team_footer():
    st.markdown(
        """
        <style>
        /* responsive avatar size variables */
        :root {
          --avatar-desktop: 180px;
          --avatar-tablet: 150px;
          --avatar-mobile: 120px;
          --avatar-border: 6px;
        }

        /* overall wrapper for each person */
        .person {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          gap: 12px;
          padding: 18px 8px;
        }

        /* circular mask wrapper — forces perfect circle and hides corners */
        .avatar-wrapper {
          width: var(--avatar-desktop);
          height: var(--avatar-desktop);
          border-radius: 50%;
          overflow: hidden;              /* THIS hides rectangular edges */
          display: inline-block;
          box-shadow: 0 10px 30px rgba(0,0,0,0.45);
          transition: transform 0.28s ease, box-shadow 0.28s ease;
          border: var(--avatar-border) solid rgba(255,255,255,0.06);
          background: radial-gradient(circle at 30% 20%, rgba(255,255,255,0.04), rgba(0,0,0,0.08));
        }

        /* ensures image covers wrapper and no visible corners */
        .avatar-wrapper img {
          width: 100%;
          height: 100%;
          object-fit: cover;    /* fills and crops nicely */
          display: block;
          transform-origin: center;
        }

        /* hover/focus effect */
        .avatar-wrapper:hover {
          transform: translateY(-6px) scale(1.04);
          box-shadow: 0 20px 40px rgba(0,0,0,0.6);
          border-color: rgba(255,255,255,0.12);
        }

        /* name & role styling (centered) */
        .person-name {
          font-weight: 700;
          font-size: 18px;
          margin: 0;
        }
        .person-role {
          margin: 2px 0 0 0;
          font-size: 15px;
          color: rgba(255,255,255,0.85);
        }
        .person-desc {
          margin-top: 6px;
          font-size: 14px;
          color: rgba(255,255,255,0.75);
          line-height: 1.25;
        }

        /* responsive adjustments */
        @media (max-width: 992px) {
          .avatar-wrapper { width: var(--avatar-tablet); height: var(--avatar-tablet); }
        }
        @media (max-width: 600px) {
          .avatar-wrapper { width: var(--avatar-mobile); height: var(--avatar-mobile); }
          .person { padding: 12px 6px; }
          .person-name { font-size: 16px; }
          .person-role { font-size: 14px; }
        }

        /* make Streamlit columns background transparent nicer spacing */
        .stApp .block-container {
          padding-top: 32px;
          padding-left: 48px;
          padding-right: 48px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # set image folder or '.' if same dir
    base = Path(".")
    a = base / "anurag_founder.jpg"
    b = base / "cofounder_cto.jpg"
    s1 = base / "supervisor1.jpg"
    s2 = base / "supervisor2.jpg"
    s3 = base / "supervisor3.jpg"

    # show warnings for missing files
    for p in (a, b, s1, s2, s3):
        if not p.exists():
            st.warning(f"Image not found: `{p.name}` — put this file in the same folder as app.py or change base path.")

    st.markdown("---")
    st.markdown("### Our Team")

    # two columns for founders
    col1, col2 = st.columns(2)
    with col1:
        # whole HTML for a centered person block
        src = img_to_base64(a) if a.exists() else ""
        st.markdown(f"""
        <div class="person">
          <div class="avatar-wrapper">
            <img src="{src}" alt="Anurag"/>
          </div>
          <p class="person-name">Anurag</p>
          <p class="person-role">Founder & CEO</p>
          <p class="person-desc">MBBS, 3rd year<br>Navodaya Medical College</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        src = img_to_base64(b) if b.exists() else ""
        st.markdown(f"""
        <div class="person">
          <div class="avatar-wrapper">
            <img src="{src}" alt="Shrishant"/>
          </div>
          <p class="person-name">Shrishant</p>
          <p class="person-role">CTO</p>
          <p class="person-desc">B.Tech 3rd year<br>NIT Andhra Pradesh</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Under the Supervision")
    # three columns for supervisors
    col3, col4, col5 = st.columns(3)

    entries = [
        (col3, s1, "Dr. Shankargouda", "MBBS MD DM\nCardiologist"),
        (col4, s2, "Dr. Shivkumar", "MBBS MD DNB, MNAMS\nForensic Medicine & Toxicology"),
        (col5, s3, "Dr. Bandenawaz", "MBBS MD\nForensic Medicine & Toxicology"),
    ]

    for col, path, title, desc in entries:
        src = img_to_base64(path) if path.exists() else ""
        col.markdown(f"""
        <div class="person">
          <div class="avatar-wrapper">
            <img src="{src}" alt="{title}"/>
          </div>
          <p class="person-name">{title}</p>
          <p class="person-role"></p>
          <p class="person-desc">{desc.replace(chr(10), '<br>')}</p>
        </div>
        """, unsafe_allow_html=True)

render_team_footer()