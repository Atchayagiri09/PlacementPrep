"""
PlaceReady - AI Placement Readiness Predictor
Run:  pip install streamlit scikit-learn pandas numpy
      streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------
# 1. Role -> required skills: (skill, importance 1-5, what to learn)
# ---------------------------------------------------------------
ROLES = {
    "Data Science / ML": [
        ("Python", 5, "Finish a Python for Data Analysis course; practice daily"),
        ("SQL", 5, "Practice joins, group by, window functions on LeetCode/HackerRank"),
        ("Statistics", 5, "Learn probability, hypothesis testing, distributions"),
        ("Machine Learning", 5, "Andrew Ng's ML course + build 2 end-to-end models"),
        ("Pandas & NumPy", 4, "Clean and analyze 2 Kaggle datasets"),
        ("Data Visualization", 3, "Matplotlib/Seaborn; build one dashboard"),
        ("Power BI / Tableau", 3, "Make a business dashboard from a public dataset"),
        ("Deep Learning", 3, "Basics of neural nets with TensorFlow/PyTorch"),
        ("Git & GitHub", 2, "Push all projects with a clear README"),
    ],
    "Web Development": [
        ("HTML & CSS", 4, "Clone 2 real websites responsively"),
        ("JavaScript", 5, "Learn ES6, DOM, async/await; build small apps"),
        ("React", 5, "Build a full React app with routing and state"),
        ("Node.js", 4, "Create a REST API with Express"),
        ("REST APIs", 4, "Learn HTTP methods, auth (JWT), Postman"),
        ("SQL", 3, "Basic queries and schema design"),
        ("MongoDB", 3, "CRUD with Mongoose in a MERN project"),
        ("Git & GitHub", 3, "Branching, pull requests, deploy via Vercel/Netlify"),
    ],
    "Software Engineer": [
        ("DSA", 5, "Solve 150+ problems: arrays, trees, graphs, DP"),
        ("OOP", 4, "Master 4 pillars in Java/C++/Python with small designs"),
        ("Core Language (Java/C++/Python)", 5, "Pick one language and go deep"),
        ("DBMS", 4, "Normalization, transactions, SQL queries"),
        ("Operating Systems", 3, "Processes, threads, scheduling, memory"),
        ("Computer Networks", 3, "OSI/TCP-IP, HTTP, DNS basics"),
        ("System Design Basics", 3, "Learn scalability, caching, load balancing"),
        ("Git & GitHub", 3, "Version control and collaboration workflow"),
    ],
}


# ---------------------------------------------------------------
# 2. Synthetic dataset + model
#    (Be upfront in the demo: training data is synthetic.)
# ---------------------------------------------------------------
FEATURES = ["cgpa", "projects", "dsa_solved", "internships", "certifications", "skill_match"]


def generate_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "cgpa": np.clip(rng.normal(7.8, 0.9, n), 5.5, 10),
        "projects": rng.integers(0, 7, n),
        "dsa_solved": rng.integers(0, 400, n),
        "internships": rng.choice([0, 1, 2, 3], n, p=[0.45, 0.35, 0.15, 0.05]),
        "certifications": rng.integers(0, 6, n),
        "skill_match": rng.beta(2, 2, n),  # 0-1 fraction of required skills known
    })
    score = (
        0.20 * (df.cgpa - 5.5) / 4.5
        + 0.15 * df.projects / 6
        + 0.20 * df.dsa_solved / 400
        + 0.15 * df.internships / 3
        + 0.05 * df.certifications / 5
        + 0.25 * df.skill_match
    )
    score += rng.normal(0, 0.05, n)  # noise
    df["placed"] = (score > score.quantile(0.5)).astype(int)
    return df


@st.cache_resource
def train_model():
    df = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["placed"], test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model, model.score(X_test, y_test)


# ---------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------
def skill_match_and_gaps(role, known):
    skills = ROLES[role]
    total = sum(w for _, w, _ in skills)
    have = sum(w for s, w, _ in skills if s in known)
    gaps = sorted(
        [(s, w, tip) for s, w, tip in skills if s not in known],
        key=lambda x: -x[1],
    )
    return have / total, gaps


def profile_tips(cgpa, projects, dsa, internships, certs):
    tips = []
    if cgpa < 8.0:
        tips.append("Raise CGPA toward 8+ (many companies use it as a cutoff).")
    if projects < 3:
        tips.append("Build at least 3 solid projects with GitHub links.")
    if dsa < 150:
        tips.append("Solve more DSA problems (target 150+).")
    if internships < 1:
        tips.append("Get an internship or a virtual internship.")
    if certs < 2:
        tips.append("Add 2 relevant certifications (NPTEL, Coursera, etc.).")
    return tips


def label_for(score):
    if score >= 75:
        return "Placement Ready"
    if score >= 50:
        return "Almost There"
    return "Needs Work"


# ---------------------------------------------------------------
# 4. UI — PlaceReady Studio redesign
# ---------------------------------------------------------------
st.set_page_config(page_title="PlaceReady — Career Studio", page_icon="🎯", layout="wide")

# Design system: forest-black, ivory, electric lime.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #101410;
  --panel: #191f19;
  --panel-2: #202820;
  --lime: #d7ff72;
  --ivory: #f5f3eb;
  --muted: #a6afa0;
  --line: rgba(245,243,235,.11);
}
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
}
.stApp {
  background:
    radial-gradient(ellipse at 85% 0%, rgba(215,255,114,.075), transparent 30%),
    var(--bg);
  color: var(--ivory);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 1rem; }
.block-container { max-width: 1280px; padding-top: 2.2rem; padding-bottom: 4rem; }
h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; color: var(--ivory); }
h1 { font-weight: 800 !important; }
p, label, .stCaption, [data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stSidebar"] {
  background: #171c17;
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
[data-testid="stSidebar"] h2 { font-size: 1.25rem; }
[data-testid="stSidebar"] label { font-size: .9rem; }
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stMultiSelect"] > div > div {
  background: #101410 !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
  background: var(--lime) !important;
  border-color: var(--lime) !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] { background: var(--line); }
.stButton > button[kind="primary"], .stFormSubmitButton > button {
  background: var(--lime) !important;
  color: #151b12 !important;
  border: 0 !important;
  border-radius: 12px !important;
  font-weight: 800 !important;
  padding: .7rem 1rem !important;
  min-height: 3rem;
  transition: transform .15s ease, filter .15s ease;
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {
  filter: brightness(1.04); transform: translateY(-1px);
}
.stButton > button:not([kind="primary"]) {
  border: 1px solid var(--line);
  border-radius: 12px;
  color: var(--ivory);
  background: var(--panel);
}
[data-testid="stMetric"] {
  background: linear-gradient(135deg, #202820, #191f19);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1rem 1.15rem;
}
[data-testid="stMetricValue"] { color: var(--lime); font-family: 'Manrope',sans-serif; }
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stProgress"] > div > div { background: var(--lime) !important; }
[data-testid="stProgress"] > div { background: #30392f !important; border-radius: 999px; }
hr { border-color: var(--line) !important; }
[data-testid="stAlert"] { border-radius: 14px; }
.pr-brand { display:flex; align-items:center; gap:12px; margin-bottom:1.6rem; }
.pr-mark {
  display:flex; align-items:center; justify-content:center;
  width:48px; height:48px; border-radius:15px;
  background:var(--lime); color:#151b12; font-size:25px; font-weight:800;
}
.pr-wordmark { font-family:'Manrope',sans-serif; font-size:1.35rem; font-weight:800; letter-spacing:-.05em; color:var(--ivory); }
.pr-eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; color:var(--lime); font-weight:800; }
.pr-hero { padding:1.4rem 0 1.1rem; }
.pr-hero h1 { font-size:clamp(2.1rem, 5vw, 3.65rem); line-height:1.04; margin:.6rem 0 .7rem; }
.pr-hero p { max-width:680px; font-size:1.04rem; line-height:1.7; }
.pr-panel {
  background:linear-gradient(145deg, rgba(32,40,32,.96), rgba(25,31,25,.96));
  border:1px solid var(--line); border-radius:20px; padding:1.2rem 1.25rem;
  height:100%;
}
.pr-panel-title { font-family:'Manrope',sans-serif; font-size:1.05rem; font-weight:800; color:var(--ivory); margin-bottom:.35rem; }
.pr-muted { color:var(--muted); font-size:.9rem; line-height:1.6; }
.pr-chip { display:inline-block; padding:.32rem .62rem; border-radius:999px; background:rgba(215,255,114,.12); color:var(--lime); font-size:.76rem; font-weight:700; }
.pr-bigscore { font-family:'Manrope',sans-serif; font-size:2.8rem; font-weight:800; color:var(--lime); letter-spacing:-.06em; line-height:1.1; }
.pr-divider { height:1px; background:var(--line); margin:1rem 0; }
.pr-step { border-left:2px solid var(--lime); padding:.15rem 0 .8rem .9rem; margin:.25rem 0; }
.pr-footer { color:#788174; font-size:.78rem; padding-top:2rem; }
@media (max-width: 700px) {
  .block-container { padding: 1.15rem 1rem 3rem; }
  .pr-hero { padding-top:.5rem; }
  .pr-hero h1 { font-size:2.35rem; }
  .pr-panel { padding:1rem; border-radius:16px; }
  [data-testid="stMetric"] { padding:.85rem; }
}
</style>
""", unsafe_allow_html=True)

model, acc = train_model()

with st.sidebar:
    st.markdown("""
    <div class="pr-brand">
      <div class="pr-mark">P</div>
      <div><div class="pr-wordmark">PlaceReady</div><div class="pr-muted">CAREER STUDIO</div></div>
    </div>
    <div class="pr-eyebrow">Your profile</div>
    <p class="pr-muted">Tell us where you are today. We'll help you choose what to work on next.</p>
    """, unsafe_allow_html=True)
    role = st.selectbox("Target role", list(ROLES.keys()))
    cgpa = st.slider("CGPA", 5.0, 10.0, 8.0, 0.1)
    projects = st.number_input("Projects completed", 0, 15, 2)
    dsa = st.number_input("DSA problems solved", 0, 1000, 50)
    internships = st.number_input("Internships", 0, 5, 0)
    certs = st.number_input("Certifications", 0, 15, 1)
    known = st.multiselect("Skills you already know", [s for s, _, _ in ROLES[role]])
    st.markdown("<div class='pr-divider'></div>", unsafe_allow_html=True)
    go = st.button("✦  Build my readiness report", type="primary", use_container_width=True)
    st.caption("Your inputs are used to generate this on-screen report.")

# Shared brand header and hero.
st.markdown("""
<div class="pr-brand">
  <div class="pr-mark">P</div>
  <div><div class="pr-wordmark">PlaceReady</div><div class="pr-muted">YOUR CAREER, IN PROGRESS</div></div>
</div>
<div class="pr-hero">
  <div class="pr-eyebrow">The career readiness workspace</div>
  <h1>Make your next move<br>your best one.</h1>
  <p>A clearer picture of your placement readiness — plus practical next steps to help you grow into your target role.</p>
</div>
""", unsafe_allow_html=True)

# Calculate the current estimate for the landing preview too, so the page is useful
# before the student presses the report button.
match, gaps = skill_match_and_gaps(role, known)
row = pd.DataFrame([[cgpa, projects, dsa, internships, certs, match]], columns=FEATURES)
score = float(model.predict_proba(row)[0][1] * 100)
tips = profile_tips(cgpa, projects, dsa, internships, certs)

# Top row: clear score and next action.
hero_left, hero_right = st.columns([1.25, 1], gap="large")
with hero_left:
    st.markdown(f"""
    <div class="pr-panel">
      <div class="pr-eyebrow">Your readiness snapshot</div>
      <div class="pr-divider"></div>
      <div class="pr-muted">Estimated readiness score</div>
      <div class="pr-bigscore">{score:.0f}<span style="font-size:1.15rem;color:#a6afa0"> / 100</span></div>
      <div style="margin:.65rem 0 1rem"><span class="pr-chip">{label_for(score)}</span></div>
      <div class="pr-muted">Skill match for {role}</div>
      <div style="font-family:Manrope,sans-serif;font-size:1.45rem;font-weight:800;color:#f5f3eb">{match:.0%}</div>
      <div class="pr-muted" style="margin-top:.7rem">This is a practice estimate based on your profile, not a guarantee of hiring outcomes.</div>
    </div>
    """, unsafe_allow_html=True)
with hero_right:
    first_gap = gaps[0] if gaps else None
    if first_gap:
        next_title, _, next_tip = first_gap
        next_desc = next_tip
    elif tips:
        next_title, next_desc = "Strengthen your profile", tips[0]
    else:
        next_title, next_desc = "Polish your interview story", "Practice explaining your projects, decisions, and impact clearly."
    st.markdown(f"""
    <div class="pr-panel">
      <div class="pr-eyebrow">Recommended next step</div>
      <div style="font-size:2rem;margin:.55rem 0">✳</div>
      <div class="pr-panel-title">{next_title}</div>
      <div class="pr-muted">{next_desc}</div>
      <div class="pr-divider"></div>
      <div class="pr-muted">Start small. Focus on the highest-impact skill first, then update your profile as you progress.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="pr-eyebrow">Your career toolkit</div>', unsafe_allow_html=True)
st.markdown("### A little progress, every day")
tool_cols = st.columns(4, gap="medium")
tools_cards = [
    ("01", "DSA & coding", "Build problem-solving habits", "code"),
    ("02", "Aptitude", "Get faster with practice", "brain"),
    ("03", "Interview prep", "Practice telling your story", "chat"),
    ("04", "Career growth", "Track skills and milestones", "chart"),
]
for col, (num, title, desc, icon) in zip(tool_cols, tools_cards):
    with col:
        st.markdown(f"""
        <div class="pr-panel">
          <div class="pr-eyebrow">{num} / TOOLKIT</div>
          <div class="pr-panel-title" style="margin-top:.8rem">{title}</div>
          <div class="pr-muted">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

if go:
    st.markdown("<div class='pr-divider'></div>", unsafe_allow_html=True)
    st.markdown('<div class="pr-eyebrow">Personalized report</div>', unsafe_allow_html=True)
    st.header("Your readiness report")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Readiness score", f"{score:.0f}/100", label_for(score))
    metric_cols[1].metric("Role skill match", f"{match:.0%}")
    metric_cols[2].metric("Skills to explore", len(gaps))
    st.progress(min(int(score), 100), text="Readiness estimate")
    st.progress(int(match * 100), text="Role skill match")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("#### Skill gaps to prioritize")
        if gaps:
            gap_df = pd.DataFrame(gaps, columns=["Skill", "Importance", "What to try next"])
            for skill, importance, tip in gaps[:5]:
                st.markdown(f"""
                <div class="pr-panel" style="margin-bottom:.65rem;padding:1rem">
                  <div style="display:flex;justify-content:space-between;gap:1rem">
                    <div class="pr-panel-title">{skill}</div>
                    <span class="pr-chip">Priority {importance}/5</span>
                  </div>
                  <div class="pr-muted">{tip}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("You cover every key skill listed for this role. Keep building depth through projects and interviews.")
    with right:
        st.markdown("#### Your 30-day roadmap")
        if gaps:
            for i, (skill, _, tip) in enumerate(gaps[:3]):
                st.markdown(f"""
                <div class="pr-step">
                  <div class="pr-eyebrow">DAYS {i*10+1}–{i*10+10}</div>
                  <div class="pr-panel-title">{skill}</div>
                  <div class="pr-muted">{tip}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="pr-step"><div class="pr-eyebrow">WEEK 1–2</div><div class="pr-panel-title">Practice mock interviews</div><div class="pr-muted">Complete two mock interviews each week.</div></div>
            <div class="pr-step"><div class="pr-eyebrow">WEEK 3–4</div><div class="pr-panel-title">Polish your portfolio</div><div class="pr-muted">Improve project READMEs and prepare concise project explanations.</div></div>
            """, unsafe_allow_html=True)

    if tips:
        st.markdown("#### Profile boosters")
        for tip in tips:
            st.markdown(f"- {tip}")

    with st.expander("How should I interpret this score?"):
        st.write(
            "This prototype uses a Random Forest classifier trained on synthetic data. "
            f"Its held-out accuracy on that synthetic dataset is {acc:.0%}. This figure does not establish "
            "real-world placement prediction accuracy. Use the score as a demo estimate and the skill gaps "
            "as preparation guidance, not as a promise or hiring probability."
        )
else:
    with st.expander("How this estimate works"):
        st.write(
            "PlaceReady combines the profile values and weighted skill match to produce a model estimate. "
            "The current Random Forest model is trained on synthetic data, so its displayed test accuracy "
            f"({acc:.0%}) is only measured on that synthetic dataset. It is not evidence of real-world hiring accuracy."
        )

st.markdown(f"""
<div class="pr-footer">
  PLACEReady · Team Byte Blazers · INNOVATE X 2026<br>
  Prototype powered by a Random Forest model · Synthetic training data · Test accuracy on synthetic data: {acc:.0%}
</div>
""", unsafe_allow_html=True)
