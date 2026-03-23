import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import time
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────
st.set_page_config(
    page_title="NewsPulse",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── STYLES ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}
section[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d !important;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stMetric"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-top: 3px solid #00d4aa !important;
    padding: 1rem !important;
    border-radius: 10px !important;
}
[data-testid="stMetricValue"] {
    color: #00d4aa !important;
    font-size: 2rem !important;
}
[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #00b894) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #8b949e !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
.stTabs [aria-selected="true"] {
    color: #00d4aa !important;
    border-bottom: 2px solid #00d4aa !important;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #e6edf3 !important;
}
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background-color: #1f2937 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}
hr { border-color: #30363d !important; }
</style>
""", unsafe_allow_html=True)

# ── NLTK SETUP ───────────────────────────────
@st.cache_resource
def setup_nltk():
    packages = ["punkt", "punkt_tab", "stopwords", "wordnet", "vader_lexicon"]
    for p in packages:
        nltk.download(p, quiet=True)
    sw = set(stopwords.words("english"))
    sw.update(["said", "say", "one", "us", "would", "could", "may",
                "also", "get", "like", "new", "year", "make", "time"])
    return sw, WordNetLemmatizer(), SentimentIntensityAnalyzer()

STOPWORDS, LEMMA, SIA = setup_nltk()

# ── API KEY ───────────────────────────────────
try:
    NEWSAPI_KEY = st.secrets["NEWSAPI_KEY"]
except Exception:
    NEWSAPI_KEY = "1fc2ecd57b2c49dfb7c6343d85ad3696"

CATEGORIES = ["technology", "business", "health", "science",
              "sports", "entertainment", "general"]

# ── SESSION STATE ─────────────────────────────
defaults = {
    "logged_in": False,
    "username": "",
    "role": "",
    "df": None,
    "tfidf_data": None,
    "lda_topics": None,
    "fetch_meta": {},
    "watchlist_data": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── AUTH ──────────────────────────────────────
USERS = {
    "analyst": {"pwd": "pulse2024", "role": "analyst"},
    "editor":  {"pwd": "editor123", "role": "editor"},
    "admin":   {"pwd": "admin999",  "role": "admin"},
}

def authenticate(uname, pwd):
    u = USERS.get(uname.strip().lower())
    if u and u["pwd"] == pwd:
        return u["role"]
    return None

# ── TEXT PROCESSING ───────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^\x20-\x7E]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def preprocess(text):
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    tokens = [LEMMA.lemmatize(t) for t in tokens
              if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

def build_df(raw):
    df = raw.copy()
    df["Title"]       = df["Title"].apply(clean_text)
    df["Description"] = df["Description"].apply(clean_text)
    df = df[df["Title"].str.strip() != ""]
    df = df[df["Title"] != "[Removed]"]
    df = df.dropna(subset=["Title"])
    df = df.drop_duplicates(subset=["Title"])
    df["Published"] = pd.to_datetime(df["Published"], utc=True, errors="coerce")
    df = df.dropna(subset=["Published"])
    df["Date"]       = df["Published"].dt.date
    df["full_text"]  = df["Title"] + " " + df["Description"]
    df["clean_text"] = df["full_text"].apply(preprocess)
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
    return df

# ── ML FUNCTIONS ──────────────────────────────
def add_sentiment(df):
    df = df.copy()
    scores = df["clean_text"].apply(
        lambda t: SIA.polarity_scores(str(t))["compound"])
    df["sentiment_score"] = scores
    df["sentiment"] = scores.apply(
        lambda s: "Positive" if s >= 0.05 else ("Negative" if s <= -0.05 else "Neutral"))
    return df

def get_tfidf(df, n=20):
    try:
        vec = TfidfVectorizer(max_features=1000, stop_words="english",
                              ngram_range=(1, 2), min_df=2)
        mat = vec.fit_transform(df["clean_text"])
        names = vec.get_feature_names_out()
        scores = np.asarray(mat.mean(axis=0)).ravel()
        pairs = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
        return {w: float(s) for w, s in pairs[:n]}
    except Exception:
        return {}

def get_lda(df, n_topics=5):
    try:
        cv = CountVectorizer(max_features=500, stop_words="english",
                             max_df=0.7, min_df=2)
        dtm = cv.fit_transform(df["clean_text"])
        lda = LatentDirichletAllocation(n_components=n_topics,
                                        max_iter=20, random_state=42)
        lda.fit(dtm)
        words = cv.get_feature_names_out()
        topics = {}
        for i, comp in enumerate(lda.components_):
            top = [words[j] for j in comp.argsort()[-8:][::-1]]
            label = "Topic {}: {} & {}".format(i+1, top[0].title(), top[1].title())
            topics[label] = top
        return topics
    except Exception:
        return {}

def get_clusters(df, n=4):
    try:
        vec = TfidfVectorizer(max_features=300, stop_words="english")
        X = vec.fit_transform(df["clean_text"])
        km = KMeans(n_clusters=n, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        df = df.copy()
        df["cluster"] = labels
        order = km.cluster_centers_.argsort()[:, ::-1]
        terms = vec.get_feature_names_out()
        cmap = {}
        for i in range(n):
            top = [terms[idx] for idx in order[i, :3]]
            cmap[i] = " / ".join(t.title() for t in top)
        df["cluster_label"] = df["cluster"].map(cmap)
        return df
    except Exception:
        df = df.copy()
        df["cluster_label"] = "General"
        return df

# ── NEWS FETCHING ─────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_category(cats, size=50):
    rows = []
    for cat in cats:
        url = ("https://newsapi.org/v2/top-headlines?"
               "country=us&category={}&pageSize={}&apiKey={}".format(
                   cat, size, NEWSAPI_KEY))
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 200:
                for a in r.json().get("articles", []):
                    rows.append({
                        "Title":       a.get("title", ""),
                        "Description": a.get("description", "") or "",
                        "Source":      a.get("source", {}).get("name", "Unknown"),
                        "Published":   a.get("publishedAt", ""),
                        "Category":    cat,
                        "URL":         a.get("url", ""),
                    })
        except Exception:
            pass
        time.sleep(0.3)
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_keywords(kws, from_d, to_d):
    rows = []
    for kw in kws:
        url = ("https://newsapi.org/v2/everything?q={}"
               "&from={}&to={}&pageSize=50&sortBy=relevancy&apiKey={}".format(
                   kw, from_d, to_d, NEWSAPI_KEY))
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 200:
                for a in r.json().get("articles", []):
                    rows.append({
                        "Title":       a.get("title", ""),
                        "Description": a.get("description", "") or "",
                        "Source":      a.get("source", {}).get("name", "Unknown"),
                        "Published":   a.get("publishedAt", ""),
                        "Category":    kw,
                        "URL":         a.get("url", ""),
                    })
        except Exception:
            pass
        time.sleep(0.3)
    return pd.DataFrame(rows)

# ── PLOTLY THEME ──────────────────────────────
PALETTE = ["#00d4aa", "#f39c12", "#e74c3c", "#3498db",
           "#9b59b6", "#1abc9c", "#e67e22", "#2ecc71"]

def theme(**kw):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#8b949e", size=12),
        title_font=dict(family="DM Serif Display, serif", color="#e6edf3", size=16),
        colorway=PALETTE,
        xaxis=dict(gridcolor="#1f2937", linecolor="#30363d"),
        yaxis=dict(gridcolor="#1f2937", linecolor="#30363d"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d"),
        margin=dict(l=40, r=20, t=55, b=40),
    )
    base.update(kw)
    return base

# ── UI HELPERS ────────────────────────────────
def header(title, sub=""):
    sub_html = ""
    if sub:
        sub_html = "<p style='color:#8b949e; font-size:.85rem; margin:.3rem 0 0;'>{}</p>".format(sub)
    st.markdown(
        "<div style='padding-bottom:1rem; border-bottom:1px solid #30363d; margin-bottom:1.5rem;'>"
        "<h2 style='margin:0; color:#e6edf3;'>{}</h2>{}</div>".format(title, sub_html),
        unsafe_allow_html=True
    )

def card(row, idx):
    colors = {"Positive": "#3fb950", "Negative": "#f85149", "Neutral": "#d29922"}
    label  = str(row.get("sentiment", "Neutral"))
    score  = float(row.get("sentiment_score", 0))
    bc     = colors.get(label, "#30363d")
    title  = str(row.get("Title", ""))[:150]
    source = str(row.get("Source", ""))
    cat    = str(row.get("Category", "")).upper()
    date   = str(row.get("Date", ""))
    st.markdown(
        "<div style='background:#161b22; border:1px solid #30363d; "
        "border-left:4px solid {bc}; border-radius:10px; padding:1.2rem 1.5rem; margin:.5rem 0;'>"
        "<div style='display:flex; justify-content:space-between;'>"
        "<span style='font-size:.68rem; color:#00d4aa; text-transform:uppercase; "
        "letter-spacing:.08em;'>{src} · {cat}</span>"
        "<span style='font-size:.65rem; color:#8b949e;'>{date}</span></div>"
        "<h4 style='font-family:\"DM Serif Display\",serif; font-size:1rem; "
        "color:#e6edf3; margin:.5rem 0; font-weight:400; line-height:1.5;'>{title}</h4>"
        "<span style='font-size:.72rem; color:{bc}; font-weight:600; text-transform:uppercase;'>"
        "● {label} &nbsp; Score: {score:.3f}</span>"
        "</div>".format(bc=bc, src=source, cat=cat, date=date,
                        title=title, label=label, score=score),
        unsafe_allow_html=True
    )

# ── LOGIN PAGE ────────────────────────────────
def show_login():
    st.markdown(
        "<div style='text-align:center; padding:4rem 0 2rem;'>"
        "<div style='font-size:4rem;'>🌐</div>"
        "<h1 style='font-family:\"DM Serif Display\",serif; color:#00d4aa; "
        "font-size:3rem; margin:.5rem 0;'>NewsPulse</h1>"
        "<p style='color:#8b949e; margin:.5rem 0 0;'>"
        "Global News Trend Analyzer · Powered by AI</p></div>",
        unsafe_allow_html=True
    )
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown(
            "<div style='background:#161b22; border:1px solid #30363d; "
            "border-radius:14px; padding:2rem; margin-top:1.5rem;'>",
            unsafe_allow_html=True
        )
        uname = st.text_input("Username", placeholder="analyst / editor / admin")
        pwd   = st.text_input("Password", type="password", placeholder="password")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign In", use_container_width=True, type="primary"):
            role = authenticate(uname, pwd)
            if role:
                st.session_state.logged_in = True
                st.session_state.username  = uname
                st.session_state.role      = role
                st.rerun()
            else:
                st.error("Invalid credentials. Try analyst / pulse2024")
        st.markdown(
            "<p style='font-size:.72rem; color:#8b949e; text-align:center; margin-top:1rem;'>"
            "analyst / pulse2024 &nbsp;|&nbsp; admin / admin999</p></div>",
            unsafe_allow_html=True
        )

# ── SIDEBAR ───────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center; padding:1rem 0;'>"
            "<div style='font-size:2rem;'>🌐</div>"
            "<h2 style='color:#00d4aa; font-family:\"DM Serif Display\",serif; "
            "font-size:1.5rem; margin:.3rem 0;'>NewsPulse</h2>"
            "<p style='color:#8b949e; font-size:.65rem; text-transform:uppercase; "
            "letter-spacing:.06em;'>AI News Analyzer</p></div>",
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown(
            "<div style='background:#1f2937; border-radius:8px; padding:.8rem; margin-bottom:1rem;'>"
            "<b style='color:#e6edf3;'>@{}</b><br>"
            "<span style='color:#00d4aa; font-size:.7rem; text-transform:uppercase;'>{}</span>"
            "</div>".format(st.session_state.username, st.session_state.role),
            unsafe_allow_html=True
        )

        st.markdown("### Data Source")
        mode = st.radio("Mode", ["Category Browse", "Keyword Search"],
                        label_visibility="collapsed")

        cats, kws, fd, td = [], [], None, None

        if mode == "Category Browse":
            cats = st.multiselect("Categories", CATEGORIES,
                                  default=["technology", "business", "health"])
        else:
            raw = st.text_input("Keywords", value="AI, climate, economy",
                                label_visibility="collapsed")
            kws = [k.strip() for k in raw.split(",") if k.strip()][:5]
            today = datetime.utcnow().date()
            c1, c2 = st.columns(2)
            fd = c1.date_input("From", value=today - timedelta(days=14),
                               label_visibility="collapsed")
            td = c2.date_input("To", value=today, label_visibility="collapsed")

        st.markdown("### ML Settings")
        n_topics   = st.slider("LDA Topics",  3, 8, 5)
        n_clusters = st.slider("K-Clusters",  2, 6, 4)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Fetch & Analyze", use_container_width=True, type="primary"):
            raw = pd.DataFrame()
            with st.spinner("Fetching news..."):
                if mode == "Category Browse" and cats:
                    raw = fetch_category(tuple(cats))
                elif mode == "Keyword Search" and kws:
                    raw = fetch_keywords(tuple(kws), fd.isoformat(), td.isoformat())
                else:
                    st.error("Select a category or enter keywords first.")

            if not raw.empty:
                with st.spinner("Processing and analyzing..."):
                    df     = build_df(raw)
                    df     = add_sentiment(df)
                    tfidf  = get_tfidf(df)
                    topics = get_lda(df, n_topics)
                    df     = get_clusters(df, n_clusters)

                st.session_state.df         = df
                st.session_state.tfidf_data = tfidf
                st.session_state.lda_topics = topics
                st.session_state.fetch_meta = {
                    "total": len(df),
                    "at": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                st.success("{} articles ready!".format(len(df)))
                st.rerun()

        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()

# ── OVERVIEW TAB ──────────────────────────────
def tab_overview(df, tfidf):
    meta = st.session_state.fetch_meta
    header("Global News Overview",
           "{} articles analysed · {}".format(meta.get("total", 0), meta.get("at", "")))

    c1, c2, c3, c4, c5 = st.columns(5)
    pos = len(df[df["sentiment"] == "Positive"])
    neg = len(df[df["sentiment"] == "Negative"])
    c1.metric("Articles",   str(len(df)))
    c2.metric("Sources",    str(df["Source"].nunique()))
    c3.metric("Categories", str(df["Category"].nunique()))
    c4.metric("Positive",   str(pos))
    c5.metric("Negative",   str(neg))

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        cat_cnt = df["Category"].value_counts().reset_index()
        cat_cnt.columns = ["Category", "Count"]
        fig = px.bar(cat_cnt, x="Category", y="Count",
                     color="Category", color_discrete_sequence=PALETTE,
                     title="Articles by Category")
        fig.update_layout(**theme(height=360, showlegend=False))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        src = df["Source"].value_counts().head(10).reset_index()
        src.columns = ["Source", "Count"]
        fig2 = px.bar(src, x="Count", y="Source", orientation="h",
                      color="Count", color_continuous_scale="Teal",
                      title="Top 10 Sources")
        fig2.update_layout(**theme(
            height=360, showlegend=False, coloraxis_showscale=False,
            yaxis=dict(autorange="reversed", gridcolor="#1f2937")))
        st.plotly_chart(fig2, use_container_width=True)

    if tfidf:
        st.markdown("#### Top Keywords (TF-IDF)")
        kdf = pd.DataFrame(list(tfidf.items())[:20], columns=["Keyword", "Score"])
        fig3 = px.bar(kdf, x="Keyword", y="Score",
                      color="Score", color_continuous_scale="Teal",
                      title="Keyword Importance Score")
        fig3.update_layout(**theme(height=320, showlegend=False,
                                   coloraxis_showscale=False))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### Recent Articles")
    sample = df.groupby("Category").head(2).head(12).reset_index(drop=True)
    cols = st.columns(2)
    for i, (_, row) in enumerate(sample.iterrows()):
        with cols[i % 2]:
            card(row, i)

# ── TRENDS TAB ────────────────────────────────
def tab_trends(df):
    header("Trend Analysis", "Volume and sentiment patterns over time")

    col1, col2 = st.columns(2)
    with col1:
        vol = df.groupby(["Date", "Category"]).size().reset_index(name="Count")
        fig = px.line(vol, x="Date", y="Count", color="Category",
                      markers=True, title="Daily Article Volume")
        fig.update_layout(**theme(height=380))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st_data = df.groupby("Date")["sentiment_score"].mean().reset_index()
        st_data.columns = ["Date", "Score"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=st_data["Date"], y=st_data["Score"],
            mode="lines+markers",
            line=dict(color="#00d4aa", width=3),
            marker=dict(size=8, color="#f39c12"),
            fill="tozeroy",
            fillcolor="rgba(0,212,170,.10)",
        ))
        fig2.add_hline(y=0.05,  line_dash="dot", line_color="#3fb950")
        fig2.add_hline(y=-0.05, line_dash="dot", line_color="#f85149")
        fig2.update_layout(title="Sentiment Over Time", **theme(height=380))
        st.plotly_chart(fig2, use_container_width=True)

    pivot = df.groupby(["Category", "sentiment"]).size().reset_index(name="Count")
    fig3 = px.bar(pivot, x="Category", y="Count", color="sentiment",
                  barmode="stack",
                  color_discrete_map={"Positive": "#3fb950",
                                      "Negative": "#f85149",
                                      "Neutral":  "#d29922"},
                  title="Sentiment Breakdown by Category")
    fig3.update_layout(**theme(height=360))
    st.plotly_chart(fig3, use_container_width=True)

# ── TOPICS TAB ────────────────────────────────
def tab_topics(df, topics):
    header("Topic Discovery", "LDA topics and K-Means clusters")

    st.markdown("#### LDA Topics")
    COLORS = ["#00d4aa","#f39c12","#e74c3c","#3498db",
               "#9b59b6","#1abc9c","#e67e22","#2ecc71"]
    cols = st.columns(2)
    for i, (name, kws) in enumerate(topics.items()):
        c = COLORS[i % len(COLORS)]
        tags = " ".join(
            "<span style='background:#1f2937; border:1px solid {c}44; color:{c}; "
            "padding:.3rem .7rem; border-radius:20px; font-size:.78rem; "
            "display:inline-block; margin:.2rem;'>{w}</span>".format(c=c, w=w)
            for w in kws
        )
        with cols[i % 2]:
            st.markdown(
                "<div style='background:#161b22; border:1px solid #30363d; "
                "border-left:4px solid {c}; border-radius:10px; "
                "padding:1.3rem; margin:.5rem 0;'>"
                "<p style='color:#8b949e; font-size:.65rem; text-transform:uppercase; "
                "margin:0 0 .4rem;'>LDA Topic</p>"
                "<h4 style='color:#e6edf3; font-family:\"DM Serif Display\",serif; "
                "margin:0 0 .8rem; font-size:1rem;'>{name}</h4>"
                "<div>{tags}</div></div>".format(c=c, name=name, tags=tags),
                unsafe_allow_html=True
            )

    if "cluster_label" in df.columns:
        st.markdown("#### K-Means Clusters")
        cl = df["cluster_label"].value_counts().reset_index()
        cl.columns = ["Cluster", "Count"]
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(cl, names="Cluster", values="Count",
                         color_discrete_sequence=PALETTE, hole=0.45,
                         title="Cluster Distribution")
            fig.update_layout(**theme(height=360))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(cl, x="Cluster", y="Count",
                          color="Cluster", color_discrete_sequence=PALETTE,
                          title="Articles per Cluster")
            fig2.update_layout(**theme(height=360, showlegend=False))
            st.plotly_chart(fig2, use_container_width=True)

# ── SENTIMENT TAB ─────────────────────────────
def tab_sentiment(df):
    header("Sentiment Analysis", "VADER scoring — Positive >= 0.05 | Negative <= -0.05")

    dist  = df["sentiment"].value_counts()
    total = len(df)
    pos   = int(dist.get("Positive", 0))
    neu   = int(dist.get("Neutral",  0))
    neg   = int(dist.get("Negative", 0))
    cmap  = {"Positive":"#3fb950","Neutral":"#d29922","Negative":"#f85149"}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positive",   str(pos), "{:.1f}%".format(pos/total*100))
    c2.metric("Neutral",    str(neu), "{:.1f}%".format(neu/total*100))
    c3.metric("Negative",   str(neg), "{:.1f}%".format(neg/total*100))
    c4.metric("Mean Score", "{:.3f}".format(df["sentiment_score"].mean()))

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(values=dist.values, names=dist.index,
                     color=dist.index, color_discrete_map=cmap,
                     hole=0.55, title="Sentiment Breakdown")
        fig.update_layout(**theme(height=400))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(df, x="sentiment_score", nbins=40,
                            color="sentiment", color_discrete_map=cmap,
                            title="Score Distribution")
        fig2.add_vline(x=0.05,  line_dash="dot", line_color="#3fb950")
        fig2.add_vline(x=-0.05, line_dash="dot", line_color="#f85149")
        fig2.update_layout(**theme(height=400))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Most Positive and Negative Articles")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Positive**")
        for i, (_, row) in enumerate(df.nlargest(4, "sentiment_score").iterrows()):
            card(row, "p{}".format(i))
    with col2:
        st.markdown("**Top Negative**")
        for i, (_, row) in enumerate(df.nsmallest(4, "sentiment_score").iterrows()):
            card(row, "n{}".format(i))

# ── ARTICLES TAB ──────────────────────────────
def tab_articles(df):
    header("Article Browser", "Search and filter all articles")

    c1, c2, c3 = st.columns(3)
    sel_cat  = c1.selectbox("Category", ["All"] + sorted(df["Category"].unique().tolist()))
    sel_sent = c2.selectbox("Sentiment", ["All","Positive","Neutral","Negative"])
    search   = c3.text_input("Search titles", placeholder="Type keyword...")

    filt = df.copy()
    if sel_cat  != "All":
        filt = filt[filt["Category"] == sel_cat]
    if sel_sent != "All":
        filt = filt[filt["sentiment"] == sel_sent]
    if search.strip():
        filt = filt[filt["Title"].str.contains(search, case=False, na=False)]

    st.markdown(
        "<p style='color:#8b949e; font-size:.8rem;'>Showing {} articles</p>".format(len(filt)),
        unsafe_allow_html=True
    )

    per_page    = 10
    total_pages = max(1, (len(filt) - 1) // per_page + 1)
    pg    = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start = (pg - 1) * per_page
    for i, (_, row) in enumerate(filt.iloc[start:start+per_page].iterrows()):
        card(row, "b{}".format(start+i))

    st.markdown("---")
    csv = df.to_csv(index=False).encode()
    st.download_button("Download CSV", data=csv,
                       file_name="newspulse.csv", mime="text/csv")

# ── ADMIN TAB ─────────────────────────────────
def tab_admin(df):
    header("Admin Dashboard", "Data quality and system overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", str(len(df)))
    c2.metric("Duplicates",    str(df.duplicated(subset=["Title"]).sum()))
    c3.metric("Sources",       str(df["Source"].nunique()))

    st.markdown("#### Raw Data Preview")
    st.dataframe(df[["Title","Source","Category","Date",
                      "sentiment","sentiment_score"]].head(30),
                 use_container_width=True)

    csv = df.to_csv(index=False).encode()
    st.download_button("Download Full Dataset", data=csv,
                       file_name="newspulse_full.csv", mime="text/csv")

    if st.button("Clear Data", type="primary"):
        st.session_state.df = None
        st.session_state.tfidf_data = None
        st.session_state.lda_topics = None
        st.rerun()

# ── EMPTY STATE ───────────────────────────────
def empty_state():
    st.markdown(
        "<div style='text-align:center; padding:8rem 2rem;'>"
        "<div style='font-size:4rem;'>📡</div>"
        "<h2 style='color:#8b949e; font-family:\"DM Serif Display\",serif; font-weight:400;'>"
        "No data yet</h2>"
        "<p style='color:#8b949e;'>Select categories or keywords in the sidebar "
        "and click <b style='color:#00d4aa;'>Fetch and Analyze</b></p>"
        "</div>",
        unsafe_allow_html=True
    )

# ── MAIN APP ──────────────────────────────────
def show_app():
    sidebar()

    meta  = st.session_state.fetch_meta
    badge = ""
    if meta:
        badge = (
            " <span style='background:#00d4aa22; color:#00d4aa; font-size:.65rem; "
            "padding:.2rem .7rem; border-radius:20px; border:1px solid #00d4aa;'>"
            "LIVE {} articles</span>".format(meta.get("total", 0))
        )

    st.markdown(
        "<h1 style='font-family:\"DM Serif Display\",serif; font-size:1.9rem; "
        "color:#e6edf3; margin:0; padding:.4rem 0; font-weight:400;'>"
        "🌐 NewsPulse {}</h1>".format(badge),
        unsafe_allow_html=True
    )
    st.markdown("---")

    df = st.session_state.df
    if df is None:
        empty_state()
        return

    tfidf  = st.session_state.tfidf_data
    topics = st.session_state.lda_topics

    labels = ["Overview", "Trends", "Topics", "Sentiment", "Articles"]
    if st.session_state.role == "admin":
        labels.append("Admin")

    tabs = st.tabs(labels)
    with tabs[0]: tab_overview(df, tfidf)
    with tabs[1]: tab_trends(df)
    with tabs[2]: tab_topics(df, topics)
    with tabs[3]: tab_sentiment(df)
    with tabs[4]: tab_articles(df)
    if len(tabs) > 5:
        with tabs[5]: tab_admin(df)

# ── ENTRY POINT ───────────────────────────────
if not st.session_state.logged_in:
    show_login()
else:
    show_app()
