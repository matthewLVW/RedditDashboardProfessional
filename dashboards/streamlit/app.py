# dashboards/streamlit_app/app.py

import streamlit as st
from streamlit_option_menu import option_menu

def load_image(path):
    from PIL import Image
    return Image.open(path)

st.set_page_config(page_title="Election NLP Dashboard", layout="wide")
st.title("Election NLP Dashboard (Demo)")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    subs = st.multiselect("Subreddits", ["politics", "conservative", "moderatepolitics", "politicaldiscussion"], default=["politics"])
    bin_win = st.slider("Bin Size (minutes)", 5, 240, 60)
    smoothing = st.selectbox("Smoothing Method", ["EWMA", "LOESS"])
    weight_mode = st.radio("Weighting Mode", ["Raw", "Equalized", "Inverse Frequency"])

# Navigation
tab = option_menu("Menu", ["Overview", "Timeline", "Semantic Explorer", "Topics & Chords", "Stance vs Emotion"], 
                  icons=["house", "line-chart", "search", "bar-chart-2", "flow-chart"], menu_icon="cast", default_index=0)

if tab == "Overview":
    st.header("Overview")
    # Key metrics cards (static values for demo)
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Sentiment", "0.15")
    col2.metric("% Sarcastic", "2.5%")
    col3.metric("Polarization (JS)", "0.123")

elif tab == "Timeline":
    st.header("Sentiment & Sarcasm Timeline")
    st.write("Click legend items to toggle series.")
    timeline_chart = load_image("dashboards/assets/timeline.png")
    st.image(timeline_chart, use_column_width=True)

elif tab == "Semantic Explorer":
    st.header("Semantic UMAP Explorer")
    st.write("Lasso-select points to read comment samples.")
    umap_chart = load_image("dashboards/assets/umap.png")
    st.image(umap_chart, use_column_width=True)

elif tab == "Topics & Chords":
    st.header("Topics and Sentiment Chord Diagram")
    chord_chart = load_image("dashboards/assets/chord.png")
    race_chart = load_image("dashboards/assets/topic_race.png")
    st.image(chord_chart, width=600)
    st.image(race_chart, width=600)

elif tab == "Stance vs Emotion":
    st.header("Stance vs. Dominant Emotion Sankey")
    sankey_chart = load_image("dashboards/assets/sankey.png")
    st.image(sankey_chart, width=800)
