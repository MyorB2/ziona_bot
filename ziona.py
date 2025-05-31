import ast
import os

import streamlit as st
import pandas as pd
from business_logic.chatbot.react_agent import ReActAgent
from business_logic.classification.classification_wrapper import LoadedClassificationModel


# Load knowledge base CSV (only once)
@st.cache_data
def load_knowledge_base(path: str):
    df = pd.read_csv(KNOWLEDGE_BASE_PATH)
    df = df[['source', 'url', 'paragraph', 'categories']]
    df = df.dropna(subset=['source', 'url', 'paragraph'])
    df = df[df['url'].apply(lambda x: x.startswith("http"))]
    df.reset_index(drop=True, inplace=True)
    # category_id is list of integers
    df["categories"] = df["categories"].apply(lambda x: ast.literal_eval(x))
    return df


# Categories explanation
# 1: Calling for annihilation, theory conspiracies, identify Jewish people as evil.
# 2: Economic stereotypes, biological racism, humiliating external imaging.
# 3: Demonize Israel, compare Israel and Zionism to Nazis, anti-Zionism, denial of the right to exist.
# 4: Be cynical about the Holocaust, Holocaust denial.
# 5: Mention of Jewish or Israeli public figures along with antisemitic connotation, Implying of Jewish people.
LABEL_MAP = {1: "antisemitic ideology", 2: "stereotypes and dehumanisation",
             3: "antisemitism against Israel or Zionism", 4: "Holocaust or Zionism denial",
             5: "indirect antisemitism or secondary objective"}

st.set_page_config(page_title="Educational Response Generator", layout="wide")
st.title("Welcome to Ziona's consulting App")
st.subheader("Tell me what is the problematic comment you accounted on social media, and i will analys it and suggest an educational response")

# UI Elements
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, 'resources', 'knowledge_base.csv')
knowledge_base = load_knowledge_base(KNOWLEDGE_BASE_PATH)
agent = ReActAgent(knowledge_base)

comment = st.text_area("Enter a potentially antisemitic comment:", height=150)

if st.button("Analyze & Generate Response"):
    with st.spinner("Retrieving knowledge and generating response..."):
        classification_model = LoadedClassificationModel(r"./models/Multi_model/1805_1/meta_model_best.pkl")
        pred = classification_model.predict(comment)
        category_id = pred["predicted_labels"][0]
        category_name = LABEL_MAP[category_id]
        result = agent.generate_response(comment, category_id, category_name)

        st.subheader("Step 1: Retrieval and Context")
        st.markdown(f"**Comment:** {comment}")
        st.markdown(f"**Classification:** {category_name}")
        st.markdown(f"**Source:** {result['source']}")
        st.markdown(f"**URL:** {result['url']}")
        st.markdown(f"**Retrieval Score:** {result.get('retrieval_score', 'N/A'):.3f}")

        st.subheader("Step 2: Suggested Educational Response")
        st.success(result["final_response"])
