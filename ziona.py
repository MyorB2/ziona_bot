import ast
import base64
import os

import streamlit as st
import pandas as pd
from business_logic.chatbot.react_agent import ReActAgent

from business_logic.classification.classification_wrapper import LoadedClassificationModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGOUND_HOME_DIR = os.path.join(BASE_DIR, 'resources', 'ziona_background_home.png')

st.set_page_config(page_title="Ziona Knows Your Reply", layout="wide")


# Convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Apply background using base64-encoded image
def set_bg_from_local(image_path):
    encoded_image = get_base64_image(image_path)
    st.markdown(
        """
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Apply background
# set_bg_from_local(BACKGOUND_HOME_DIR)

# Inject custom CSS
st.markdown("""
    <style>
    /* Style buttons */
    div.stButton > button {
        color: white;
        background-color: #007BFF;
        border: none;
        padding: 0.6em 1.2em;
        border-radius: 8px;
        transition: background-color 0.3s ease;
        font-size: 16px;
    }

    /* Hover effect */
    div.stButton > button:hover {
        background-color: #339AFF; /* Light blue */
        color: white;
    }

    /* Click (active) effect */
    div.stButton > button:active {
        background-color: #1C6DD0; /* Darker blue */
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# Load knowledge base CSV (only once)
@st.cache_data
def load_knowledge_base(path: str):
    df = pd.read_csv(KNOWLEDGE_BASE_PATH)
    df = df[['source', 'url', 'paragraph', 'categories']]
    df = df.dropna(subset=['source', 'url', 'paragraph'])
    df = df[df['url'].apply(lambda x: x.startswith("http"))]
    df.reset_index(drop=True, inplace=True)
    # category_id is list of integers
    df["primary_categories"] = df["primary_categories"].apply(lambda x: ast.literal_eval(x))
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

st.title("Welcome to Ziona's consulting App")
st.subheader("Tell me what is the problematic comment you accounted on social media, "
             "I will analys it and suggest an educational response.")

# UI Elements
comment = st.text_area("Enter a potentially anti-Semitic/ anti-Israeli comment:", height=150)

if st.button("Analyze & Generate Response"):
    if not comment.strip():
        st.warning("Please enter a comment.")
    else:
        with st.status("Classifying comment...", expanded=True) as status:
            # Step 1: Classify the comment
            classification_model = LoadedClassificationModel(r"./resources/meta_model_best.pkl")
            pred = classification_model.predict(comment)
            category_id = pred["predicted_labels"][0]
            category_name = LABEL_MAP[category_id]
            st.write(f"**Category ID:** {category_id}")
            st.write(f"**Category Name:** {category_name}")
            status.update(label="Comment classified", state="complete")

        with st.status("Retrieving resources and generating response...", expanded=True) as status:
            # Step 2: Generate the educational response
            KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, 'resources', 'knowledge_base.csv')
            knowledge_base = load_knowledge_base(KNOWLEDGE_BASE_PATH)
            agent = ReActAgent(knowledge_base)
            response = agent.generate_response(comment, category_id, category_name)

            st.write("### Selected Resource")
            st.markdown(f"- **Source**: {response['source']}")
            st.markdown(f"- **URL**: [{response['url']}]({response['url']})")
            st.markdown(f"- **Retrieval Score**: {response['retrieval_score']:.2f}")
            st.markdown(f"- **Relevant Paragraph**:\n\n> {response['final_response'][:300]}...")

            status.update(label="Response generated", state="complete")

        st.success("Final Response")
        st.write(response["final_response"])
