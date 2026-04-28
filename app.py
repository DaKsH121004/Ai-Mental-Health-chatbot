# =============================
# IMPORTS
# =============================
import streamlit as st
import numpy as np
import faiss
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from sentence_transformers import SentenceTransformer

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Mental Health Chatbot", layout="centered")

st.title("🧠 AI Mental Health Chatbot")
st.markdown("Talk freely. This system provides supportive guidance.")

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_resources():
    try:
        model = SentenceTransformer("sbert_model")
        index = faiss.read_index("faiss_index.bin")

        with open("data.pkl", "rb") as f:
            data = pickle.load(f)

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    return model, index, data

model, index, data = load_resources()

questions = data["questions"]
answers = data["answers"]
labels = data["labels"]

# =============================
# RESPONSE LOGIC
# =============================
def generate_response(label):
    label = label.lower()

    if label == "suicidal":
        return "I'm really concerned about you. Please reach out to someone you trust or a professional immediately."

    elif label == "depression":
        return "It sounds like you're going through a tough time. You're not alone."

    elif label == "anxiety":
        return "It seems like you're feeling anxious. Try slow breathing or grounding techniques."

    elif label == "stress":
        return "You seem stressed. Taking breaks and organizing tasks can help."

    else:
        return "I'm here to listen. Tell me more."

# =============================
# CHAT FUNCTION
# =============================
def chatbot(query):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), 1)

    idx = indices[0][0]
    distance = distances[0][0]   # 👈 IMPORTANT FOR GRAPH
    label = labels[idx]

    response = generate_response(label)

    return response, label, distance

# =============================
# SESSION STATE
# =============================
if "history" not in st.session_state:
    st.session_state.history = []

# =============================
# INPUT
# =============================
user_input = st.chat_input("Type your message...")

if user_input:
    response, label, distance = chatbot(user_input)

    st.session_state.history.append({
        "user": user_input,
        "bot": response,
        "label": label,
        "distance": distance,
        "time": datetime.now()
    })

# =============================
# DISPLAY CHAT
# =============================
for chat in st.session_state.history:
    with st.chat_message("user"):
        st.write(chat["user"])

    with st.chat_message("assistant"):
        st.write(chat["bot"])

    st.caption(f"Detected: {chat['label']} | Similarity Distance: {chat['distance']:.4f}")

# =============================
# 📊 ANALYTICS SECTION
# =============================
st.divider()
st.subheader("📊 Conversation Analytics (For Research)")

if len(st.session_state.history) > 0:

    labels_list = [c["label"] for c in st.session_state.history]
    distances_list = [c["distance"] for c in st.session_state.history]
    time_steps = list(range(1, len(labels_list)+1))

    # -----------------------------
    # 1. Emotion Distribution
    # -----------------------------
    st.markdown("### Emotion Distribution")
    label_counts = Counter(labels_list)

    fig1 = plt.figure()
    plt.bar(label_counts.keys(), label_counts.values())
    plt.xlabel("Emotion")
    plt.ylabel("Frequency")
    plt.title("Detected Emotion Distribution")
    st.pyplot(fig1)

    # -----------------------------
    # 2. Emotion Timeline
    # -----------------------------
    st.markdown("### Emotion Timeline")

    fig2 = plt.figure()
    plt.plot(time_steps, labels_list, marker='o')
    plt.xlabel("Message Index")
    plt.ylabel("Emotion")
    plt.title("Emotion Trend Over Conversation")
    st.pyplot(fig2)

    # -----------------------------
    # 3. Distance (Confidence Proxy)
    # -----------------------------
    st.markdown("### Similarity Distance Trend (Lower = Better Match)")

    fig3 = plt.figure()
    plt.plot(time_steps, distances_list, marker='o')
    plt.xlabel("Message Index")
    plt.ylabel("Distance")
    plt.title("FAISS Similarity Distance Over Time")
    st.pyplot(fig3)

else:
    st.info("Start chatting to generate analytics.")