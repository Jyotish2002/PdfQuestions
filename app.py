import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import datetime
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------------
# Local Question-Answer History (unchanged)
# ---------------------------
qa_history = {}

def log_question_answer(user_id, question, answer):
    if user_id not in qa_history:
        qa_history[user_id] = []
    qa_history[user_id].append({
        'question': question,
        'answer': answer,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def get_question_answer_history(user_id):
    return qa_history.get(user_id, [])

# ---------------------------
# PDF + LangChain functions (unchanged)
# ---------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in provided context just say, 
    "answer is not available in the context".
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question, user_id):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except ValueError as e:
        st.error(f"Error loading FAISS index: {e}")
        return
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    try:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        answer = response["output_text"]
        st.markdown(f"<div class='bot-bubble'>{answer}</div>", unsafe_allow_html=True)
        log_question_answer(user_id, user_question, answer)
    except Exception as e:
        st.error(f"Error generating response: {e}")

def get_gemini_response(question):
    model = genai.GenerativeModel("gemini-2.0-flash")
    chat = model.start_chat(history=[])
    response = chat.send_message(question, stream=True)
    return response

# ---------------------------
# App with Light/Dark Mode Toggle (sidebar upload UNCHANGED)
# ---------------------------
def main():
    st.set_page_config(page_title="Tech-Titans", layout="wide")

    # Theme state
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"  # default

    # Header + theme toggle
    left, right = st.columns([0.8, 0.2])
    with left:
        st.markdown(
            "<h1 style='margin-bottom:0'>⚡ Tech-Titans — AI PDF Assistant</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='margin-top:4px;opacity:0.8'>Ask book-accurate answers or switch to Gemini-powered explanations.</p>",
            unsafe_allow_html=True
        )
    with right:
        mode = st.toggle("🌙 Dark mode", value=(st.session_state.theme == "dark"))
        st.session_state.theme = "dark" if mode else "light"

    # THEME CSS (uses CSS variables; switches by injecting different variable sets)
    common_css = """
        <style>
            :root {
                --radius: 12px;
            }
            .stButton>button {
                border: none;
                border-radius: var(--radius);
                padding: 8px 16px;
                font-weight: 600;
                transition: transform .08s ease;
            }
            .stButton>button:active { transform: scale(0.98); }
            .card {
                border-radius: var(--radius);
                padding: 16px;
                border: 1px solid var(--border);
                background: var(--surface);
                margin-bottom: 16px;
            }
            .bot-bubble, .user-bubble {
                border-radius: 12px;
                padding: 12px 14px;
                margin: 6px 0;
                border: 1px solid var(--border);
            }
            .bot-bubble { background: var(--bubble-bot-bg); color: var(--bubble-bot-fg); }
            .user-bubble { background: var(--bubble-user-bg); color: var(--bubble-user-fg); text-align: right; }
            .stTextInput>div>div>input {
                border-radius: 10px;
                border: 1px solid var(--border);
                background: var(--input-bg);
                color: var(--fg);
            }
            .stTextInput>div>div>input::placeholder { color: var(--muted); }
            .main { background: var(--bg); color: var(--fg); }
        </style>
    """
    dark_vars = """
        <style>
            :root {
                --bg: #0d1117;
                --fg: #e6edf3;
                --muted: #9aa7b2;
                --surface: #0f1420;
                --border: #2b3240;
                --accent1: #00C9FF;
                --accent2: #92FE9D;
                --input-bg: #0f1522;
                --bubble-bot-bg: #121a2a;
                --bubble-bot-fg: #e6edf3;
                --bubble-user-bg: #1b2a46;
                --bubble-user-fg: #e6edf3;
            }
            .stButton>button {
                background: linear-gradient(90deg, var(--accent1), var(--accent2));
                color: #051016;
            }
        </style>
    """
    light_vars = """
        <style>
            :root {
                --bg: #f6f7fb;
                --fg: #0b1220;
                --muted: #6b7280;
                --surface: #ffffff;
                --border: #e6e8ef;
                --accent1: #2563eb;
                --accent2: #22c55e;
                --input-bg: #ffffff;
                --bubble-bot-bg: #f1f5ff;
                --bubble-bot-fg: #0b1220;
                --bubble-user-bg: #eefbf3;
                --bubble-user-fg: #0b1220;
            }
            .stButton>button {
                background: linear-gradient(90deg, var(--accent1), var(--accent2));
                color: #ffffff;
            }
        </style>
    """
    st.markdown(common_css, unsafe_allow_html=True)
    st.markdown(dark_vars if st.session_state.theme == "dark" else light_vars, unsafe_allow_html=True)

    user_id = "default_user"

    # ---------------- Main Area ----------------
    st.markdown("<div class='card'><h3>📄 Ask Questions from PDFs</h3></div>", unsafe_allow_html=True)
    user_question = st.text_input("Type your question (bookish answer from your PDFs):")
    if user_question:
        st.markdown(f"<div class='user-bubble'>{user_question}</div>", unsafe_allow_html=True)
        user_input(user_question, user_id)

    st.divider()

    st.markdown("<div class='card'><h3>🤖 Ask AI (Gemini) for Explanations</h3></div>", unsafe_allow_html=True)
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    chat_input = st.text_input("Ask anything (simplified/explanatory answers):", key="chat_input")
    send = st.button("🚀 Send")
    if send and chat_input:
        st.markdown(f"<div class='user-bubble'>{chat_input}</div>", unsafe_allow_html=True)
        resp = get_gemini_response(chat_input)
        resp_text = "".join([chunk.text for chunk in resp if getattr(chunk, 'text', None)]).strip()
        st.markdown(f"<div class='bot-bubble'>{resp_text}</div>", unsafe_allow_html=True)
        st.session_state['chat_history'].append(("You", chat_input))
        st.session_state['chat_history'].append(("Bot", resp_text))
        log_question_answer(user_id, chat_input, resp_text)

    with st.expander("📜 Chat History"):
        for role, text in st.session_state.get('chat_history', []):
            if role == "You":
                st.markdown(f"<div class='user-bubble'>{text}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-bubble'>{text}</div>", unsafe_allow_html=True)

    with st.expander("🗂 PDF Q&A History"):
        history = get_question_answer_history(user_id)
        if history:
            for item in history:
                st.markdown(
                    f"**Q:** {item['question']}  \n**A:** {item['answer']}  \n*🕒 {item['timestamp']}*"
                )
        else:
            st.write("No Q&A history found.")

    if st.button("💾 Save Chat History"):
        with open("chat_history.txt", "w") as f:
            for role, text in st.session_state.get('chat_history', []):
                f.write(f"{role}: {text}\n")
        st.success("Chat history saved!")
        with open("chat_history.txt", "rb") as f:
            st.download_button("⬇️ Download Chat History", data=f, file_name="chat_history.txt", mime="text/plain")

    # ---------------- Sidebar (UNCHANGED) ----------------
    with st.sidebar:
        st.title("📂 Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process", key="sidebar_process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
