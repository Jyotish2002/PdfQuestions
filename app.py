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
# Main Tech-Themed UI
# ---------------------------
def main():
    st.set_page_config(page_title="Tech-Titans", layout="wide")

    # Dark tech theme CSS
    st.markdown("""
        <style>
            .main {background-color: #0d1117; color: #e6edf3;}
            .stButton>button {
                background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
                color: black;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                border: none;
            }
            .stButton>button:hover {transform: scale(1.05);}
            .stTextInput>div>div>input {
                background-color: #161b22;
                color: #e6edf3;
                border-radius: 8px;
                border: 1px solid #30363d;
            }
            .bot-bubble {
                background: #1f6feb;
                color: white;
                padding: 12px;
                border-radius: 10px;
                margin: 6px 0;
            }
            .user-bubble {
                background: #2ea043;
                color: white;
                padding: 12px;
                border-radius: 10px;
                margin: 6px 0;
                text-align: right;
            }
            .card {
                background: #161b22;
                padding: 15px;
                border-radius: 12px;
                border: 1px solid #30363d;
                margin-bottom: 15px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("<h1 style='color:#00C9FF;text-align:center;'>⚡ Tech-Titans - AI PDF Assistant ⚡</h1>", unsafe_allow_html=True)

    user_id = "default_user"

    # PDF Q&A
    st.markdown("<div class='card'><h3>📄 Ask Questions from PDFs</h3></div>", unsafe_allow_html=True)
    user_question = st.text_input("Type your question:")
    if user_question:
        st.markdown(f"<div class='user-bubble'>{user_question}</div>", unsafe_allow_html=True)
        user_input(user_question, user_id)

    st.divider()

    # Chat interface
    st.markdown("<div class='card'><h3>🤖 Chat with AI</h3></div>", unsafe_allow_html=True)
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    chat_input = st.text_input("Ask me anything:", key="chat_input")
    submit_chat = st.button("🚀 Send")

    if submit_chat and chat_input:
        st.markdown(f"<div class='user-bubble'>{chat_input}</div>", unsafe_allow_html=True)
        response = get_gemini_response(chat_input)
        response_text = "".join([chunk.text for chunk in response if chunk.text]).strip()
        st.markdown(f"<div class='bot-bubble'>{response_text}</div>", unsafe_allow_html=True)

        st.session_state['chat_history'].append(("You", chat_input))
        st.session_state['chat_history'].append(("Bot", response_text))
        log_question_answer(user_id, chat_input, response_text)

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
                st.markdown(f"**Q:** {item['question']}  \n**A:** {item['answer']}  \n*🕒 {item['timestamp']}*")
        else:
            st.write("No Q&A history found.")

    if st.button("💾 Save Chat History"):
        with open("chat_history.txt", "w") as file:
            for role, text in st.session_state.get('chat_history', []):
                file.write(f"{role}: {text}\n")
        st.success("Chat history saved!")

        with open("chat_history.txt", "rb") as file:
            st.download_button(
                label="⬇️ Download Chat History",
                data=file,
                file_name="chat_history.txt",
                mime="text/plain",
            )

    # Sidebar (UNCHANGED)
    with st.sidebar:
        st.title("📂 Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on Submit & Process", accept_multiple_files=True)
        if st.button("Submit & Process", key="sidebar_process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ PDFs processed successfully!")

if __name__ == "__main__":
    main()
