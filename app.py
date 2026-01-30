import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import datetime

# ===============================
# Load Environment Variables
# ===============================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found! Please set it in .env or Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# ===============================
# Chat History Storage
# ===============================

qa_history = {}


def log_question_answer(user_id, question, answer):
    if user_id not in qa_history:
        qa_history[user_id] = []

    qa_history[user_id].append({
        "question": question,
        "answer": answer,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


def get_question_answer_history(user_id):
    return qa_history.get(user_id, [])


# ===============================
# PDF Processing Functions
# ===============================

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


# ===============================
# Gemini QA Chain
# ===============================

def get_conversational_chain():
    prompt_template = """
    Answer the question using ONLY the given context.
    If answer is not present, say:
    "Answer is not available in the provided context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain


def user_question_answer(user_question, user_id):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    try:
        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except:
        st.error("❌ Please upload and process PDF first.")
        return

    docs = db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    answer = response["output_text"]

    st.success("✅ Answer Generated")
    st.write(answer)

    log_question_answer(user_id, user_question, answer)


# ===============================
# Gemini Normal Chat (No PDF)
# ===============================

def gemini_chat(question):
    model = genai.GenerativeModel("gemini-2.0-flash")
    chat = model.start_chat(history=[])
    response = chat.send_message(question)
    return response.text


# ===============================
# Streamlit UI
# ===============================

def main():
    st.set_page_config(page_title="PDF AI Chatbot", layout="wide")

    st.title("📄 PDF + Gemini AI Chatbot")
    st.write("Upload PDF → Ask Questions → Get AI Answers")

    user_id = "default_user"

    # ---------------- PDF Upload ----------------

    with st.sidebar:
        st.header("📂 Upload PDF Files")

        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("🚀 Process PDF"):
            if not pdf_docs:
                st.warning("⚠ Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success("✅ PDF Processing Complete")

    # ---------------- PDF QA ----------------

    st.subheader("🔍 Ask Question From PDF")

    user_question = st.text_input("Enter your PDF question:")

    if st.button("Ask PDF"):
        if user_question:
            user_question_answer(user_question, user_id)

    st.divider()

    # ---------------- Gemini Normal Chat ----------------

    st.subheader("🤖 Ask Gemini AI (General Chat)")

    gemini_input = st.text_input("Ask anything:")

    if st.button("Ask Gemini"):
        if gemini_input:
            reply = gemini_chat(gemini_input)
            st.success("✅ Gemini Response")
            st.write(reply)

    st.divider()

    # ---------------- History ----------------

    st.subheader("📜 Chat History")

    history = get_question_answer_history(user_id)

    if history:
        for item in history[::-1]:
            st.markdown(f"""
            **Q:** {item['question']}  
            **A:** {item['answer']}  
            ⏱ {item['time']}
            ---
            """)
    else:
        st.info("No history available")


if __name__ == "__main__":
    main()
