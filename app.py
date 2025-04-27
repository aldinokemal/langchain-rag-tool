import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("Chatbot with Google Generative AI Gemini Flash 1.5")

# Set credentials (you might want to use a more secure way in production)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'hydrogendioxide2104-5b3396eb8d75.json'

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load PDF
    loader = PDFPlumberLoader(tmp_path)
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = text_splitter.split_documents(documents)
    st.success(f"PDF berhasil diproses")

    # Embeddings dan FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(docs_split, embeddings)

    # UI untuk tanya jawab
    question = st.text_input("Tanyakan sesuatu dari PDF")
    if question:
        retriever = db.as_retriever()
        relevant_docs = retriever.get_relevant_documents(question)

        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        chain = load_qa_chain(model, chain_type="stuff")
        result = chain.run(input_documents=relevant_docs, question=question)

        st.markdown("### Jawaban:")
        st.write(result)

        with st.expander("📄 Dokumen yang relevan"):
            for i, doc in enumerate(relevant_docs):
                st.markdown(f"**Doc {i+1}**\n\n{doc.page_content[:500]}...")