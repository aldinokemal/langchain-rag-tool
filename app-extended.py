import streamlit as st
import os
import tempfile
from pathlib import Path
import re # Added for sanitizing filename
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.retrievers import MergerRetriever # Added for querying multiple indexes

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("Chatbot with Google Generative AI Gemini Flash 2.5")

# --- Configuration ---
FAISS_INDEX_DIR = Path("faiss_indexes")
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists

# Set credentials (you might want to use a more secure way in production)
# Check if the credentials file exists
CREDENTIALS_PATH = 'hydrogendioxide2104-5b3396eb8d75.json'
if os.path.exists(CREDENTIALS_PATH):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH
    try:
        # Initialize embeddings early
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        google_creds_valid = True
    except Exception as e:
        st.error(f"Failed to initialize Google Embeddings: {e}")
        google_creds_valid = False
else:
    st.error(f"Credentials file not found at {CREDENTIALS_PATH}. Please provide valid credentials.")
    google_creds_valid = False
    embeddings = None # Set embeddings to None if creds are invalid

# --- Helper Functions ---
def sanitize_filename(filename: str) -> str:
    """Creates a safe directory name from a filename."""
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]
    # Replace non-alphanumeric characters (except underscore) with underscore
    sanitized = re.sub(r'[^\w-]+', '_', name_without_ext)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Limit length (optional, but good practice)
    return sanitized[:50] or "default_doc" # Ensure it's not empty

def get_available_documents() -> list[str]:
    """Lists available document IDs (index directory names)."""
    if not FAISS_INDEX_DIR.exists():
        return []
    return [d.name for d in FAISS_INDEX_DIR.iterdir() if d.is_dir()]

# --- Main App Logic ---
if google_creds_valid: # Only proceed if embeddings are available
    # Always check for available documents first
    available_docs = get_available_documents()

    col1, col2 = st.columns(2)

    # Column 1: PDF Upload (Optional)
    with col1:
        st.header("Upload New PDF (Optional)")
        uploaded_file = st.file_uploader("Upload PDF to add it to the knowledge base", type="pdf", key="pdf_uploader")

        if uploaded_file is not None:
            document_id = sanitize_filename(uploaded_file.name)
            index_path = FAISS_INDEX_DIR / document_id

            # Prevent reprocessing if index already exists (optional, but good practice)
            if index_path.exists():
                 st.warning(f"An index for a document named '{document_id}' already exists. Upload with a different name or manage indexes manually if you want to replace it.")
            else:
                with st.spinner(f"Processing '{uploaded_file.name}'..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name

                        # Load PDF
                        loader = PDFPlumberLoader(tmp_path)
                        documents = loader.load()

                        # Split documents
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        docs_split = text_splitter.split_documents(documents)

                        # Embeddings dan FAISS
                        db = FAISS.from_documents(docs_split, embeddings)

                        # Save FAISS index locally
                        db.save_local(str(index_path))

                        st.success(f"PDF '{uploaded_file.name}' (ID: {document_id}) processed and saved.")
                        # Clear uploader and refresh available docs - triggers rerun
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")
                    finally:
                        # Clean up temporary file
                        if 'tmp_path' in locals() and os.path.exists(tmp_path):
                            os.remove(tmp_path)

    # Column 2: Ask Questions (Always available if documents exist)
    with col2:
        st.header("Ask Questions")
        # We use the 'available_docs' list fetched at the start of the 'if google_creds_valid:' block
        if not available_docs:
            st.info("No documents found in the knowledge base. Upload a PDF using the panel on the left to get started.")
        else:
            doc_options = ["All Documents"] + available_docs
            selected_doc_id = st.selectbox(
                "Choose document(s) to query:",
                options=doc_options,
                index=0 # Default to "All Documents"
            )

            question = st.text_input("Tanyakan sesuatu dari dokumen yang dipilih", key="qa_input")

            if question and selected_doc_id:
                retrievers = []
                loaded_dbs = []
                final_retriever = None

                with st.spinner("Searching relevant documents..."):
                    try:
                        if selected_doc_id == "All Documents":
                            # Load all indexes
                            indices_to_load = available_docs
                        else:
                            # Load specific index
                            indices_to_load = [selected_doc_id]

                        for doc_id in indices_to_load:
                             index_path_to_load = FAISS_INDEX_DIR / doc_id
                             if index_path_to_load.exists():
                                 try:
                                     db = FAISS.load_local(
                                         str(index_path_to_load),
                                         embeddings,
                                         allow_dangerous_deserialization=True # Important!
                                     )
                                     retrievers.append(db.as_retriever())
                                     loaded_dbs.append(db) # Optional: keep track if needed later
                                 except Exception as e:
                                     st.warning(f"Could not load index for '{doc_id}': {e}")
                             else:
                                st.warning(f"Index directory for '{doc_id}' not found, skipping.")


                        if not retrievers:
                            st.error("Failed to load any valid document indexes for the selection.")
                        elif len(retrievers) == 1:
                            final_retriever = retrievers[0]
                        else:
                            final_retriever = MergerRetriever(retrievers=retrievers)


                        if final_retriever:
                            relevant_docs = final_retriever.get_relevant_documents(question)

                            if not relevant_docs:
                                st.info("No relevant information found for your question in the selected document(s).")
                            else:
                                # Initialize model and chain inside the 'if' block
                                model = ChatGoogleGenerativeAI(model="gemini-2.5-flash") # Updated model
                                chain = load_qa_chain(model, chain_type="stuff")
                                result = chain.invoke({"input_documents": relevant_docs, "question": question}) # Use invoke

                                st.markdown("### Jawaban:")
                                st.write(result.get('output_text', "No answer generated.")) # Access output_text safely

                                with st.expander("📄 Dokumen yang relevan"):
                                    # Try to determine the source document ID for each relevant doc
                                    doc_sources = {}
                                    if selected_doc_id == "All Documents":
                                        # This is tricky with MergerRetriever without more complex metadata handling
                                        # For now, just show content
                                        pass
                                    else:
                                        doc_sources = {i: selected_doc_id for i in range(len(relevant_docs))}

                                    for i, doc in enumerate(relevant_docs):
                                        source_id = doc_sources.get(i, "Unknown")
                                        # If querying all, source might be in metadata if loader added it
                                        source_info = f"Source: {doc.metadata.get('source', source_id)}"
                                        st.markdown(f"**Doc {i+1} ({source_info})**\n\n{doc.page_content[:500]}...")
                        else:
                            # Handle case where retriever couldn't be created (e.g., loading failed)
                            st.error("Could not initialize document retriever based on your selection.")

                    except Exception as e:
                        st.error(f"An error occurred during question answering: {e}")


# Add instructions for user if credentials are not valid
else:
    st.warning("Google Application Credentials are not valid or the file is missing. Please configure credentials to use the application.")