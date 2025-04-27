import os
import tempfile
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.retrievers import MergerRetriever
from typing import Optional

# --- Configuration ---
# Set credentials (ensure the JSON file is accessible)
# Consider using environment variables for production
CREDENTIALS_PATH = 'hydrogendioxide2104-5b3396eb8d75.json'
if os.path.exists(CREDENTIALS_PATH):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH
else:
    print(f"Warning: Credentials file not found at {CREDENTIALS_PATH}. Google AI features might not work.")
    # Optionally raise an error or handle appropriately
    # raise FileNotFoundError(f"Credentials file not found at {CREDENTIALS_PATH}")

FAISS_INDEX_DIR = Path("faiss_indexes")
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists

# --- Models ---
class AskRequest(BaseModel):
    document_id: Optional[str] = Field(None, description="The specific document ID to search within. If omitted, searches across all documents.")
    question: str

class UploadResponse(BaseModel):
    document_id: str
    filename: str

class AskResponse(BaseModel):
    answer: str
    relevant_docs: list[str] # Return content of relevant docs for context

# --- FastAPI App ---
app = FastAPI(
    title="PDF Question Answering API",
    description="API to upload PDFs and ask questions using LangChain and Google Generative AI.",
    version="0.1.0",
)

# --- Helper Functions (to be added) ---

def _process_pdf_and_create_index(file_path: str, document_id: str):
    """Loads PDF, splits text, creates embeddings, and saves FAISS index."""
    try:
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs_split = text_splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.from_documents(docs_split, embeddings)

        index_path = FAISS_INDEX_DIR / document_id
        db.save_local(str(index_path))
        print(f"FAISS index saved to: {index_path}")

    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        # Clean up index directory if creation failed partially
        if 'index_path' in locals() and index_path.exists():
            shutil.rmtree(index_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

def _cleanup_temp_file(file_path: str):
    """Removes a temporary file."""
    try:
        os.remove(file_path)
        print(f"Removed temporary file: {file_path}")
    except OSError as e:
        print(f"Error removing temporary file {file_path}: {e}")

# --- API Endpoints (to be added) ---
@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accepts PDF upload, processes it, saves FAISS index, returns document ID."""
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF is allowed.")

    document_id = str(uuid.uuid4())
    file_path = None

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            file_path = tmp_file.name
            print(f"Temporary file saved at: {file_path}")

        # Process the PDF and create index (can take time)
        _process_pdf_and_create_index(file_path, document_id)

        # Schedule the temporary file cleanup
        background_tasks.add_task(_cleanup_temp_file, file_path)

        return UploadResponse(document_id=document_id, filename=file.filename)

    except HTTPException as http_exc:
        # If processing failed, ensure temp file is cleaned up immediately
        if file_path and os.path.exists(file_path):
            _cleanup_temp_file(file_path)
        raise http_exc # Re-raise the processing error
    except Exception as e:
        # Catch any other unexpected errors during upload/saving
        if file_path and os.path.exists(file_path):
            _cleanup_temp_file(file_path)
        print(f"Unexpected error during upload: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        await file.close() # Ensure file handle is closed

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Accepts question and optional document ID. Returns answer using specific index or all indexes."""
    question = request.question
    document_id = request.document_id
    retrievers = []
    loaded_dbs = [] # Keep track of loaded dbs to avoid loading the same one multiple times if needed later

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        if document_id:
            # Search within a specific document
            index_path = FAISS_INDEX_DIR / document_id
            if not index_path.exists() or not index_path.is_dir():
                raise HTTPException(status_code=404, detail=f"Document ID '{document_id}' not found or index is invalid.")
            try:
                db = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
                retrievers.append(db.as_retriever())
                loaded_dbs.append(db)
            except Exception as e:
                 raise HTTPException(status_code=500, detail=f"Failed to load index for document ID '{document_id}': {e}")
        else:
            # Search across all documents
            index_files = [p for p in FAISS_INDEX_DIR.iterdir() if p.is_dir()]
            if not index_files:
                 raise HTTPException(status_code=404, detail="No document indexes found to search across.")

            for index_path in index_files:
                try:
                    db = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
                    retrievers.append(db.as_retriever())
                    loaded_dbs.append(db)
                    print(f"Loaded index: {index_path.name}")
                except Exception as e:
                    # Log error but continue loading others if possible
                    print(f"Warning: Failed to load index {index_path.name}: {e}. Skipping this index.")

            if not retrievers:
                 raise HTTPException(status_code=500, detail="Failed to load any valid document indexes.")

        # Combine retrievers if multiple were loaded
        if len(retrievers) == 0:
             # This case should theoretically be caught earlier, but as a safeguard:
             raise HTTPException(status_code=404, detail="No valid retriever could be initialized.")
        elif len(retrievers) == 1:
            final_retriever = retrievers[0]
        else:
            final_retriever = MergerRetriever(retrievers=retrievers)
            print(f"Created MergerRetriever with {len(retrievers)} individual retrievers.")

        # Get relevant documents using the final retriever
        relevant_docs = final_retriever.get_relevant_documents(question)

        if not relevant_docs:
            return AskResponse(answer="No relevant information found in the specified document(s) for your question.", relevant_docs=[])

        # Ensure model is initialized
        if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ or not os.path.exists(os.environ['GOOGLE_APPLICATION_CREDENTIALS']):
             raise HTTPException(status_code=500, detail="Google credentials not configured or file not found.")

        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        chain = load_qa_chain(model, chain_type="stuff")
        result = chain.invoke({"input_documents": relevant_docs, "question": question})

        relevant_docs_content = [doc.page_content for doc in relevant_docs]

        return AskResponse(answer=result['output_text'], relevant_docs=relevant_docs_content)

    except ModuleNotFoundError:
         print("Error: google-generativeai package not found. Please install it.")
         raise HTTPException(status_code=500, detail="Server configuration error: Missing required package.")
    except HTTPException as http_exc:
        # Re-raise known HTTP exceptions
        raise http_exc
    except Exception as e:
        print(f"Error during question answering: {e}")
        if "API key not valid" in str(e):
             raise HTTPException(status_code=401, detail=f"Google AI Authentication Error: {e}")
        # Catch-all for other unexpected errors
        raise HTTPException(status_code=500, detail=f"Failed to process question: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
