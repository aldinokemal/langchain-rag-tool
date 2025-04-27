# PDF Chatbot with Google Gemini

This project is a Streamlit web application that allows you to chat with your PDF documents using Google's Generative AI (Gemini Flash 1.5). It also includes a FastAPI backend for interacting with the PDF processing and Q&A functionality programmatically.

## Features

* Upload PDF files via Streamlit UI or API endpoint.
* Ask questions about the content of the uploaded PDF(s) via Streamlit UI or API endpoint.
* View relevant snippets from the document that support the answer (Streamlit UI).
* Retrieve answers and relevant document content via API.
* Persists document indexes in the `faiss_indexes/` directory.

## How it Works

1. **Upload:** The user uploads a PDF file through the Streamlit interface (`app.py`) or the `/upload` API endpoint (`api.py`).
2. **Load & Split:** The PDF is loaded using `PDFPlumberLoader`, and the text is split into manageable chunks using `RecursiveCharacterTextSplitter`.
3. **Embed & Store:** The text chunks are converted into embeddings using `GoogleGenerativeAIEmbeddings` (`models/embedding-001`) and stored in a `FAISS` vector store. The index is saved locally in the `faiss_indexes/` directory, identified by a unique ID (`api.py`) or processed in memory (`app.py`).
4. **Question Answering:** When the user asks a question (via UI or `/ask` API):
    * The question is embedded.
    * Relevant document chunks are retrieved from the appropriate FAISS index (or a combination of indexes in the API) based on semantic similarity.
    * The retrieved documents and the question are passed to the `ChatGoogleGenerativeAI` model (`gemini-2.0-flash`) using a Langchain QA chain (`load_qa_chain`) to generate an answer.
5. **Display/Return:** The answer and relevant document snippets are displayed in the Streamlit app, or the answer and relevant content are returned by the API.

## Setup & Usage

1. **Prerequisites:**
    * Python 3.x
    * Pip (Python package installer)
    * Google Cloud credentials JSON file (`hydrogendioxide2104-5b3396eb8d75.json` in this project) with appropriate permissions for Generative AI models. **Note:** For production environments, consider using environment variables or a more secure secrets management solution instead of hardcoding the file path.

2. **Installation:**
    * Clone the repository (if applicable).
    * Install dependencies from `requirements.txt`:

        ```bash
        pip install -r requirements.txt
        ```

    * Ensure your Google Cloud credentials file (`hydrogendioxide2104-5b3396eb8d75.json`) is in the project directory or the `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set correctly.

3. **Running the Streamlit App:**

    ```bash
    streamlit run app.py
    ```

4. **Running the FastAPI:**

    ```bash
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
    ```

    *(You can access the API docs at `http://localhost:8000/docs`)*

5. **Access Streamlit:** Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Key Files

* `app.py`: The main Streamlit application script.
* `api.py`: The FastAPI backend script providing API endpoints for upload and Q&A.
* `requirements.txt`: Lists all the required Python packages.
* `faiss_indexes/`: Directory where persistent FAISS vector store indexes are saved by `api.py`.
* `hydrogendioxide2104-5b3396eb8d75.json`: Google Cloud credentials file (handle with care!).
* `chatbot_cs.ipynb`: Jupyter notebook potentially containing development or experimentation code (optional).
* `panduan_*.pdf`: Example PDF files included in the repository.

## Dependencies

* streamlit
* langchain-community
* langchain
* langchain-google-genai
* pdfplumber
* faiss-cpu / faiss-gpu
* fastapi
* uvicorn[standard]
* pydantic
* google-cloud-aiplatform
