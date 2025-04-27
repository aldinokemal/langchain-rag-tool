# PDF Chatbot with Google Gemini

This project is a Streamlit web application that allows you to chat with your PDF documents using Google's Generative AI (Gemini Flash 1.5).

## Features

* Upload PDF files.
* Ask questions about the content of the uploaded PDF.
* View relevant snippets from the document that support the answer.

## How it Works

1. **Upload:** The user uploads a PDF file through the Streamlit interface.
2. **Load & Split:** The PDF is loaded using `PDFPlumberLoader`, and the text is split into manageable chunks using `RecursiveCharacterTextSplitter`.
3. **Embed & Store:** The text chunks are converted into embeddings using `GoogleGenerativeAIEmbeddings` (`models/embedding-001`) and stored in a `FAISS` vector store.
4. **Question Answering:** When the user asks a question:
    * The question is embedded.
    * Relevant document chunks are retrieved from the FAISS index based on semantic similarity.
    * The retrieved documents and the question are passed to the `ChatGoogleGenerativeAI` model (`gemini-2.0-flash`) using a Langchain QA chain (`load_qa_chain`) to generate an answer.
5. **Display:** The answer and the relevant document snippets are displayed in the Streamlit app.

## Setup & Usage

1. **Prerequisites:**
    * Python 3.x
    * Pip (Python package installer)
    * Google Cloud credentials JSON file (`hydrogendioxide2104-5b3396eb8d75.json` in this project) with appropriate permissions for Generative AI models.

2. **Installation:**
    * Clone the repository (if applicable).
    * Install dependencies (you might need to create a `requirements.txt` file based on the imports in `app.py`):

        ```bash
        pip install streamlit langchain-community langchain langchain-google-genai pdfplumber faiss-cpu # Or faiss-gpu if you have CUDA
        ```

    * Ensure your Google Cloud credentials file (`hydrogendioxide2104-5b3396eb8d75.json`) is in the project directory.

3. **Running the App:**

    ```bash
    streamlit run app.py
    ```

4. **Access:** Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Key Files

* `app.py`: The main Streamlit application script.
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
* google-cloud-aiplatform (implicitly required by langchain-google-genai)

*(Note: You should create a `requirements.txt` file listing these dependencies)*
