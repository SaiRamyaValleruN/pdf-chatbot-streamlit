import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Set page config at the top!
st.set_page_config(page_title="Free PDF Q&A Chatbot", layout="wide")
st.title("📄 Free PDF Q&A Chatbot ")

# PDF Upload
with st.sidebar:
    st.header("📎 Upload your PDF")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")

# Handle file
if uploaded_pdf:
    pdf_reader = PdfReader(uploaded_pdf)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if not text.strip():
        st.warning("⚠️ Could not extract any text from the PDF. Try another one.")
    else:
        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_text(text)

        if len(chunks) == 0:
            st.warning("⚠️ No valid text chunks to embed.")
        else:
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            try:
                vector_store = FAISS.from_texts(chunks, embeddings)
                st.success("✅ PDF processed successfully. You can now ask questions!")
                
                query = st.text_input("❓ Ask a question about the PDF:")
                if query:
                    docs = vector_store.similarity_search(query)
                    if docs:
                        st.subheader("🤖 Answer:")
                        st.success(docs[0].page_content)
                    else:
                        st.error("😢 No relevant answer found in the document.")
            except Exception as e:
                st.error(f"Something went wrong while embedding: {e}")

