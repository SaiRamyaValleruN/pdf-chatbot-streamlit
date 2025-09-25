import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# Set page config
st.set_page_config(page_title="Free PDF Q&A Chatbot", layout="wide")
st.title("📄 Free PDF Q&A Chatbot ")

# Sidebar upload
with st.sidebar:
    st.header("📎 Upload your PDF")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")

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
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_text(text)

        if len(chunks) == 0:
            st.warning("⚠️ No valid text chunks to embed.")
        else:
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            try:
                vector_store = FAISS.from_texts(chunks, embeddings)

                # ✅ Use MMR retriever for better coverage
                retriever = vector_store.as_retriever(
                    search_type="mmr", search_kwargs={"k": 20}
                )

                st.success("✅ PDF processed successfully. You can now ask questions!")

                # ✅ Extractive QA model
                qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

                def answer_question(query):
                    # Retrieve candidate chunks
                    docs = retriever.get_relevant_documents(query)

                    if not docs:
                        return "❌ No relevant context found in the PDF.", None

                    best_answer = None
                    best_score = -1
                    best_doc = None

                    for doc in docs:
                        result = qa_pipeline(question=query, context=doc.page_content)

                        # ✅ Always keep the best scoring answer
                        if result["score"] > best_score:
                            best_answer = result["answer"]
                            best_score = result["score"]
                            best_doc = doc.page_content

                    if not best_answer or best_answer.strip() == "":
                        # Fallback: show nearest context
                        return "⚠️ No clear answer, but here’s something related:", docs[0].page_content[:400]

                    # ✅ Return even low-confidence answers
                    return f"{best_answer} (nearest match, confidence: {round(best_score, 2)})", best_doc

                # Ask questions
                query = st.text_input("❓ Ask a question about the PDF:")
                if query:
                    answer, context = answer_question(query)
                    st.subheader("🤖 Answer:")
                    st.success(answer)

                    # ✅ Show context if available
                    if context:
                        with st.expander("📄 Show source context from PDF"):
                            st.write(context)

            except Exception as e:
                st.error(f"Something went wrong while embedding: {e}")
