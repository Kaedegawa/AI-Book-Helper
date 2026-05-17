from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from ebooklib import epub
from bs4 import BeautifulSoup

#-----------------

def epub_to_text(epub_path):
    book = epub.read_epub(epub_path)
    
    all_text = []
    
    for item in book.get_items():
        if item.get_type() == 9:
            html_content = item.get_content()
            
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text(separator="\n")
            
            lines = text.splitlines()
            clean_lines = []
            
            for line in lines:
                line = line.strip()
                
                if line:
                    clean_lines.append(line)
            clean_text = "\n".join(clean_lines)
            all_text.append(clean_text)
            
    final_text = "\n\n".join(all_text)
    return final_text

#-----------------

def chunker(text, chunk_size):
    chunks = []
    words = text.split()

    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks

#-----------------
@st.cache_resource

def build_vector_store(chunks_tuple):
    docs = []
    ids = []
    
    chunks = list(chunks_tuple)
    
    for i, chunk in enumerate(chunks):
        docs.append(Document(page_content=chunk, metadata={"chunk": i}))
        ids.append(f"chunk_{i}")
        
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        ids=ids,
        collection_name = "book_chunk"
    )
    
    return vector_store
        

#-----------------

def ask_ai_langchain(question, chunks):

    vector_store = build_vector_store(tuple(chunks))

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    retrieved_docs = retriever.invoke(question)

    context = ""

    for doc in retrieved_docs:
        context += doc.page_content + "\n\n---\n\n"

    prompt = ChatPromptTemplate.from_template("""
Use the book excerpts below to answer the question.
Only answer using the excerpts.
If the answer is not clear, say the excerpts do not provide enough information.

Book Excerpts:
{context}

Question:
{question}
""")

    model = ChatOllama(model="llama3.2")

    chain = prompt | model

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content, retrieved_docs
#-----------------

st.title("AI Book Analyzer")

st.write("Ask questions about your book using local AI.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    


uploaded_file = st.file_uploader("Upload an EPUB file", type = ["epub"])

    
if uploaded_file is not None:
    with open ("uploaded_book.epub", "wb") as f:
        f.write(uploaded_file.getbuffer())
    text = epub_to_text("uploaded_book.epub")
    chunks = chunker(text, chunk_size=500)
    question = st.text_input("Ask a question about the book:")
    
    if st.button("Ask AI"):
        with st.spinner("Thinking..."):
            answer, retrieved_docs = ask_ai_langchain(question, chunks)
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "docs": retrieved_docs
            })
    for chat in st.session_state.chat_history:
        st.subheader("Question")
        st.write(chat["question"])
        
        st.subheader("Answer")
        st.write(chat["answer"])
        with st.expander("Retrieved Chunks"):
            for doc in chat["docs"]:
                st.write(doc.page_content)
                st.divider()
                
        





