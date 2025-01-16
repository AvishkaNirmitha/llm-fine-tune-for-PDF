import os
import torch
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from functools import lru_cache
from chromadb.config import Settings

# Set environment variable to bypass the OpenMP runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    try:
        # Check if GPU is available
        is_gpu_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if is_gpu_available else "CPU"
        print(f"Using device: {'GPU: ' + device_name if is_gpu_available else 'CPU'}")

        # 1. PDF Loading
        print("Starting PDF loading...")
        pdf_paths = ["C:\\Users\\menuk\\Desktop\\BOT-4\\Karbon User Guide.pdf"]
        data = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            data.extend(loader.load())
        print(f"Total number of pages loaded: {len(data)}")

        # 2. Document Splitting
        print("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs = text_splitter.split_documents(data)
        print(f"Total number of chunks: {len(docs)}")

        # 3. Initialize Embeddings
        print("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        print("Embeddings initialized successfully")

        # 4. Create or Load Vector Store
        print("Initializing vector store...")
        vectorstore_dir = "vectorstore_dir"
        if os.path.exists(vectorstore_dir):
            print("Loading existing vector store...")
            vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)
        else:
            print("Creating new vector store...")
            vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=vectorstore_dir)
            print("Saving vector store...")
            vectorstore.persist()
        print("Vector store is ready")

        # 5. Create Retriever
        print("Creating retriever...")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        print("Retriever created successfully")

        # 6. Initialize Llama 3 through Ollama
        print("Initializing Llama 3 model...")
        llm = OllamaLLM(
            model="llama3.1:8b",  # Use a smaller model for faster inference
            temperature=0.3,      # Lower temperature for faster response
        )
        print("Llama 3 model initialized successfully")

        # 7. Create Chain
        print("Creating chain...")
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant. Use the following context to answer the question concisely.

Context: {context}

Question: {question}

Answer:"""
        )
        
        llm_chain = prompt | llm | StrOutputParser()
        rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain
        print("Chain created successfully")

        # 8. Run Query with Caching
        @lru_cache(maxsize=50)
        def cached_response(question):
            return rag_chain.invoke(question)

        print("\nBot is ready! Ask your questions or type 'quit' to exit.")
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
            
            print("\nProcessing...")
            start_time = time.time()  # Record the start time
            response = cached_response(question)
            elapsed_time = time.time() - start_time
            print(f"\nResponse: {response}")
            print(f"Time taken: {elapsed_time:.2f} seconds")

        print("\nProcess completed successfully!")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if _name_ == "__main__":
    main()
