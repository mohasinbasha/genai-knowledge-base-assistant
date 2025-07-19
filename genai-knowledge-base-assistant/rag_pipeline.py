from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load PDF and split
loader = PyPDFLoader("data/airline_docs.pdf")
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(docs)

# Embed & store in vector DB
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Set retriever
retriever = vectorstore.as_retriever()

# QA Chain
llm = OpenAI(temperature=0.1)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Example query
query = "What does the airline refund policy say about delays?"
response = qa_chain.run(query)
print("\nAnswer:", response)