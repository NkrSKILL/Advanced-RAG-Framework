from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from src.advanced_rag.core import AdvancedRAG

# Example usage
loader = TextLoader('sample.txt')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(splits, embeddings)

retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model='gpt-4o-mini')

rag = AdvancedRAG(llm, retriever)
print(rag.query('Your question here'))