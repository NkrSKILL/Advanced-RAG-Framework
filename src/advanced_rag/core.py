from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class AdvancedRAG:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    
    def query(self, question: str):
        return self.qa_chain.run(question)