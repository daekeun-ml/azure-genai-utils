from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from abc import ABC, abstractmethod
from operator import itemgetter
from langchain import hub


class RetrievalChain(ABC):
    """
    Base class for retrieval chains.

    :param source_uri: Source URI
    :param k: Number of results to retrieve
    :param embedding_name: Name of the embedding
    :param model_name: Name of the model
    """

    def __init__(self, **kwargs):
        self.source_uri = kwargs.get("source_uri", None)
        self.k = kwargs.get("k", 10)
        self.embedding_name = kwargs.get("embedding_name", "text-embedding-3-large")
        self.model_name = kwargs.get("model_name", "gpt-4o-mini")

    @abstractmethod
    def load_documents(self, source_uris):
        """Load documents from source URIs"""
        pass

    @abstractmethod
    def create_text_splitter(self):
        """Create a text splitter"""
        pass

    def split_documents(self, docs, text_splitter):
        """Split documents using a text splitter"""
        return text_splitter.split_documents(docs)

    def create_embedding(self):
        """Create an embedding"""
        return AzureOpenAIEmbeddings(model=self.embedding_name)

    def create_vectorstore(self, split_docs):
        """Create a vectorstore"""
        return FAISS.from_documents(
            documents=split_docs, embedding=self.create_embedding()
        )

    def create_retriever(self, vectorstore):
        """Create a retriever"""
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )
        return dense_retriever

    def create_model(self):
        """Create a model"""
        return AzureChatOpenAI(model_name=self.model_name, temperature=0)

    def create_prompt(self):
        """Create RAG prompt"""
        return hub.pull("daekeun-ml/rag-history-baseline")

    @staticmethod
    def format_docs(docs):
        return "\n".join(docs)

    def create_chain(self):
        """Create a retrieval chain"""
        docs = self.load_documents(self.source_uri)
        text_splitter = self.create_text_splitter()
        split_docs = self.split_documents(docs, text_splitter)
        self.vectorstore = self.create_vectorstore(split_docs)
        self.retriever = self.create_retriever(self.vectorstore)
        model = self.create_model()
        prompt = self.create_prompt()
        self.chain = (
            {
                "chat_history": itemgetter("chat_history"),
                "question": itemgetter("question"),
                "context": itemgetter("context"),
            }
            | prompt
            | model
            | StrOutputParser()
        )
        return self
