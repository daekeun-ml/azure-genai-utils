from azure_genai_utils.rag.base import RetrievalChain
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    PyPDFLoader,
    PyMuPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Annotated, Optional


class PDFRetrievalChain(RetrievalChain):
    def __init__(
        self,
        source_uri: Annotated[List[str], "Source URI"],
        k: Annotated[int, "Number of results to retrieve"] = 10,
        loader_type: Annotated[str, "PDF Loader Type"] = "PDFPlumber",
        chunk_size: Annotated[int, "Chunk Size"] = 300,
        chunk_overlap: Annotated[int, "Chunk Overlap"] = 50,
        **kwargs,  # Additional keyword arguments for parent class
    ):
        """
        Initialize the retrieval chain with the source URI and loader type.

        :param source_uri: URI of the PDF source
        :param loader_type: Type of PDF loader to use. Options are "PDFPlumber", "PyPDF", or "PyMuPDF".
        """
        super().__init__(**kwargs)
        self.source_uri = source_uri  # Override if needed
        self.k = k  # Override if needed
        self.loader_type = loader_type  # Specific to PDFRetrievalChain
        self.loader_map = {
            "PDFPlumber": PDFPlumberLoader,
            "PyPDF": PyPDFLoader,
            "PyMuPDF": PyMuPDFLoader,
        }
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def get_loader(self, loader_type: Optional[str] = None):
        """
        Returns the loader class based on the loader type.

        :param loader_type: Optional; the type of loader to use.
        :return: The corresponding loader class.
        """
        loader_type = loader_type or self.loader_type
        if loader_type not in self.loader_map:
            raise ValueError(
                f"Unsupported loader type: {loader_type}. "
                f"Supported types are: {list(self.loader_map.keys())}"
            )
        return self.loader_map[loader_type]

    def load_documents(self, source_uris: List[str]):
        """
        Load documents using the selected loader type.

        :param source_uris: List of source URIs
        :return: List of loaded documents
        """
        docs = []
        loader_class = self.get_loader()

        for source_uri in source_uris:
            loader = loader_class(source_uri)
            docs.extend(loader.load())

        return docs

    def create_text_splitter(self):
        """
        Create a text splitter for processing the documents.

        :return: RecursiveCharacterTextSplitter instance
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
