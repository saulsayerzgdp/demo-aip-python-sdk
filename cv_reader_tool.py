"""PDF reader tool for reading CV/resume files."""

from typing import Type

from gllm_docproc.loader.pdf import PDFMinerLoader, PDFPlumberLoader
from gllm_docproc.loader.pipeline_loader import PipelineLoader
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class DocumentReaderInput(BaseModel):
    """Input schema for the DocumentReader tool."""

    file_path: str = Field(..., description="Path to the document file to be read")


class PDFReaderTool(BaseTool):
    """Tool to read and extract text from PDF files."""

    name: str = "pdf_reader_tool"
    description: str = "Read a PDF file and extract its text content. Input should be the path to the PDF file."
    args_schema: Type[BaseModel] = DocumentReaderInput
    loader: PipelineLoader = Field(default_factory=PipelineLoader)

    def __init__(self):
        """Initialize the PDF reader tool."""
        super().__init__()
        self._setup_loader()

    def _setup_loader(self):
        """Set up the PDF loaders."""
        self.loader.add_loader(PDFMinerLoader())
        self.loader.add_loader(PDFPlumberLoader())

    def _run(self, file_path: str) -> str:
        try:
            loaded_elements = self.loader.load(file_path)
            full_text = "\n".join(element["text"] for element in loaded_elements)
            return full_text.strip()
        except Exception as e:
            return f"Error reading file: {str(e)}"


pdf_reader_tool = PDFReaderTool()
