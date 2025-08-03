import os
import sys
from utils.model_loader import ModelLoader
from logger.custom_logging import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt.prompt_library import PROPMT_REGISTRY
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentAnalyzer:
    """
    Analyzes documents using a pre-trained model.
    Automatically logs all actions and supports session-based organization.
    """
    def __init__(self):
        self.log = CustomLogger().get_logger(__name__)
        try:
            self.loader=ModelLoader()
            self.llm=self.loader.load_llm()
            
            # Prepare parsers
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            
            self.prompt = PROPMT_REGISTRY["document_analysis"]
            
            self.log.info("DocumentAnalyzer initialized successfully")
            
            
        except Exception as e:
            self.log.error(f"Error initializing DocumentAnalyzer: {e}")
            raise DocumentPortalException("Error in DocumentAnalyzer initialization", e)
        
        
    
    def analyze_document(self, document_text: str) -> dict:
        """
        Analyze a document's text and extract structured metadata & summary.
        Automatically chunks input to avoid token limits.
        """
        def split_text(text, chunk_size=3000, chunk_overlap=200):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            return splitter.split_text(text)

        try:
            chain = self.prompt | self.llm | self.fixing_parser
            self.log.info("Meta-data analysis chain initialized")

            chunks = split_text(document_text)
            aggregated_result = {}

            for idx, chunk in enumerate(chunks):
                self.log.info(f"Processing chunk {idx + 1}/{len(chunks)}")
                try:
                    response = chain.invoke({
                        "format_instructions": self.parser.get_format_instructions(),
                        "document_text": chunk
                    })
                    aggregated_result.update(response)
                except Exception as chunk_error:
                    self.log.error("LLM invocation failed on chunk", error=str(chunk_error))
                    continue

            self.log.info("Metadata extraction successful", keys=list(aggregated_result.keys()))
            return aggregated_result

        except Exception as e:
            self.log.error("Metadata analysis failed", error=str(e))
            raise DocumentPortalException("Metadata extraction failed", e)
        
    