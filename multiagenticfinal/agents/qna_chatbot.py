# agents/qna_chatbot.py

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional
import os # Added os import

from dotenv import load_dotenv # Added for .env loading
load_dotenv() # Load environment variables from .env file

# ---vvv--- Add parent directory to sys.path to find config.py ---vvv---
sys.path.append(str(Path(__file__).parent.parent))
# ---^^^-----------------------------------------------------------^^^---

# --- Third-party Imports ---
# Core LangChain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# ---vvv--- Ollama Integration ---vvv---
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
# ---^^^---------------------------^^^---

# Document Handling
from langchain_community.document_loaders import TextLoader # Keep for flexibility, though PyMuPDF is primary for PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---vvv--- Add PyMuPDFLoader import ---vvv---
from langchain_community.document_loaders import PyMuPDFLoader
# ---^^^---------------------------------^^^---

# Vector Store & Retriever
from langchain_community.vectorstores import FAISS

# --- Local Imports ---
import config # Import shared configuration

# --- Logger Setup ---
log_file = config.LOG_DIR / "qna_chatbot.log"
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class QnAChatbotAgent:
    """
    Provides a Q&A interface using RAG with local Ollama models.
    """

    def __init__(self,
                 qa_model_name: str = config.LLM_MODEL_QA,
                 embedding_model_name: str = config.LLM_MODEL_EMBEDDING,
                 chunk_size: int = config.DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
                 retriever_k: int = config.DEFAULT_RETRIEVER_K):
        """Initializes the QnA Chatbot Agent using Ollama."""
        self.logger = logging.getLogger(__name__)
        
        self.qa_model_name = qa_model_name
        self.embedding_model_name = embedding_model_name

        self.logger.info(f"Initializing QnAChatbotAgent with Ollama (QA: {self.qa_model_name}, Embed: {self.embedding_model_name})")
        self.logger.info(f"Ollama Base URL: {config.OLLAMA_BASE_URL}")
        self.logger.info(f"RAG Params: Chunk Size={chunk_size}, Overlap={chunk_overlap}, Retriever K={retriever_k}")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever_k = retriever_k

        self.vector_store: Optional[FAISS] = None
        self.retriever = None
        self.rag_chain = None
        self.raw_text_path: Optional[Path] = None

        try:
            self.llm = ChatOllama(
                model=self.qa_model_name,
                base_url=config.OLLAMA_BASE_URL,
                temperature=config.DEFAULT_TEMPERATURE,
                num_ctx=config.OLLAMA_NUM_CTX, # Context window size for Ollama
                # Other common Ollama parameters can be added here if needed:
                # top_k=config.OLLAMA_TOP_K, 
                # top_p=config.OLLAMA_TOP_P,
                # repeat_penalty=config.OLLAMA_REPEAT_PENALTY,
            )
            self.logger.info(f"QA LLM ({self.qa_model_name}) initialized for Ollama.")

            self.embeddings = OllamaEmbeddings(
                model=self.embedding_model_name,
                base_url=config.OLLAMA_BASE_URL
            )
            self.logger.info(f"Embeddings ({self.embedding_model_name}) initialized for Ollama.")

        except Exception as e:
            self.logger.error(f"Failed to initialize models for Ollama: {e}", exc_info=True)
            raise ValueError(f"Could not initialize models for Ollama.") from e

        self._setup_prompt()

    def _setup_prompt(self):
        """Sets up the prompt template - simplified and stricter."""
        template = """You are a Question Answering assistant.
Answer the following 'Question' ONLY based on the 'Context' provided below.
Be concise.
If the answer is NOT found in the 'Context', respond with EXACTLY the following phrase: 'I don't know based on the provided context.'
Do NOT add any information not present in the 'Context'. Do NOT apologize or explain why you cannot answer.

Context:
{context}

Question: {question}

Answer:"""
        self.prompt = PromptTemplate.from_template(template)
        self.logger.debug("RAG prompt template set up.")

    def setup_rag_pipeline(self, raw_text_filepath: Path) -> bool:
        """Loads text, creates embeddings, vector store, and RAG chain."""
        self.logger.info(f"Setting up RAG pipeline for: {raw_text_filepath}")
        if not raw_text_filepath.is_file():
            self.logger.error(f"Input file not found: {raw_text_filepath}")
            return False
        self.raw_text_path = raw_text_filepath # Keep track of the original PDF path

        try:
            # ---vvv--- Modified to use PyMuPDFLoader for PDF input ---vvv---
            self.logger.debug(f"Loading PDF document: {raw_text_filepath}")
            loader = PyMuPDFLoader(str(raw_text_filepath))
            # ---^^^----------------------------------------------------^^^---
            docs = loader.load()
            if not docs:
                 self.logger.error(f"PyMuPDFLoader failed to load content from {raw_text_filepath}.")
                 return False
            full_text_length = sum(len(doc.page_content) for doc in docs)
            self.logger.debug(f"Loaded {len(docs)} parts, total length: {full_text_length} chars.")

            self.logger.debug(f"Splitting document (size={self.chunk_size}, overlap={self.chunk_overlap})...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            splits = text_splitter.split_documents(docs)
            if not splits:
                 self.logger.error(f"Text splitting resulted in zero chunks.")
                 return False
            self.logger.info(f"Document split into {len(splits)} chunks.")
            if splits: self.logger.debug(f"Sample chunk 0 length: {len(splits[0].page_content)} chars")

            self.logger.debug(f"Creating FAISS vector store using '{self.embedding_model_name}' embeddings...")
            start_embed_time = time.time()
            try:
                self.vector_store = FAISS.from_documents(splits, self.embeddings)
            except Exception as faiss_e:
                self.logger.error(f"FAISS vector store creation failed: {faiss_e}", exc_info=True)
                return False
            embed_time = time.time() - start_embed_time
            self.logger.info(f"Vector store created successfully in {embed_time:.2f} seconds.")

            self.retriever = self.vector_store.as_retriever(
                 search_type="similarity",
                 search_kwargs={'k': self.retriever_k}
            )
            self.logger.debug(f"Retriever created (k={self.retriever_k}).")

            self.rag_chain = (
                RunnableParallel(
                    {"context": self.retriever, "question": RunnablePassthrough()}
                )
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            self.logger.info("RAG chain assembled successfully.")
            return True

        except Exception as e:
            self.logger.error(f"Error during RAG pipeline setup for {raw_text_filepath}: {e}", exc_info=True)
            self.vector_store = None; self.retriever = None; self.rag_chain = None; self.raw_text_path = None
            return False

    def ask_question(self, question: str):
        """Asks a question to the RAG pipeline and streams the answer."""
        if not self.rag_chain:
            self.logger.error("RAG pipeline not set up. Cannot ask question.")
            yield "Error: The Q&A system is not ready."
            return
        if not question or not question.strip():
            self.logger.warning("Question is empty.")
            yield "Please provide a question."
            return

        self.logger.info(f"Processing question (streaming): '{question}'")
        start_time = time.time()
        try:
            # ---vvv--- Use .stream() for streaming responses ---vvv---
            full_response = ""
            for chunk in self.rag_chain.stream(question):
                yield chunk
                full_response += chunk # Accumulate the full response for logging if needed
            # ---^^^--------------------------------------------^^^---
            end_time = time.time()
            self.logger.info(f"Streaming finished in {end_time - start_time:.2f} seconds.")
            self.logger.debug(f"Full streamed answer: {full_response.strip()}")

        except Exception as e:
            if "llama runner process has terminated" in str(e):
                 self.logger.error(f"Ollama runner crashed during Q&A stream: {e}", exc_info=True)
            else:
                self.logger.error(f"Error streaming RAG chain for question '{question}': {e}", exc_info=True)
            if "context window" in str(e).lower():
                 self.logger.error(f"Potential context window issue during QA stream (Model: {self.qa_model_name}). Check retriever K/chunk size.")
            yield "Sorry, an error occurred while processing your question."

    def start_chat(self):
        """Starts an interactive command-line chat session with streaming output."""
        if not self.rag_chain or not self.raw_text_path:
            self.logger.error("Chatbot cannot start. RAG pipeline must be set up.")
            print("\nERROR: Chatbot not ready. Setup failed.")
            return

        print("\n" + "="*60)
        print(f" Q&A Chatbot Ready - Document: {self.raw_text_path.name}")
        print(f" Using QA model: {self.qa_model_name}")
        print(" Type your question and press Enter.")
        print(" Type 'quit', 'exit', or press Ctrl+C/Ctrl+D to end.")
        print("="*60)

        while True:
            try:
                question = input("\nQuestion: ").strip()
                if not question: continue
                if question.lower() in ["quit", "exit"]:
                    print("Exiting chatbot. Goodbye!")
                    break
                
                # ---vvv--- Handle streaming output ---vvv---
                print("\nAnswer: ", end="", flush=True)
                full_answer_received = False
                try:
                    for token in self.ask_question(question):
                        print(token, end="", flush=True)
                        full_answer_received = True
                    if not full_answer_received:
                        print("[No answer received or error occurred]")
                except Exception as stream_e:
                    self.logger.error(f"Error receiving stream in chat: {stream_e}", exc_info=True)
                    print("\n[Error during streaming]")
                print() # Newline after the full answer
                # ---^^^-----------------------------^^^---
            except (EOFError, KeyboardInterrupt):
                 print("\nExiting chatbot. Goodbye!")
                 break
            except Exception as e:
                 self.logger.error(f"Unexpected error during chat loop: {e}", exc_info=True)
                 print("\nAn unexpected error occurred.")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QnA Chatbot Agent using Ollama RAG.")
    parser.add_argument("pdf_filepath", type=str, help="Filename of the PDF paper (e.g., 'attention.pdf').")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (DEBUG level).")
    parser.add_argument("--chunk_size", type=int, default=None, help=f"Override chunk size (default: {config.DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk_overlap", type=int, default=None, help=f"Override chunk overlap (default: {config.DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("--retriever_k", type=int, default=None, help=f"Override retriever K (default: {config.DEFAULT_RETRIEVER_K})")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler): handler.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled (DEBUG level).")
    else: logger.setLevel(logging.INFO)

    chunk_size = args.chunk_size if args.chunk_size is not None else config.DEFAULT_CHUNK_SIZE
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else config.DEFAULT_CHUNK_OVERLAP
    retriever_k = args.retriever_k if args.retriever_k is not None else config.DEFAULT_RETRIEVER_K

    # ---vvv--- Revised PDF Path Resolution Logic ---vvv---
    # Attempt to resolve the provided path directly first
    input_pdf_path = Path(args.pdf_filepath)

    if not input_pdf_path.is_file():
        # If direct path fails, try resolving it against the RESEARCH_PAPERS_DIR
        # This handles cases where just a filename or a path relative to RESEARCH_PAPERS_DIR might be given
        # For example, if args.pdf_filepath is "p1111.pdf" or "subdir/p1111.pdf"
        # and RESEARCH_PAPERS_DIR is "multiagenticfinal/research_papers"
        potential_path_in_research = config.RESEARCH_PAPERS_DIR / args.pdf_filepath
        if potential_path_in_research.is_file():
            input_pdf_path = potential_path_in_research
        else:
            # As a final fallback, if args.pdf_filepath might be just a name
            # and it's directly inside RESEARCH_PAPERS_DIR.
            # This also covers if args.pdf_filepath was like "research_papers/p1111.pdf"
            # and CWD is "multiagenticfinal", then Path(args.pdf_filepath) would have worked.
            # This fallback is more for just filename "p1111.pdf"
            path_in_research_using_name = config.RESEARCH_PAPERS_DIR / Path(args.pdf_filepath).name
            if path_in_research_using_name.is_file():
                 input_pdf_path = path_in_research_using_name
            else:
                # Log all checked paths for better debugging
                checked_paths_log = (
                    f"  1. Direct as given: {Path(args.pdf_filepath).resolve()}\\n"
                    f"  2. Relative to RESEARCH_PAPERS_DIR ({config.RESEARCH_PAPERS_DIR}): {potential_path_in_research.resolve()}\\n"
                    f"  3. Filename in RESEARCH_PAPERS_DIR: {path_in_research_using_name.resolve()}"
                )
                logger.critical(
                    f"CRITICAL ERROR: PDF file not found.\\n"
                    f"Input given: '{args.pdf_filepath}'\\n"
                    f"Checked paths:\\n{checked_paths_log}"
                )
                sys.exit(1)
    # ---^^^-------------------------------------------------^^^---

    if not input_pdf_path.is_file(): # This check is somewhat redundant now but good for sanity
        # Try to see if it\'s an absolute path
        input_pdf_path_abs = Path(args.pdf_filepath)
        if input_pdf_path_abs.is_file():
            input_pdf_path = input_pdf_path_abs
        else:
            # ---vvv--- Check in config.RESEARCH_PAPERS_DIR ---vvv---
            # This directory is defined in config.py and points to multiagenticfinal/research_papers/
            input_pdf_path_config_research = config.RESEARCH_PAPERS_DIR / args.pdf_filepath
            if input_pdf_path_config_research.is_file():
                input_pdf_path = input_pdf_path_config_research
            else:
                logger.critical(f"CRITICAL ERROR: PDF file not found at {config.RAW_TEXT_DIR / args.pdf_filepath}, as an absolute path {args.pdf_filepath}, or in {config.RESEARCH_PAPERS_DIR / args.pdf_filepath}")
                sys.exit(1)
            # ---^^^---------------------------------------------^^^---

    try:
        qna_agent = QnAChatbotAgent(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retriever_k=retriever_k
        )
        logger.info(f"Attempting to set up RAG pipeline with {input_pdf_path}...")
        setup_success = qna_agent.setup_rag_pipeline(input_pdf_path)

        if setup_success:
            qna_agent.start_chat()
        else:
            logger.error("Failed to set up RAG pipeline. Cannot start chat.")
            print("\nERROR: Could not initialize the RAG pipeline. Check logs.")
            sys.exit(1)

    except ValueError as ve:
        logger.critical(f"Initialization Error: {ve}")
        print(f"\nCRITICAL ERROR during agent initialization: {ve}")
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        print("\nAn unexpected critical error occurred. Check logs.")

    logger.info("--- Q&A Chatbot Session Finished ---")
    