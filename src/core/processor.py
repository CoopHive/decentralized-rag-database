"""
Document processing module.

This module provides a Processor class for handling the end-to-end processing
of scientific documents, including conversion, chunking, embedding, and storage.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import certifi
import requests

from src.core.chunker import chunk
from src.core.converter import convert
from src.core.embedder import embed, embed_batch
from src.db.graph_db import IPFSNeo4jGraph
from src.utils.logging_utils import get_logger, get_user_logger


class Processor:
    """Base class for text processing."""

    def __init__(
        self,
        authorPublicKey: str,
        user_email: str,
        project_root: Optional[Path] = None,
    ):
        """
        Initialize the processor.

        Args:
            authorPublicKey: Public key of the author
            ipfs_api_key: API key for Lighthouse IPFS
            user_email: Email of the user for creating user-specific folders
            project_root: Path to project root directory
        """
        # Use user-specific logger
        self.logger = get_user_logger(user_email, "processor") if user_email else get_logger(__name__ + ".Processor")
        
        self.authorPublicKey = authorPublicKey  # Author Public Key
        self.ipfs_api_key = os.getenv("LIGHTHOUSE_TOKEN")
        self.convert_cache: Dict[str, str] = {}  # Cache for converted text
        self.chunk_cache: Dict[str, List[str]] = {}  # Cache for chunked text
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.user_email = user_email

        # Create temp directory for temporary files
        self.temp_dir = self.project_root / "temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Create user-specific folder inside temp directory
        # Sanitize email for use as folder name (replace @ and . with _)
        self.user_temp_dir = self.temp_dir / self.user_email
        os.makedirs(self.user_temp_dir, exist_ok=True)
        
        self.logger.info(f"Using user temp directory: {self.user_temp_dir}")

        # Paths for user-specific temporary files
        self.tmp_file_path = self.user_temp_dir / "tmp.txt"

        # Set SSL certificate path explicitly
        os.environ["SSL_CERT_FILE"] = certifi.where()

        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        self.graph_db = IPFSNeo4jGraph(
            uri=neo4j_uri, username=neo4j_username, password=neo4j_password
        )

        self.__write_to_file(self.authorPublicKey, str(self.tmp_file_path))
        self.logger.info(
            f"Uploading author public key to Lighthouse: {self.authorPublicKey[:10]}..."
        )
        self.author_cid = self.__upload_text_to_lighthouse(
            str(self.tmp_file_path)
        ).split("ipfs/")[-1]
        self.logger.info(f"Author CID: {self.author_cid}")
        self.graph_db.add_ipfs_node(self.author_cid)

    def __upload_text_to_lighthouse(self, filename: str) -> str:
        """Uploads a string as a file to Lighthouse IPFS and returns the IPFS hash (CID).

        - content: The string content to be uploaded.
        - filename: The name of the file for the uploaded content.
        - Returns: IPFS hash (CID) of the uploaded file.
        """
        url = "https://node.lighthouse.storage/api/v0/add"

        headers = {"Authorization": f"Bearer {self.ipfs_api_key}"}

        with open(filename, "rb") as file:
            files = {"file": file}
            response = requests.post(url, headers=headers, files=files)

        response.raise_for_status()

        # Explicitly cast to string to satisfy type checker
        hash_value: str = response.json()["Hash"]
        return hash_value

    def __write_to_file(self, content: str, file_path: Union[str, Path]) -> None:
        """Writes the content to a file.

        - content: The content to be written to the file.
        - file_path: The path to the file to write the content to.
        """
        try:
            path_str = str(file_path)
            os.makedirs(os.path.dirname(path_str), exist_ok=True)
            with open(path_str, "w") as file:
                file.write(content)
        except Exception as e:
            self.logger.error(f"Error writing to file {file_path}: {e}")

    def __read_mappings(self, mapping_file_path: Union[str, Path]) -> Dict[str, List[str]]:
        """Read mappings from JSON file.
        
        Args:
            mapping_file_path: Path to the mappings JSON file
            
        Returns:
            Dictionary mapping PDF CIDs to list of database combinations
        """
        try:
            if os.path.exists(mapping_file_path):
                with open(mapping_file_path, "r") as file:
                    content = file.read().strip()
                    if not content:
                        self.logger.debug(f"Mappings file {mapping_file_path} is empty, returning empty dict")
                        return {}
                    return json.loads(content)
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error reading mappings from {mapping_file_path}: {e}")
            # Return empty dict and try to fix the file
            try:
                with open(mapping_file_path, "w") as file:
                    json.dump({}, file)
                self.logger.info(f"Reset corrupted mappings file {mapping_file_path}")
            except Exception as write_error:
                self.logger.error(f"Failed to reset mappings file: {write_error}")
            return {}
        except Exception as e:
            self.logger.error(f"Error reading mappings from {mapping_file_path}: {e}")
            return {}

    def __write_mappings(self, mappings: Dict[str, List[str]], mapping_file_path: Union[str, Path]) -> None:
        """Write mappings to JSON file.
        
        Args:
            mappings: Dictionary mapping PDF CIDs to list of database combinations
            mapping_file_path: Path to the mappings JSON file
        """
        try:
            os.makedirs(os.path.dirname(str(mapping_file_path)), exist_ok=True)
            with open(mapping_file_path, "w") as file:
                json.dump(mappings, file, indent=2)
            self.logger.debug(f"Updated mappings in {mapping_file_path}")
        except Exception as e:
            self.logger.error(f"Error writing mappings to {mapping_file_path}: {e}")

    def __update_mappings(self, pdf_cid: str, db_combination: str) -> None:
        """Update both global and user-specific mappings.
        
        Args:
            pdf_cid: The PDF CID that was processed
            db_combination: The database combination in format "converter_chunker_embedder"
        """
        # Global mappings file
        global_mappings_path = self.temp_dir / "mappings.json"
        global_mappings = self.__read_mappings(global_mappings_path)
        
        if pdf_cid not in global_mappings:
            global_mappings[pdf_cid] = []
        if db_combination not in global_mappings[pdf_cid]:
            global_mappings[pdf_cid].append(db_combination)
        
        self.__write_mappings(global_mappings, global_mappings_path)
        
        # User-specific mappings file
        user_mappings_path = self.user_temp_dir / "mappings.json"
        user_mappings = self.__read_mappings(user_mappings_path)
        
        if pdf_cid not in user_mappings:
            user_mappings[pdf_cid] = []
        if db_combination not in user_mappings[pdf_cid]:
            user_mappings[pdf_cid].append(db_combination)
        
        self.__write_mappings(user_mappings, user_mappings_path)

    def process(self, pdf_path: str, databases: List[dict]) -> None:
        """
        Processes the PDF according to the list of database configurations passed.

        Args:
            pdf_path: Path to the input PDF
            databases: A list of configs, each containing a converter, chunker, and embedder
        """
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        self.logger.info(f"Processing document: {doc_id}")
        self.convert_cache = {}
        self.chunk_cache = {}

        metadata = self.default_metadata()

        metadata = {
            key: (
                json.dumps(value)
                if isinstance(value, (list, dict))
                else value
                if value is not None
                else "N/A"
            )
            for key, value in metadata.items()
        }

        self.logger.info(f"Uploading PDF to IPFS: {pdf_path}")
        metadata["pdf_ipfs_cid"] = self.__upload_text_to_lighthouse(pdf_path)

        if not metadata["pdf_ipfs_cid"]:
            self.logger.error(f"Failed to upload PDF to IPFS: {pdf_path}")
            return

        self.logger.info(f"Adding PDF CID to graph: {metadata['pdf_ipfs_cid']}")
        self.graph_db.add_ipfs_node(metadata["pdf_ipfs_cid"])
        self.graph_db.create_relationship(
            metadata["pdf_ipfs_cid"], self.author_cid, "AUTHORED_BY"
        )
        
        for db_config in databases:
            
            converter_func = db_config["converter"]
            chunker_func = db_config["chunker"]
            embedder_func = db_config["embedder"]
            db_combination = f"{converter_func}_{chunker_func}_{embedder_func}"

            # Check if this PDF + database combination already exists in global mappings
            global_mappings_path = self.temp_dir / "mappings.json"
            global_mappings = self.__read_mappings(global_mappings_path)
            
            self.logger.debug(f"Checking if {db_combination} already exists for PDF CID {metadata['pdf_ipfs_cid']}")
            
            if metadata["pdf_ipfs_cid"] in global_mappings and db_combination in global_mappings[metadata["pdf_ipfs_cid"]]:
                self.logger.info(f"Skipping {db_combination} - already processed for this PDF")
                self.__update_mappings(metadata["pdf_ipfs_cid"], db_combination)
                continue

            self.logger.info(f"Processing new combination: {db_combination}")

            # Step 2.1: Conversion
            # Check if markdown conversion already exists for this PDF CID
            converted_text_ipfs_cid = self.graph_db.get_converted_markdown_cid(
                metadata["pdf_ipfs_cid"], converter_func
            )

            # If the conversion already exists, use the existing conversion
            if converted_text_ipfs_cid:
                # Fetch converted text content from IPFS
                converted_text = self.graph_db._query_ipfs_content(
                    converted_text_ipfs_cid
                )
                if converted_text:
                    self.convert_cache[converter_func] = converted_text
                    self.logger.info("Using existing markdown conversion")
                else:
                    self.logger.warning(
                        "Failed to fetch existing conversion content, performing new conversion"
                    )
                    converted_text_ipfs_cid = None  # Reset to trigger new conversion

            # If no existing conversion was found or content could not be fetched, perform conversion
            if not converted_text_ipfs_cid or converter_func not in self.convert_cache:
                self.logger.info(
                    "No existing conversion found, performing new conversion"
                )
                if converter_func not in self.convert_cache:
                    converted_text = convert(
                        conversion_type=converter_func, input_path=pdf_path
                    )
                    self.convert_cache[converter_func] = converted_text
                else:
                    converted_text = self.convert_cache[converter_func]

                # Upload converted text to IPFS and commit to Git
                self.__write_to_file(converted_text, self.tmp_file_path)

                converted_text_ipfs_cid = self.__upload_text_to_lighthouse(self.tmp_file_path)

                self.graph_db.add_ipfs_node(converted_text_ipfs_cid)
                self.graph_db.create_relationship(
                    metadata["pdf_ipfs_cid"],
                    converted_text_ipfs_cid,
                    "CONVERTED_BY_" + converter_func,
                )
                self.graph_db.create_relationship(
                    converted_text_ipfs_cid, self.author_cid, "AUTHORED_BY"
                )

            # Step 2.2: Chunking
            self.logger.info(f"Starting chunking process for {converter_func}_{chunker_func}")
            converted_text = self.convert_cache[converter_func]
            chunk_cache_key = f"{converter_func}_{chunker_func}"
            if chunk_cache_key not in self.chunk_cache:
                self.logger.debug(f"Chunking text with {chunker_func}")
                chunked_text = chunk(
                    chunker_type=chunker_func, input_text=converted_text
                )
                self.chunk_cache[chunk_cache_key] = chunked_text
            else:
                chunked_text = self.chunk_cache[chunk_cache_key]
                self.logger.debug(f"Using cached chunks for {chunk_cache_key}")

            # Step 2.2.1: Process all chunks and get their CIDs
            chunk_cids = []
            self.logger.info(f"Processing {len(chunked_text)} chunks...")
            
            for chunk_i in chunked_text:
                self.__write_to_file(chunk_i, self.tmp_file_path)

                chunk_text_ipfs_cid = self.__upload_text_to_lighthouse(self.tmp_file_path)

                self.graph_db.add_ipfs_node(chunk_text_ipfs_cid)
                self.graph_db.create_relationship(
                    converted_text_ipfs_cid,
                    chunk_text_ipfs_cid,
                    "CHUNKED_BY_" + chunker_func,
                )
                self.graph_db.create_relationship(
                    chunk_text_ipfs_cid, self.author_cid, "AUTHORED_BY"
                )
                chunk_cids.append(chunk_text_ipfs_cid)

            # Step 2.3: Batch Embedding
            self.logger.info(f"Batch processing embeddings for {len(chunked_text)} chunks...")
            embeddings = embed_batch(embeder_type=embedder_func, input_texts=chunked_text, batch_size=32, user_email=self.user_email)
            
            # Step 2.4: Create embedding relationships
            for chunk_cid, embedding in zip(chunk_cids, embeddings):
                self.__write_to_file(json.dumps(embedding), self.tmp_file_path)

                embedding_ipfs_cid = self.__upload_text_to_lighthouse(self.tmp_file_path)
                self.graph_db.add_ipfs_node(embedding_ipfs_cid)
                self.graph_db.create_relationship(
                    chunk_cid,
                    embedding_ipfs_cid,
                    "EMBEDDED_BY_" + embedder_func,
                )
                self.graph_db.create_relationship(
                    embedding_ipfs_cid, self.author_cid, "AUTHORED_BY"
                )
                
            self.__update_mappings(metadata["pdf_ipfs_cid"], db_combination)
        
    def default_metadata(self) -> Dict[str, Any]:
        """Returns default metadata in case None is found.

        The metadata.json file aready exists for Arxiv papers.

        Returns:
            A dictionary with default metadata values
        """
        # Define a dictionary with an explicit type cast
        metadata: Dict[str, Any] = {}

        # Add each field individually
        metadata["title"] = "Unknown Title"
        metadata["authors"] = "Unknown Authors"
        metadata["categories"] = "Unknown Categories"
        metadata["abstract"] = "No abstract available."
        metadata["doi"] = "No DOI available"

        return metadata
