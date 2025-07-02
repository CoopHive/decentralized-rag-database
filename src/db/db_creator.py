"""
Database creation module.

This module provides functionality for creating and populating databases
with scientific document data for the system.
"""

import json
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.utils.logging_utils import get_logger, get_user_logger

# Get module logger
logger = get_logger(__name__)

load_dotenv()


class DatabaseCreator:
    """
    Creates and populates vector databases from graph relationships.

    This class retrieves embeddings and content from IPFS based on graph
    relationships and inserts them into ChromaDB collections.
    """

    def __init__(self, graph, vector_db_manager, user_email=None):
        """
        Initialize the DatabaseCreator.

        Args:
            graph: IPFSNeo4jGraph instance for graph database operations
            vector_db_manager: VectorDatabaseManager instance for vector database operations
            user_email: Optional user email for user-specific logging
        """
        self.graph = graph
        self.vector_db_manager = vector_db_manager
        self.user_email = user_email
        
        # Use user-specific logger if user_email is provided
        if user_email:
            self.logger = get_user_logger(user_email, "database_creator")
        else:
            self.logger = get_logger(__name__ + ".DatabaseCreator")

    def query_lighthouse_for_embedding(self, cid):
        """
        Query Lighthouse IPFS gateway for an embedding vector.

        Args:
            cid: IPFS CID of the embedding

        Returns:
            List representation of the embedding vector or None if retrieval fails
        """
        url = f"https://gateway.lighthouse.storage/ipfs/{cid}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            embedding_vector = json.loads(response.text)
            return embedding_vector
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to retrieve embedding for CID {cid}: {e}")
            return None

    def query_ipfs_content(self, cid):
        url = f"https://gateway.lighthouse.storage/ipfs/{cid}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            content = response.text
            return content
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to retrieve IPFS content for CID {cid}: {e}")
            return None

    def process_paths(self, start_cid, path, db_name):
        paths = self.graph.recreate_path(start_cid, path)

        if paths is False:
            self.logger.error(f"No valid paths found for CID {start_cid}")
            return

        self.logger.info(f"Found {len(paths)} paths for CID {start_cid}")

        for path_nodes in paths:
            if len(path_nodes) < 2:
                self.logger.error(f"Path {path_nodes} is too short.")
                continue

            content_cid = path_nodes[-2]
            embedding_cid = path_nodes[-1]

            embedding_vector = self.query_lighthouse_for_embedding(embedding_cid)
            if embedding_vector is None:
                self.logger.error(
                    f"Skipping path {path_nodes} due to failed embedding retrieval."
                )
                continue

            content = self.query_ipfs_content(content_cid)
            if content is None:
                self.logger.error(
                    f"Skipping path {path_nodes} due to failed IPFS content retrieval."
                )
                continue

            metadata = {
                "content_cid": content_cid,
                "root_cid": start_cid,
                "embedding_cid": embedding_cid,
                "content": content,
            }

            try:
                self.vector_db_manager.insert_document(
                    db_name, embedding_vector, metadata, embedding_cid
                )
                self.logger.info(
                    f"Inserted document into '{db_name}' with CID {embedding_cid}"
                )
            except Exception as e:
                self.logger.error(f"Failed to insert document into '{db_name}': {e}")
