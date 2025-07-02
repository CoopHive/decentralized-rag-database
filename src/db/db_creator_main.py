"""
Database creation entrypoint.

This module serves as the main entrypoint for creating and populating databases
with scientific document data for the system.
"""

import os
from pathlib import Path
from typing import List
import json

import yaml
from dotenv import load_dotenv

from src.db.chroma_client import VectorDatabaseManager
from src.db.db_creator import DatabaseCreator
from src.db.graph_db import IPFSNeo4jGraph
from src.utils.logging_utils import get_logger, get_user_logger

# Get module logger
logger = get_logger(__name__)

load_dotenv()

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_config():
    """
    Load configuration from the config file.

    Returns:
        dict: Configuration dictionary
    """
    config_path = PROJECT_ROOT / "config" / "db_creator.yml"

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def increment_job_progress(user_email, increment=1):
    """Increment completed job count for user"""
    # Create user-specific logger for this operation
    user_logger = get_user_logger(user_email, "job_progress")
    
    jobs_file = PROJECT_ROOT / "temp" / "jobs.json"
    try:
        if jobs_file.exists():
            with open(jobs_file, 'r') as f:
                jobs = json.load(f)
            
            if user_email in jobs:
                if isinstance(jobs[user_email], list) and len(jobs[user_email]) >= 2:
                    jobs[user_email][1] += increment
                    
                    with open(jobs_file, 'w') as f:
                        json.dump(jobs, f, indent=2)
                    user_logger.debug(f"Incremented job progress: {jobs[user_email][1]}/{jobs[user_email][0]}")
                else:
                    user_logger.error(f"Invalid job structure for {user_email}")
            else:
                user_logger.warning(f"No job found for {user_email}")
    except Exception as e:
        user_logger.error(f"Error updating job progress: {e}")


def create_user_database(user_email: str):
    """
    Create and populate the database from IPFS CIDs for a specific user.
    
    Args:
        user_email: Email of the user for creating user-specific database
    """
    # Create user-specific logger for this operation
    user_logger = get_user_logger(user_email, "database_creation")
    
    # Load configuration
    config = load_config()

    # Extract Neo4j connection parameters
    neo4j_config = config["neo4j"]
    neo4j_uri = os.getenv(neo4j_config["uri"].replace("${", "").replace("}", ""))
    neo4j_username = os.getenv(
        neo4j_config["username"].replace("${", "").replace("}", "")
    )
    neo4j_password = os.getenv(
        neo4j_config["password"].replace("${", "").replace("}", "")
    )

    user_logger.info(f"Creating databases for user: {user_email}")

    # Initialize Neo4j graph
    graph = IPFSNeo4jGraph(
        uri=neo4j_uri, username=neo4j_username, password=neo4j_password
    )

    # Look for user-specific mappings file
    user_temp_path = PROJECT_ROOT / "temp" / user_email
    mappings_file_path = user_temp_path / "mappings.json"
    mapping_embed_file_path = user_temp_path / "mapping_embed.json"

    if not mappings_file_path.exists():
        user_logger.error(f"No mappings.json file found for user {user_email}. Please run processor first.")
        user_logger.error(f"Checked path: {mappings_file_path}")
        return

    # Load mappings
    with open(mappings_file_path, "r") as file:
        mappings = json.load(file)

    if not mappings:
        user_logger.warning(f"Empty mappings file for user {user_email}")
        return

    # Load existing embedded mappings if they exist
    mapping_embed = {}
    if mapping_embed_file_path.exists():
        try:
            with open(mapping_embed_file_path, "r") as file:
                mapping_embed = json.load(file)
            user_logger.info(f"Loaded existing mapping_embed.json with {len(mapping_embed)} entries")
        except Exception as e:
            user_logger.warning(f"Error loading mapping_embed.json: {e}. Starting fresh.")
            mapping_embed = {}
    else:
        user_logger.info("No existing mapping_embed.json found. Starting fresh.")

    # Find what needs to be processed (items in mappings but not in mapping_embed)
    items_to_process = {}
    for pdf_cid, db_combinations in mappings.items():
        if pdf_cid not in mapping_embed:
            # CID not embedded at all
            items_to_process[pdf_cid] = db_combinations
        else:
            # CID exists, check which combinations are missing
            existing_combinations = set(mapping_embed[pdf_cid])
            increment_job_progress(user_email, len(existing_combinations))
            new_combinations = [combo for combo in db_combinations if combo not in existing_combinations]
            if new_combinations:
                items_to_process[pdf_cid] = new_combinations

    if not items_to_process:
        user_logger.info("No new items to process. All mappings are already embedded.")
        return

    user_logger.info(f"Found {sum(len(combos) for combos in items_to_process.values())} new CID-database combinations to process")

    # Get all unique database names from items to process
    db_names = set()
    for pdf_cid, db_combinations in items_to_process.items():
        for db_combination in db_combinations:
            db_names.add(db_combination)
      
    user_logger.info(f"Discovered {len(db_names)} unique databases: {sorted(db_names)}")

    # Set up user-specific vector database path
    base_db_path = PROJECT_ROOT / config["vector_db"]["path"]
    user_db_path = base_db_path / user_email
    os.makedirs(user_db_path, exist_ok=True)
    user_logger.info(f"Using database path: {user_db_path}")

    # Initialize database manager with the specific db names found in mappings
    vector_db_manager = VectorDatabaseManager(list(db_names), db_path=str(user_db_path))
    create_db = DatabaseCreator(graph, vector_db_manager, user_email)

    # Process only the new CID-database combinations
    total_processed = 0
    successfully_processed = {}
    
    for pdf_cid, db_combinations in items_to_process.items():
        user_logger.info(f"Processing CID {pdf_cid} with {len(db_combinations)} new database combinations")
        
        successfully_processed[pdf_cid] = []
        
        for db_combination in db_combinations:
            # Parse database combination (format: converter_chunker_embedder)
            # Handle case where chunker name might contain underscores (e.g., fixed_length)
            parts = db_combination.split("_")
            if len(parts) < 3:
                user_logger.error(f"Invalid database combination format: {db_combination}")
                continue
            
            # For combinations like "markitdown_fixed_length_bge", we need to parse carefully
            # The format should be: converter_chunker_embedder
            # If we have more than 3 parts, the middle parts belong to the chunker
            converter = parts[0]
            embedder = parts[-1]  # Last part is always the embedder
            chunker = "_".join(parts[1:-1])  # Middle parts form the chunker name
            
            # Construct relationship path for this combination
            relationship_path = [
                f"CONVERTED_BY_{converter}",
                f"CHUNKED_BY_{chunker}",
                f"EMBEDDED_BY_{embedder}"
            ]
            
            try:
                create_db.process_paths(pdf_cid, relationship_path, db_combination)
                successfully_processed[pdf_cid].append(db_combination)
                total_processed += 1
                increment_job_progress(user_email, 1)
            except Exception as e:
                user_logger.error(f"Error processing {pdf_cid} with {db_combination}: {e}")
                continue

    # Update mapping_embed.json with successfully processed items
    for pdf_cid, successful_combinations in successfully_processed.items():
        if successful_combinations:  # Only update if we successfully processed some combinations
            if pdf_cid in mapping_embed:
                # Extend existing combinations
                existing_set = set(mapping_embed[pdf_cid])
                new_set = set(successful_combinations)
                mapping_embed[pdf_cid] = list(existing_set.union(new_set))
            else:
                # Add new CID
                mapping_embed[pdf_cid] = successful_combinations

    # Save updated mapping_embed.json
    try:
        with open(mapping_embed_file_path, "w") as file:
            json.dump(mapping_embed, file, indent=2)
        user_logger.info(f"Updated mapping_embed.json with {total_processed} new entries")
    except Exception as e:
        user_logger.error(f"Error saving mapping_embed.json: {e}")

    user_logger.info(f"Completed processing {total_processed} CID-database combinations for user {user_email}")


def main():
    """
    Main function to create and populate the database from IPFS CIDs.
    Uses the user email from config file.
    """
    # Load configuration
    config = load_config()
    
    # Get user email from config
    user_email = config["user"]["email"]
    
    # Call the user-specific database creation function
    create_user_database(user_email)


if __name__ == "__main__":
    main()
