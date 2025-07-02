"""
Main module for the package.

This module contains the entry point for the processor, which handles
PDF processing, conversion, chunking, embedding, and storage in various databases.
"""

import hashlib
import os
import subprocess
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv

from src.core.processor import Processor
from src.utils.logging_utils import get_logger, get_user_logger

# Get module logger
logger = get_logger(__name__)

load_dotenv(override=True)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
# Parent directory of the project (coophive folder)
COOPHIVE_DIR = PROJECT_ROOT.parent

# Create a thread pool executor for CPU-intensive tasks
_thread_pool = ThreadPoolExecutor(max_workers=2)


def load_config():
    """
    Load configuration from the config file.

    Returns:
        dict: Configuration dictionary
    """
    config_path = PROJECT_ROOT / "config" / "processor.yml"

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def test_processor():
    """
    Test the document processing pipeline with sample papers.
    """
    # Load configuration
    config = load_config()

    # Process configuration
    processing_config = config["processing"]
    author_config = config["author"]
    api_keys = config["api_keys"]

    # Setup paths
    papers_directory = PROJECT_ROOT / processing_config["papers_directory"]
    metadata_file = PROJECT_ROOT / processing_config["metadata_file"]
    storage_directory = COOPHIVE_DIR / processing_config["storage_directory"]

    # Setup API keys and database connection
    lighthouse_api_key = os.getenv(
        api_keys["lighthouse_token"].replace("${", "").replace("}", "")
    )

    # Get papers
    papers = [
        os.path.join(papers_directory, f)
        for f in os.listdir(papers_directory)
        if f.endswith(".pdf")
    ][: processing_config["max_papers"]]

    # Setup database configurations
    databases = config["databases"]

    # Generate db_names from configurations
    for db_config in databases:
        converter = db_config["converter"]
        chunker = db_config["chunker"]
        embedder = db_config["embedder"]

        db_name = f"{converter}_{chunker}_{embedder}"
        db_config["db_name"] = db_name


    processor = Processor(
        authorPublicKey=author_config["public_key"],
        metadata_file=str(metadata_file),
        ipfs_api_key=lighthouse_api_key,
        user_email=author_config["email"],
        project_root=PROJECT_ROOT,
    )

    try:
        for paper in papers:
            logger.info(f"Processing {paper}...")
            random_data = os.urandom(32)
            hash_value = hashlib.sha256(random_data).hexdigest()

            paper_dir = storage_directory / hash_value
            try:
                os.makedirs(paper_dir, exist_ok=True)
                # Store the current directory
                current_dir = os.getcwd()
                # Change to the paper directory for git operations
                os.chdir(paper_dir)
                subprocess.run(["git", "init"], check=True)
                # Change back to the original directory after git init
                os.chdir(current_dir)
            except Exception as e:
                logger.error(f"Error initializing git repository: {e}")
                continue

            processor.process(pdf_path=paper, databases=databases, git_path=str(paper_dir))

        # Clean up: Delete all PDF files after processing
        logger.info("Starting cleanup: Deleting processed PDF files...")
        deleted_files = []
        
        for paper_path in papers:
            try:
                if os.path.exists(paper_path) and paper_path.lower().endswith('.pdf'):
                    filename = os.path.basename(paper_path)
                    os.remove(paper_path)
                    deleted_files.append(filename)
                    logger.info(f"Deleted: {filename}")
            except Exception as e:
                logger.error(f"Error deleting {paper_path}: {str(e)}")

        logger.info(f"Cleanup complete: Deleted {len(deleted_files)} PDF files from papers/ directory")
    except Exception as e:
        logger.error(f"Error in test_processor: {e}")
        raise


def _process_single_paper_sync(processor, paper_path, databases, paper_dir):
    """
    Synchronous helper function to process a single paper.
    This runs in a separate thread to avoid blocking the event loop.
    """
    try:
        processor.process(pdf_path=str(paper_path), databases=databases, git_path=str(paper_dir))
        return True, None
    except Exception as e:
        return False, str(e)


async def process_combination(converter: str, chunker: str, embedder: str, papers_list: List[str], user_papers_dir: str, user_email: str):
    """
    Process papers for a specific combination of converter, chunker, and embedder.
    
    Args:
        converter: The converter to use (e.g., 'markitdown', 'marker', 'openai')
        chunker: The chunker to use (e.g., 'paragraph', 'sentence', 'word', 'fixed_length')
        embedder: The embedder to use (e.g., 'openai', 'nvidia', 'bge')
        papers_list: List of paper file paths to process
        user_papers_dir: Path to the papers directory (user-specific)
        user_email: Email of the user for user-specific database path
    """
    # Create user-specific logger for this processing session
    user_logger = get_user_logger(user_email, "processor")
    
    user_logger.info(f"Processing combination: {converter}_{chunker}_{embedder}")
    
    # Load configuration
    config = load_config()
    
    # Process configuration
    processing_config = config["processing"]
    author_config = config["author"]
    api_keys = config["api_keys"]

    # Setup paths
    papers_directory = Path(user_papers_dir)
    metadata_file = PROJECT_ROOT / processing_config["metadata_file"]
    storage_directory = COOPHIVE_DIR / processing_config["storage_directory"]

    # Setup API keys and database connection
    lighthouse_api_key = os.getenv(
        api_keys["lighthouse_token"].replace("${", "").replace("}", "")
    )

    # Create database configuration for this specific combination
    db_config = {
        "converter": converter,
        "chunker": chunker,
        "embedder": embedder,
        "db_name": f"{converter}_{chunker}_{embedder}"
    }
    
    databases = [db_config]

    processor = Processor(
        authorPublicKey=author_config["public_key"],
        metadata_file=str(metadata_file),
        ipfs_api_key=lighthouse_api_key,
        user_email=user_email,
        project_root=PROJECT_ROOT,
    )

    # Process each paper
    for paper_filename in papers_list:
        # Construct full path to paper
        paper_path = papers_directory / paper_filename
        
        if not paper_path.exists():
            user_logger.warning(f"Paper not found: {paper_path}")
            continue
            
        user_logger.info(f"Processing {paper_path} with {converter}_{chunker}_{embedder}...")
        random_data = os.urandom(32)
        hash_value = hashlib.sha256(random_data).hexdigest()

        paper_dir = storage_directory / hash_value
        try:
            os.makedirs(paper_dir, exist_ok=True)
            # Store the current directory
            current_dir = os.getcwd()
            # Change to the paper directory for git operations
            os.chdir(paper_dir)
            subprocess.run(["git", "init"], check=True)
            # Change back to the original directory after git init
            os.chdir(current_dir)
        except Exception as e:
            user_logger.error(f"Error initializing git repository: {e}")
            continue

        try:
            # Run the CPU-intensive processing in a separate thread
            loop = asyncio.get_event_loop()
            success, error = await loop.run_in_executor(
                _thread_pool,
                _process_single_paper_sync,
                processor,
                paper_path,
                databases,
                str(paper_dir)
            )
            
            if success:
                user_logger.info(f"Successfully processed {paper_filename} with {converter}_{chunker}_{embedder}")
                increment_job_progress(user_email)
            else:
                user_logger.error(f"Error processing {paper_filename} with {converter}_{chunker}_{embedder}: {error}")
        except Exception as e:
            user_logger.error(f"Error processing {paper_filename} with {converter}_{chunker}_{embedder}: {e}")

        # Small async sleep to yield control back to the event loop
        await asyncio.sleep(0.1)


def increment_job_progress(user_email):
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
                    jobs[user_email][1] += 1
                    
                    with open(jobs_file, 'w') as f:
                        json.dump(jobs, f, indent=2)
                    user_logger.debug(f"Incremented job progress: {jobs[user_email][1]}/{jobs[user_email][0]}")
                else:
                    user_logger.error(f"Invalid job structure for {user_email}")
            else:
                user_logger.warning(f"No job found for {user_email}")
    except Exception as e:
        user_logger.error(f"Error updating job progress: {e}")


if __name__ == "__main__":
    logger.info("Running test_processor")
    test_processor()
