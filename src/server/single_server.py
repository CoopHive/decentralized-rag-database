# Create new file: src/server/single_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import os
import sys
import itertools
import asyncio
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# Import your entry points
from src.query.evaluation_agent import EvaluationAgent
from src.utils.gdrive_scraper import scrape_gdrive_pdfs
from src.core.processor_main import process_combination
from src.db.db_creator_main import create_user_database
from src.utils.logging_utils import get_user_logger

# Setup FastAPI app
app = FastAPI(
    title="SRC API",
    description="API - a decentralized RAG database",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
# Parent directory of the project (coophive folder)
COOPHIVE_DIR = PROJECT_ROOT.parent

# Define request/response models
class EvaluationRequest(BaseModel):
    query: str
    collections: Optional[List[str]] = None  # If None, auto-discovers collections
    db_path: Optional[str] = None
    model_name: str = "openai/gpt-4o-mini"
    user_email: str

class IngestGDriveRequest(BaseModel):
    drive_url: str = Field(..., description="Public Google Drive folder URL")
    converters: Optional[List[str]] = Field(
        default=["markitdown"], 
        description="List of converters to use (marker, openai, markitdown)"
    )
    chunkers: Optional[List[str]] = Field(
        default=["recursive"], 
        description="List of chunkers to use (fixed_length, recursive, markdown_aware, semantic_split)"
    )
    embedders: Optional[List[str]] = Field(
        default=["bge"], 
        description="List of embedders to use (openai, nvidia, bge)"
    )
    user_email: str

class IngestGDriveResponse(BaseModel):
    success: bool
    message: str
    downloaded_files: List[str]
    total_files: int
    processing_combinations: List[str]

class UserStatusResponse(BaseModel):
    user_email: str
    total_papers: int
    completed_jobs: int
    completion_percentage: float
    papers_directory: str
    mappings_file_path: str

@app.post("/api/evaluate")
async def evaluate_endpoint(request: EvaluationRequest):
    """Endpoint for evaluation (maps to run_evaluation.sh)"""
    # Create user-specific logger for this request
    user_logger = get_user_logger(request.user_email, "evaluation")
    
    try:
        user_logger.info(f"Evaluating query: {request.query}")
        user_logger.debug(f"DB Path: {request.db_path}")
        user_logger.debug(f"Model Name: {request.model_name}")
        user_logger.info(f"User Email: {request.user_email}")
        
        # Construct user-specific database path if not provided
        if request.db_path is None:
            base_db_path = PROJECT_ROOT / "src" / "database" / request.user_email
            user_db_path = str(base_db_path)
        else:
            user_db_path = request.db_path
        
        user_logger.info(f"Using database path: {user_db_path}")
        
        # Initialize evaluation agent
        agent = EvaluationAgent(model_name=request.model_name)

        # Run query on collections with user-specific database path
        results_file = agent.query_collections(
            query=request.query,
            db_path=user_db_path,
            user_email=request.user_email,
        )
        
        # Read the results from the JSON file
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        user_logger.info("Successfully completed evaluation query")
        return results
    except Exception as e:
        user_logger.error(f"Error in evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class EmbedRequest(BaseModel):
    user_email: str

@app.post("/api/embed")
async def embed_endpoint(request: EmbedRequest):
    """Endpoint to create user database"""
    # Create user-specific logger for this request
    user_logger = get_user_logger(request.user_email, "embed_database")
    
    try:
        user_logger.info(f"Creating database for user: {request.user_email}")
        
        # Create user database
        create_user_database(request.user_email)
        
        user_logger.info(f"Successfully created database for user: {request.user_email}")
        return {
            "success": True,
            "message": f"Successfully created database for user: {request.user_email}",
            "user_email": request.user_email
        }
    except Exception as e:
        user_logger.error(f"Error creating user database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating user database: {str(e)}")

@app.post("/api/ingest/gdrive", response_model=IngestGDriveResponse)
async def ingest_gdrive_pdfs(request: IngestGDriveRequest):
    """
    Ingest PDFs from a public Google Drive folder.
    
    This endpoint scrapes all PDFs from a public Google Drive folder
    and downloads them to a user-specific papers directory.
    It also creates a Cartesian product of all converter, chunker, and embedder combinations.
    """
    # Create user-specific logger for this request
    user_logger = get_user_logger(request.user_email, "gdrive_ingestion")
    
    try:
        user_logger.info(f"Processing Google Drive ingestion for user: {request.user_email}")
        user_logger.info(f"Drive URL: {request.drive_url}")
        
        # Validate the Google Drive URL
        if "drive.google.com" not in request.drive_url:
            raise HTTPException(
                status_code=400, 
                detail="Invalid Google Drive URL. Please provide a valid Google Drive folder link."
            )
        
        # Validate component lists
        valid_converters = ["marker", "openai", "markitdown"]
        valid_chunkers = ["fixed_length", "recursive", "markdown_aware", "semantic_split"]
        valid_embedders = ["openai", "nvidia", "bge"]
        
        # Validate requested components
        for converter in request.converters:
            if converter not in valid_converters:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid converter '{converter}'. Valid options: {valid_converters}"
                )
        
        for chunker in request.chunkers:
            if chunker not in valid_chunkers:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid chunker '{chunker}'. Valid options: {valid_chunkers}"
                )
        
        for embedder in request.embedders:
            if embedder not in valid_embedders:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid embedder '{embedder}'. Valid options: {valid_embedders}"
                )
        
        papers_directory = PROJECT_ROOT / "papers"

        # Create user-specific directory structure
        user_papers_dir = os.path.join(str(papers_directory), request.user_email)
        
        # Create directories if they don't exist
        os.makedirs(user_papers_dir, exist_ok=True)
        
        # Generate Cartesian product of all combinations
        processing_combinations = list(itertools.product(
            request.converters,
            request.chunkers,
            request.embedders
        ))
        
        combination_strings = [
            f"{converter}_{chunker}_{embedder}"
            for converter, chunker, embedder in processing_combinations
        ]
        
        user_logger.info(f"Will process {len(processing_combinations)} combinations")
        
        # Download PDFs from Google Drive to user-specific papers folder
        downloaded_files = scrape_gdrive_pdfs(
            drive_url=request.drive_url,
            download_dir=user_papers_dir,
            user_email=request.user_email
        )
        
        if not downloaded_files:
            user_logger.warning("No PDF files found or downloaded from the Google Drive folder")
            return IngestGDriveResponse(
                success=False,
                message="No PDF files found or downloaded from the Google Drive folder",
                downloaded_files=[],
                total_files=0,
                processing_combinations=combination_strings
            )
        
        user_logger.info(f"Downloaded {len(downloaded_files)} files, starting processing...")
        
        # Process each combination
        for converter, chunker, embedder in processing_combinations:
            try:
                user_logger.info(f"Processing combination: {converter}_{chunker}_{embedder}")
                # Call the processor for this specific combination with user-specific db path
                
                await process_combination(
                    converter=converter,
                    chunker=chunker,
                    embedder=embedder,
                    papers_list=downloaded_files,
                    user_papers_dir=user_papers_dir,
                    user_email=request.user_email
                )
                
                user_logger.info(f"Successfully completed combination: {converter}_{chunker}_{embedder}")
                # Create user database after processing
            except Exception as e:
                # Log the error but continue with other combinations
                user_logger.error(f"Error processing combination {converter}_{chunker}_{embedder}: {str(e)}")
        
        user_logger.info("All processing combinations completed")
        return IngestGDriveResponse(
            success=True,
            message=f"Successfully processed {len(downloaded_files)} PDF files with {len(processing_combinations)} combinations and cleaned up papers/ directory.",
            downloaded_files=downloaded_files,
            total_files=len(downloaded_files),
            processing_combinations=combination_strings
        )
        
    except ValueError as e:
        user_logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        user_logger.error(f"Error ingesting PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ingesting PDFs: {str(e)}")

@app.get("/api/status")
async def get_user_status(user_email: str):
    """Get processing status for a specific user"""
    # Create user-specific logger for this request
    user_logger = get_user_logger(user_email, "status_check")
    
    try:
        # Construct user-specific paths
        papers_directory = PROJECT_ROOT / "papers" / user_email
        mappings_file_path = PROJECT_ROOT / "temp" / user_email / "mappings.json"
        
        # Count papers in user's papers directory
        total_papers = 0
        if papers_directory.exists():
            # Count PDF files in the directory
            pdf_files = list(papers_directory.glob("*.pdf"))
            total_papers = len(pdf_files)
        
        # Count completed jobs from mappings file
        completed_jobs = 0
        if mappings_file_path.exists():
            with open(mappings_file_path, 'r') as f:
                mappings = json.load(f)
                # Count total database combinations processed across all PDFs
                for pdf_cid, db_combinations in mappings.items():
                    completed_jobs += len(db_combinations)
        
        # Calculate completion percentage
        if total_papers == 0:
            completion_percentage = 0.0
        else:
            completion_percentage = (completed_jobs / total_papers) * 100.0
        
        user_logger.debug(f"Status check: {completed_jobs} jobs completed for {total_papers} papers")
        
        return UserStatusResponse(
            user_email=user_email,
            total_papers=total_papers,
            completed_jobs=completed_jobs,
            completion_percentage=round(completion_percentage, 2),
            papers_directory=str(papers_directory),
            mappings_file_path=str(mappings_file_path)  # Updated to reflect new file
        )
        
    except Exception as e:
        user_logger.error(f"Error getting user status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting user status: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) 