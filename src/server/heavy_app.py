# Heavy FastAPI server for resource-intensive endpoints (ingestion, processing)
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import itertools
import asyncio
import glob
import json
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# Import your entry points
from src.utils.gdrive_scraper import scrape_gdrive_pdfs
from src.core.processor_main import process_combination
from src.db.db_creator_main import create_user_database
from src.utils.logging_utils import get_user_logger

# Setup FastAPI app
app = FastAPI(
    title="Heavy API",
    description="Heavy API - handles resource-intensive processing tasks",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "https://coophive-wine.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Global lock for database creation to prevent concurrent write conflicts
_db_creation_lock = asyncio.Lock()

# Job tracking functions
def load_jobs():
    """Load jobs from temp/jobs.json"""
    jobs_file = PROJECT_ROOT / "temp" / "jobs.json"
    try:
        if jobs_file.exists():
            with open(jobs_file, 'r') as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        # Use a general logger for system-level operations
        logger = get_user_logger("system", "heavy_server")
        logger.error(f"Error loading jobs.json: {e}")
        return {}

def save_jobs(jobs_data):
    """Save jobs to temp/jobs.json"""
    jobs_file = PROJECT_ROOT / "temp" / "jobs.json"
    try:
        # Ensure temp directory exists
        jobs_file.parent.mkdir(exist_ok=True)
        with open(jobs_file, 'w') as f:
            json.dump(jobs_data, f, indent=2)
        # Use a general logger for system-level operations
        logger = get_user_logger("system", "heavy_server")
        logger.info("Updated jobs.json for user")
    except Exception as e:
        logger = get_user_logger("system", "heavy_server")
        logger.error(f"Error saving jobs.json: {e}")

# Background processing function
async def background_processing(
    processing_combinations: List[tuple],
    downloaded_files: List[str],
    user_papers_dir: str,
    user_email: str
):
    """Handle all processing in the background"""
    # Create user-specific logger for this processing session
    logger = get_user_logger(user_email, "background_processing")
    
    try:
        logger.info(f"Starting background processing for {user_email}")
        
        # Process each combination
        processing_tasks = []
        for converter, chunker, embedder in processing_combinations:
            logger.info(f"Creating task for combination: {converter}_{chunker}_{embedder}")
            task = asyncio.create_task(
                process_combination(
                    converter=converter,
                    chunker=chunker,
                    embedder=embedder,
                    papers_list=downloaded_files,
                    user_papers_dir=user_papers_dir,
                    user_email=user_email
                )
            )
            processing_tasks.append(task)
        
        # Wait for all processing tasks to complete concurrently
        logger.info(f"Running {len(processing_tasks)} combinations concurrently...")
        results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Log any errors from the concurrent processing
        for i, result in enumerate(results):
            converter, chunker, embedder = processing_combinations[i]
            if isinstance(result, Exception):
                logger.error(f"Error processing combination {converter}_{chunker}_{embedder}: {str(result)}")
            else:
                logger.info(f"Successfully completed combination {converter}_{chunker}_{embedder}")
        
        logger.info("All processing combinations completed")
        
        # Clean up: Delete all PDFs from user's papers directory after processing
        try:
            logger.info(f"Cleaning up PDFs from user directory: {user_papers_dir}")
            pdf_files = glob.glob(os.path.join(user_papers_dir, "*.pdf"))
            deleted_count = 0
            for pdf_file in pdf_files:
                try:
                    os.remove(pdf_file)
                    deleted_count += 1
                    logger.debug(f"Deleted: {os.path.basename(pdf_file)}")
                except Exception as delete_error:
                    logger.error(f"Error deleting {pdf_file}: {str(delete_error)}")
            
            logger.info(f"Successfully deleted {deleted_count} PDF files from user directory")
        except Exception as cleanup_error:
            logger.error(f"Error during PDF cleanup: {str(cleanup_error)}")

        # Automatically create user database after processing completes
        try:
            logger.info(f"Creating database for user: {user_email}")
            # Use async lock to prevent concurrent database creation conflicts
            async with _db_creation_lock:
                logger.debug(f"Acquired lock for database creation: {user_email}")
                # Run database creation directly in async context to avoid SQLite threading issues
                create_user_database(user_email)
            logger.info(f"Successfully created database for user: {user_email}")
        except Exception as db_error:
            logger.error(f"Error creating user database: {str(db_error)}")

        logger.info(f"Background processing completed for {user_email}")
        
    except Exception as e:
        logger.error(f"Error in background processing for {user_email}: {str(e)}")


# Define request/response models
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
    processing_started: bool

class EmbedRequest(BaseModel):
    user_email: str

@app.post("/api/ingest/gdrive", response_model=IngestGDriveResponse)
async def ingest_gdrive_pdfs(request: IngestGDriveRequest):
    """
    Ingest PDFs from a public Google Drive folder.
    
    This endpoint scrapes all PDFs from a public Google Drive folder,
    downloads them to a user-specific papers directory, and starts
    background processing for all converter, chunker, and embedder combinations.
    Returns immediately after starting the processing.
    """
    # Create user-specific logger for this request
    logger = get_user_logger(request.user_email, "gdrive_ingestion")
    
    try:
        logger.info(f"Processing Google Drive ingestion for user: {request.user_email}")
        logger.info(f"Drive URL: {request.drive_url}")
        
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
        logger.debug(f"Valid chunkers: {valid_chunkers}")
        for chunker in request.chunkers:
            if chunker not in valid_chunkers:
                logger.error(f"Invalid chunker: {chunker}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid chunker '{chunker}'. Valid options: {valid_chunkers}"
                )
        logger.debug(f"Validated chunkers: {request.chunkers}")
        
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
        
        logger.info(f"Will process {len(processing_combinations)} combinations")
        
        # Download PDFs from Google Drive to user-specific papers folder
        downloaded_files = scrape_gdrive_pdfs(
            drive_url=request.drive_url,
            download_dir=user_papers_dir,
            user_email=request.user_email
        )
        
        if not downloaded_files:
            logger.warning("No PDF files found or downloaded from the Google Drive folder")
            return IngestGDriveResponse(
                success=False,
                message="No PDF files found or downloaded from the Google Drive folder",
                downloaded_files=[],
                total_files=0,
                processing_combinations=combination_strings,
                processing_started=False
            )
        
        logger.info(f"Downloaded {len(downloaded_files)} files, starting background processing...")
        
        # Update job tracking when request comes in
        jobs = load_jobs()
        total_jobs = (len(processing_combinations) * len(downloaded_files)) * 2
        jobs[request.user_email] = [total_jobs, 0]  # [total_jobs, completed_jobs]
        save_jobs(jobs)
        logger.info(f"Initialized job tracking for {request.user_email}: 0/{total_jobs}")

        # Start background processing (don't await)
        asyncio.create_task(background_processing(
            processing_combinations=processing_combinations,
            downloaded_files=downloaded_files,
            user_papers_dir=user_papers_dir,
            user_email=request.user_email
        ))
        
        logger.info("Background processing started, returning response immediately")

        # Return immediately
        return IngestGDriveResponse(
            success=True,
            message=f"Processing started for {len(downloaded_files)} PDF files with {len(processing_combinations)} combinations. Check status for progress updates.",
            downloaded_files=downloaded_files,
            total_files=len(downloaded_files),
            processing_combinations=combination_strings,
            processing_started=True
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error ingesting PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ingesting PDFs: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "heavy"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002) 