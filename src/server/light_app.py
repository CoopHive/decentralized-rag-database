# Light FastAPI server for quick endpoints (evaluation, status)
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import json
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

# Import your entry points
from src.query.evaluation_agent import EvaluationAgent
from src.scraper.openalex_scraper import OpenAlexScraper
from src.scraper.config import ScraperConfig
from src.utils.logging_utils import get_user_logger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a thread pool executor for CPU-intensive tasks
_thread_pool = ThreadPoolExecutor(max_workers=2)

# Setup FastAPI app
app = FastAPI(
    title="Light API",
    description="Light API - handles quick queries and status checks",
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
WHITELIST_PATH = Path(__file__).parent / "whitelisted_emails.txt"

def _scrape_papers_sync(scraper, cleanup_pdfs):
    """
    Synchronous helper function to scrape papers.
    This runs in a separate thread to avoid blocking the event loop.
    """
    try:
        return scraper.scrape_and_create_zip(cleanup_pdfs)
    except Exception as e:
        return False, str(e), [], None

def cleanup_zip_file(zip_path: str):
    """Background task to clean up zip file after serving."""
    # Use system logger for cleanup operations
    system_logger = get_user_logger("system", "light_server")
    try:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            system_logger.info(f"Cleaned up zip file: {zip_path}")
    except Exception as e:
        system_logger.warning(f"Error cleaning up zip file {zip_path}: {e}")

# Define request/response models
class EvaluationRequest(BaseModel):
    query: str
    collections: Optional[List[str]] = None  # If None, auto-discovers collections
    db_path: Optional[str] = None
    model_name: str = "openai/gpt-4o-mini"
    user_email: str

class UserStatusResponse(BaseModel):
    user_email: str
    total_papers: int
    completed_jobs: int
    completion_percentage: float
    papers_directory: str
    mappings_file_path: str

class EmailValidationRequest(BaseModel):
    email: EmailStr

class EmailValidationResponse(BaseModel):
    isValid: bool

class ResearchScrapeRequest(BaseModel):
    research_area: str
    user_email: str

def load_whitelisted_emails() -> set:
    """Load whitelisted emails from the file"""
    if not WHITELIST_PATH.exists():
        return set()
    
    with open(WHITELIST_PATH, 'r') as f:
        emails = {line.strip() for line in f 
                 if line.strip() and not line.startswith('#')}
    return emails

@app.post("/api/auth/validate-email", response_model=EmailValidationResponse)
async def validate_email(request: EmailValidationRequest):
    """Validate if an email is whitelisted"""
    # Create user-specific logger for this request
    user_logger = get_user_logger(request.email, "email_validation")
    
    try:
        whitelisted_emails = load_whitelisted_emails()
        email = request.email.lower()
        
        # Check if email is in whitelist
        is_valid = email in whitelisted_emails
        
        user_logger.info(f"Email validation request: {email} - Valid: {is_valid}")
        
        return EmailValidationResponse(isValid=is_valid)
    except Exception as e:
        user_logger.error(f"Error validating email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error validating email: {str(e)}")

@app.post("/api/research/scrape")
async def scrape_research_papers(request: ResearchScrapeRequest, background_tasks: BackgroundTasks):
    """Scrape research papers from OpenAlex and return zip file"""
    # Create user-specific logger for this request
    user_logger = get_user_logger(request.user_email, "research_scrape")
    
    try:
        user_logger.info(f"Starting research scrape for: {request.research_area}")
        user_logger.info(f"User email: {request.user_email}")
        
        # Validate research area
        if not request.research_area.strip():
            raise HTTPException(status_code=400, detail="Research area cannot be empty")
        
        # Create scraper configuration
        config = ScraperConfig.from_research_area(
            research_area=request.research_area.strip(),
            user_email=request.user_email
        )
        
        # Create scraper instance
        scraper = OpenAlexScraper(config)
        
        # Run scraping in the shared thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        success, result_message, downloaded_files, zip_path = await loop.run_in_executor(
            _thread_pool,
            _scrape_papers_sync,
            scraper,
            True  # cleanup_pdfs=True
        )
        
        if success and zip_path:
            user_logger.info(f"Successfully completed scraping for {request.user_email}")
            user_logger.info(f"Returning zip file: {zip_path}")
            
            # Schedule cleanup of the zip file after response is sent
            background_tasks.add_task(cleanup_zip_file, zip_path)
            
            # Get a clean filename for the download
            safe_topic = "".join(c for c in request.research_area[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
            download_filename = f"research_papers_{safe_topic}.zip"
            
            return FileResponse(
                path=zip_path,
                filename=download_filename,
                media_type='application/zip',
                headers={
                    "Content-Disposition": f"attachment; filename={download_filename}",
                    "X-Papers-Count": str(len(downloaded_files)),
                    "X-Research-Area": request.research_area[:100]
                }
            )
        else:
            user_logger.error(f"Scraping failed for {request.user_email}: {result_message}")
            raise HTTPException(status_code=404, detail=result_message)
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error scraping research papers: {str(e)}"
        user_logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/evaluate")
async def evaluate_endpoint(request: EvaluationRequest):
    """Endpoint for evaluation - fast queries"""
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
        
        user_logger.info(f"Successfully completed evaluation query")
        return results
    except Exception as e:
        user_logger.error(f"Error in evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_user_status(user_email: str):
    """Get processing status for a specific user - fast status check"""
    # Create user-specific logger for this request
    user_logger = get_user_logger(user_email, "status_check")
    
    jobs_file = PROJECT_ROOT / "temp" / "jobs.json"
    try:
        with open(jobs_file, 'r') as f:
            jobs = json.load(f)
        total_jobs, completed_jobs = jobs.get(user_email, [0, 0])
        
        user_logger.debug(f"Status check: {completed_jobs}/{total_jobs} jobs completed")
        
        return {
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "completion_percentage": (completed_jobs / total_jobs) * 100
        }
    except Exception as e:
        user_logger.error(f"Error getting user status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "light"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001) 