# Light FastAPI server for quick endpoints (evaluation, status)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# Import your entry points
from descidb.query.evaluation_agent import EvaluationAgent

# Setup FastAPI app
app = FastAPI(
    title="DeSciDB Light API",
    description="Light API for DeSciDB - handles quick queries and status checks",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Define request/response models
class EvaluationRequest(BaseModel):
    query: str
    collections: Optional[List[str]] = None  # If None, auto-discovers collections
    db_path: Optional[str] = None
    model_name: str = "openai/gpt-3.5-turbo"
    user_email: str

class UserStatusResponse(BaseModel):
    user_email: str
    total_papers: int
    completed_jobs: int
    completion_percentage: float
    papers_directory: str
    mappings_file_path: str

class EmailValidationRequest(BaseModel):
    email: str

class EmailValidationResponse(BaseModel):
    isValid: bool

def load_whitelisted_emails() -> set:
    """Load whitelisted emails from the text file"""
    whitelist_path = Path(__file__).parent / "whitelisted_emails.txt"
    if not whitelist_path.exists():
        return set()
    
    with open(whitelist_path, 'r') as f:
        return {email.strip().lower() for email in f.readlines() if email.strip()}

@app.post("/api/auth/validate-email")
async def validate_email(request: EmailValidationRequest):
    """Validate if an email is whitelisted"""
    try:
        whitelisted_emails = load_whitelisted_emails()
        is_valid = request.email.lower() in whitelisted_emails
        return EmailValidationResponse(isValid=is_valid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating email: {str(e)}")

@app.post("/api/evaluate")
async def evaluate_endpoint(request: EvaluationRequest):
    """Endpoint for evaluation - fast queries"""
    try:
        print(f"[LIGHT] Evaluating query: {request.query}")
        print(f"[LIGHT] DB Path: {request.db_path}")
        print(f"[LIGHT] Model Name: {request.model_name}")
        print(f"[LIGHT] User Email: {request.user_email}")
        
        # Construct user-specific database path if not provided
        if request.db_path is None:
            base_db_path = PROJECT_ROOT / "descidb" / "database" / request.user_email
            user_db_path = str(base_db_path)
        else:
            user_db_path = request.db_path
        
        print(f"[LIGHT] Using database path: {user_db_path}")
        
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
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_user_status(user_email: str):
    """Get processing status for a specific user - fast status check"""
    try:
        print(f"[LIGHT] Getting status for user: {user_email}")
        
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
        
        return UserStatusResponse(
            user_email=user_email,
            total_papers=total_papers,
            completed_jobs=completed_jobs,
            completion_percentage=round(completion_percentage, 2),
            papers_directory=str(papers_directory),
            mappings_file_path=str(mappings_file_path)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting user status: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "light"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001) 