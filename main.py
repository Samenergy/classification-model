from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import json
from pathlib import Path
from typing import List, Dict, Any
import uvicorn
from pydantic import BaseModel

from model_service import ModelService
from data_processor import DataProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Article Relevance Classifier API",
    description="API for classifying news articles based on company objectives using transformer-based ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
model_service = ModelService()
data_processor = DataProcessor()

class ClassificationRequest(BaseModel):
    company_objective: str
    articles: List[Dict[str, Any]]

class ClassificationResponse(BaseModel):
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    model_info: Dict[str, Any]

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Article Relevance Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "/classify-upload": "Upload CSV file with company objective",
            "/classify-data": "Classify provided JSON data",
            "/health": "Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if model is loaded
        model_loaded = model_service.is_model_loaded()
        return {
            "status": "healthy" if model_loaded else "model_not_loaded",
            "model_loaded": model_loaded,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/classify-upload", response_model=ClassificationResponse)
async def classify_uploaded_file(
    file: UploadFile = File(..., description="CSV file containing articles"),
    company_objective: str = Form(..., description="Company objective for relevance classification")
):
    """
    Classify articles from uploaded CSV file based on company objective
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Read CSV content
        content = await file.read()
        
        # Process the CSV data
        df = data_processor.process_csv(content)
        
        # Validate required columns
        required_columns = ['title', 'content']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV file must contain columns: {missing_columns}"
            )
        
        # Classify articles
        results = model_service.classify_articles(
            df, 
            company_objective,
            use_custom_objective=True
        )
        
        return ClassificationResponse(**results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/classify-data", response_model=ClassificationResponse)
async def classify_data(request: ClassificationRequest):
    """
    Classify articles from provided JSON data based on company objective
    """
    try:
        # Convert JSON data to DataFrame
        df = pd.DataFrame(request.articles)
        
        # Validate required columns
        required_columns = ['title', 'content']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Data must contain columns: {missing_columns}"
            )
        
        # Classify articles
        results = model_service.classify_articles(
            df, 
            request.company_objective,
            use_custom_objective=True
        )
        
        return ClassificationResponse(**results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.post("/classify-predefined")
async def classify_with_predefined_model(
    file: UploadFile = File(..., description="CSV file containing articles"),
    use_case: str = Form(..., description="Predefined use case: mobile_wallet, healthcare_tech, or edtech")
):
    """
    Classify articles using predefined company objectives
    """
    try:
        # Validate use case
        valid_use_cases = ["mobile_wallet", "healthcare_tech", "edtech"]
        if use_case not in valid_use_cases:
            raise HTTPException(
                status_code=400, 
                detail=f"Use case must be one of: {valid_use_cases}"
            )
        
        # Get predefined objective
        company_objective = model_service.get_predefined_objective(use_case)
        
        # Read CSV content
        content = await file.read()
        
        # Process the CSV data
        df = data_processor.process_csv(content)
        
        # Validate required columns
        required_columns = ['title', 'content']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV file must contain columns: {missing_columns}"
            )
        
        # Classify articles
        results = model_service.classify_articles(
            df, 
            company_objective,
            use_custom_objective=False
        )
        
        return ClassificationResponse(**results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
