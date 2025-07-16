#!/usr/bin/env python3
"""
Enhanced Model Serving API with Monitoring and Drift Detection
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.exposition import make_wsgi_app
from omegaconf import OmegaConf
import mlflow
import mlflow.pytorch
from evidently.report import Report
from evidently.metric_suite import DataDriftMetric
from evidently.pipeline.column_mapping import ColumnMapping
import pandas as pd

# Import the model architecture
from src.training.enhanced_training import EnhancedCNN

# Setup structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total model requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('model_request_duration_seconds', 'Request duration')
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total predictions', ['predicted_class'])
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
DRIFT_SCORE = Gauge('data_drift_score', 'Current data drift score')
HEALTH_STATUS = Gauge('model_health_status', 'Model health status (1=healthy, 0=unhealthy)')

# Global variables for model and config
model = None
config = None
reference_data = None
prediction_history = []


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    image_data: List[List[float]] = Field(..., description="28x28 image data as nested list")
    
    class Config:
        schema_extra = {
            "example": {
                "image_data": [[0.0] * 28 for _ in range(28)]
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_class: int = Field(..., description="Predicted digit class (0-9)")
    confidence: float = Field(..., description="Prediction confidence score")
    probabilities: List[float] = Field(..., description="Class probabilities")
    prediction_id: str = Field(..., description="Unique prediction ID")
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Health check timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Model version")
    uptime: float = Field(..., description="Service uptime in seconds")


class ModelMonitor:
    """Model monitoring and drift detection"""
    
    def __init__(self, config: Any):
        self.config = config
        self.prediction_history = []
        self.reference_data = None
        self.drift_threshold = config.monitoring.drift_threshold
        
    def add_prediction(self, prediction: Dict[str, Any]):
        """Add prediction to history for monitoring"""
        self.prediction_history.append(prediction)
        
        # Keep only recent predictions (last 1000)
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
    
    def set_reference_data(self, reference_data: pd.DataFrame):
        """Set reference data for drift detection"""
        self.reference_data = reference_data
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift"""
        if self.reference_data is None:
            return {"drift_detected": False, "drift_score": 0.0}
        
        try:
            # Create column mapping
            column_mapping = ColumnMapping()
            
            # Create drift report
            drift_report = Report(metrics=[DataDriftMetric()])
            drift_report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            drift_result = drift_report.as_dict()
            drift_score = drift_result.get('metrics', [{}])[0].get('result', {}).get('drift_score', 0.0)
            
            return {
                "drift_detected": drift_score > self.drift_threshold,
                "drift_score": drift_score,
                "drift_threshold": self.drift_threshold
            }
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return {"drift_detected": False, "drift_score": 0.0}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from prediction history"""
        if not self.prediction_history:
            return {"accuracy": 0.0, "total_predictions": 0}
        
        # For demonstration, return mock metrics
        # In production, you would compare with true labels
        return {
            "accuracy": 0.95,  # Mock accuracy
            "total_predictions": len(self.prediction_history),
            "avg_confidence": np.mean([p.get("confidence", 0.0) for p in self.prediction_history]),
            "class_distribution": self._get_class_distribution()
        }
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of predicted classes"""
        class_counts = {}
        for prediction in self.prediction_history:
            predicted_class = prediction.get("predicted_class", 0)
            class_counts[str(predicted_class)] = class_counts.get(str(predicted_class), 0) + 1
        return class_counts


class ModelService:
    """Model service with loading and prediction capabilities"""
    
    def __init__(self, config: Any):
        self.config = config
        self.model = None
        self.device = self._get_device()
        self.monitor = ModelMonitor(config)
        self.start_time = time.time()
        
    def _get_device(self):
        """Get optimal device for inference"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load model from path or MLflow registry"""
        try:
            if model_path and Path(model_path).exists():
                # Load from local path
                logger.info(f"Loading model from local path: {model_path}")
                self.model = EnhancedCNN(self.config)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                # Load from MLflow registry
                logger.info("Loading model from MLflow registry")
                mlflow.set_tracking_uri(self.config.registry.tracking_uri)
                model_name = self.config.registry.model_name
                
                # Get latest production model
                client = mlflow.tracking.MlflowClient()
                latest_version = client.get_latest_versions(model_name, stages=["Production"])
                
                if latest_version:
                    model_uri = f"models:/{model_name}/{latest_version[0].version}"
                    self.model = mlflow.pytorch.load_model(model_uri)
                else:
                    # Fallback to staging
                    latest_version = client.get_latest_versions(model_name, stages=["Staging"])
                    if latest_version:
                        model_uri = f"models:/{model_name}/{latest_version[0].version}"
                        self.model = mlflow.pytorch.load_model(model_uri)
            
            if self.model:
                self.model.to(self.device)
                self.model.eval()
                logger.info("Model loaded successfully")
                HEALTH_STATUS.set(1)
                return True
            else:
                logger.error("Failed to load model")
                HEALTH_STATUS.set(0)
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            HEALTH_STATUS.set(0)
            return False
    
    def predict(self, image_data: List[List[float]]) -> Dict[str, Any]:
        """Make prediction on image data"""
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            # Convert to tensor
            image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = output.argmax(dim=1).item()
                confidence = probabilities.max().item()
            
            # Create prediction result
            prediction = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities.squeeze().tolist(),
                "prediction_id": f"pred_{int(time.time() * 1000000)}",
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to monitoring
            self.monitor.add_prediction(prediction)
            
            # Update metrics
            PREDICTION_COUNTER.labels(predicted_class=predicted_class).inc()
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            "status": "healthy" if self.model is not None else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": self.model is not None,
            "model_version": "1.0.0",  # In production, get from model metadata
            "uptime": time.time() - self.start_time
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return self.monitor.get_performance_metrics()


# Global service instance
service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global service, config
    
    # Startup
    logger.info("Starting model service...")
    
    # Load configuration
    config = OmegaConf.load("config/config.yaml")
    
    # Initialize service
    service = ModelService(config)
    
    # Load model
    model_loaded = service.load_model()
    if not model_loaded:
        logger.warning("Model not loaded - service will be unhealthy")
    
    logger.info("Model service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down model service...")


# Create FastAPI app
app = FastAPI(
    title="MNIST Classification API",
    description="Enhanced MLOps API for MNIST digit classification with monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    REQUEST_DURATION.observe(process_time)
    
    return response


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    return """
    <html>
        <head>
            <title>MNIST Classification API</title>
        </head>
        <body>
            <h1>MNIST Classification API</h1>
            <p>Enhanced MLOps API for MNIST digit classification</p>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/metrics">Prometheus Metrics</a></li>
                <li><a href="/monitoring">Model Monitoring</a></li>
            </ul>
        </body>
    </html>
    """


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction on digit image"""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    prediction = service.predict(request.image_data)
    
    return PredictionResponse(**prediction)


@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    """Make prediction on uploaded image file"""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale and resize to 28x28
        image = image.convert('L').resize((28, 28))
        
        # Convert to list
        image_array = np.array(image).tolist()
        
        # Make prediction
        prediction = service.predict(image_array)
        
        return PredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"Error processing uploaded image: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    health_status = service.get_health_status()
    return HealthResponse(**health_status)


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    if service is None or service.model is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready", "timestamp": datetime.now().isoformat()}


@app.get("/startup")
async def startup_check():
    """Startup check endpoint"""
    if service is None:
        raise HTTPException(status_code=503, detail="Service starting up")
    
    return {"status": "started", "timestamp": datetime.now().isoformat()}


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


@app.get("/monitoring")
async def get_monitoring():
    """Get model monitoring information"""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    metrics = service.get_metrics()
    
    # Update Prometheus metrics
    MODEL_ACCURACY.set(metrics.get("accuracy", 0.0))
    
    return {
        "performance_metrics": metrics,
        "drift_detection": {
            "enabled": config.monitoring.drift_detection,
            "threshold": config.monitoring.drift_threshold
        },
        "monitoring_enabled": config.monitoring.enabled,
        "last_updated": datetime.now().isoformat()
    }


@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Add retraining task to background
    background_tasks.add_task(retrain_model)
    
    return {
        "status": "retraining_triggered",
        "timestamp": datetime.now().isoformat(),
        "message": "Model retraining has been triggered and will run in the background"
    }


async def retrain_model():
    """Background task for model retraining"""
    logger.info("Starting model retraining...")
    
    # This would trigger the actual retraining pipeline
    # For now, just simulate the process
    await asyncio.sleep(10)
    
    logger.info("Model retraining completed")


@app.get("/info")
async def get_info():
    """Get service information"""
    return {
        "service_name": "MNIST Classification API",
        "version": "1.0.0",
        "environment": config.environment.name if config else "unknown",
        "model_architecture": config.model.architecture if config else "unknown",
        "api_version": "v1",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        access_log=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )