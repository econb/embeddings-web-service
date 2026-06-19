import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is not configured"
    )

client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=30.0,
)

# ------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------

app = FastAPI(
    title="Embedding Service",
    version="1.0.0",
)

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
class EmbeddingRequest(BaseModel):
    text: str = Field(
        min_length=1,
        description="Text to embed"
    )

class EmbeddingResponse(BaseModel):
    dimension: int
    embedding: list[float]

# ------------------------------------------------------------------
# Exception Handlers
# ------------------------------------------------------------------
@app.exception_handler(AuthenticationError)
async def authentication_handler(request: Request, exc: AuthenticationError):
    logger.error("OpenAI authentication failed")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "OpenAI authentication failed"
        }
    )

@app.exception_handler(RateLimitError)
async def rate_limit_handler(request: Request, exc: RateLimitError):
    logger.warning("OpenAI rate limit exceeded")
    return JSONResponse(
        status_code=429,
        content={
            "success": False,
            "error": "OpenAI rate limit exceeded"
        }
    )

@app.exception_handler(APITimeoutError)
async def timeout_handler(request: Request, exc: APITimeoutError):
    logger.warning("OpenAI request timed out")
    return JSONResponse(
        status_code=504,
        content={
            "success": False,
            "error": "OpenAI request timed out"
        }
    )

@app.exception_handler(APIConnectionError)
async def connection_handler(request: Request, exc: APIConnectionError):
    logger.error("Unable to connect to OpenAI")
    return JSONResponse(
        status_code=503,
        content={
            "success": False,
            "error": "Unable to connect to OpenAI"
        }
    )

@app.exception_handler(InternalServerError)
async def internal_server_handler(request: Request, exc: InternalServerError):
    logger.error("OpenAI service unavailable")
    return JSONResponse(
        status_code=503,
        content={
            "success": False,
            "error": "OpenAI service temporarily unavailable"
        }
    )

@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    logger.exception("Unexpected error")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Unexpected server error"
        }
    )

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "success": True,
        "status": "healthy"
    }

@app.post("/embedding", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    logger.info("Input text: %s", text)

    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )

    embedding = response.data[0].embedding

    return {
        "dimension": len(embedding),
        "embedding": embedding,
    }
