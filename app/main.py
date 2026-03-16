import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import check_database_health
from app.models import HealthResponse, QueryRequest, QueryResponse
from app.rag.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)

generator: AnswerGenerator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    generator = AnswerGenerator()
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Retrieval-Augmented Generation system for USCIS immigration policy. "
        "Provides grounded, auditable answers from the USCIS Policy Manual."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    """System health check including database connectivity."""
    db_health = check_database_health()
    return HealthResponse(
        status=db_health["status"],
        version=settings.app_version,
        database=db_health.get("status", "unknown"),
        timestamp=datetime.now(timezone.utc),
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Answer a USCIS policy question with full retrieval metadata and audit trail."""
    if generator is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    try:
        response = generator.answer(
            question=request.question,
            top_k=request.top_k,
        )
        return response
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your question.",
        )
