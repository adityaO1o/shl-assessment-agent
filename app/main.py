"""FastAPI entrypoint for the SHL conversational recommendation system."""

from __future__ import annotations

import logging
from typing import Any, List

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.recommender import SHLRecommender
from app.retriever import SHLRetriever

logger = logging.getLogger(__name__)


retriever_instance = None
recommender_instance = None


class ChatMessage(BaseModel):
	"""Single chat message sent to the recommender."""

	model_config = ConfigDict(extra="forbid")

	role: str = Field(..., description="Message role, typically 'user' or 'assistant'")
	content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
	"""Request payload for the /chat endpoint."""

	model_config = ConfigDict(extra="forbid")

	messages: List[ChatMessage] = Field(..., min_length=1)


class RecommendationItem(BaseModel):
	"""Catalog-safe recommendation returned by the API."""

	model_config = ConfigDict(extra="forbid")

	name: str
	url: str | None = None
	test_type: str
    


class ChatResponse(BaseModel):
	"""Response payload returned by the /chat endpoint."""

	model_config = ConfigDict(extra="forbid")

	reply: str
	recommendations: List[RecommendationItem]
	end_of_conversation: bool


class HealthResponse(BaseModel):
	"""Health check response."""

	model_config = ConfigDict(extra="forbid")

	status: str = "ok"


app = FastAPI(
	title="SHL Conversational Recommendation API",
	version="1.0.0",
	description="Grounded, catalog-only SHL assessment recommendations.",
)


@app.on_event("startup")
async def preload_ai_components() -> None:
	"""Load heavyweight AI dependencies once during application startup."""
	global retriever_instance, recommender_instance

	logger.warning("SHL FastAPI application starting up")
	try:
		print("Loading embedding model...", flush=True)
		retriever_instance = SHLRetriever(auto_initialize=False)
		retriever_instance._load_dependencies()
		print("Building/loading FAISS index...", flush=True)
		retriever_instance.build_index()
		recommender_instance = SHLRecommender(retriever=retriever_instance)
		app.state.retriever = retriever_instance
		app.state.recommender = recommender_instance
		print("Recommender initialized successfully", flush=True)
	except Exception:
		logger.exception("Application startup failed while preloading AI components")
		raise


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
	"""Return a clean validation payload for malformed requests."""
	logger.warning("Request validation failed: %s", exc)
	return JSONResponse(
		status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
		content={"detail": exc.errors()},
	)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
	"""Preserve FastAPI HTTP exceptions with consistent JSON output."""
	logger.warning("HTTP error %s: %s", exc.status_code, exc.detail)
	return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
	"""Hide internal errors from clients while preserving observability."""
	logger.exception("Unhandled API error: %s", exc)
	return JSONResponse(
		status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
		content={"detail": "Internal server error"},
	)


def get_recommender(request: Request) -> SHLRecommender:
	"""Resolve the preloaded shared recommender instance."""
	recommender = getattr(request.app.state, "recommender", None) or recommender_instance
	if recommender is None:
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail="Recommendation service is temporarily unavailable",
		)
	return recommender


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
	"""Lightweight health endpoint for monitoring and evaluators."""
	return HealthResponse()


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(payload: ChatRequest, recommender: SHLRecommender = Depends(get_recommender)) -> ChatResponse:
	"""Generate grounded SHL recommendations for a conversation."""
	logger.info("Received chat request with %d messages", len(payload.messages))

	try:
		response = recommender.generate_reply([message.model_dump() for message in payload.messages])
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception("Recommender execution failed: %s", exc)
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to generate recommendations",
		) from exc

	reply = str(response.get("reply", ""))
	recommendations = response.get("recommendations") or []
	end_of_conversation = bool(response.get("end_of_conversation", False))

	safe_recommendations: List[RecommendationItem] = []
	for item in recommendations:
		if not isinstance(item, dict):
			continue
		try:
			safe_recommendations.append(RecommendationItem.model_validate(item))
		except Exception:
			logger.warning("Skipping non-serializable recommendation item: %r", item)

	return ChatResponse(
		reply=reply,
		recommendations=safe_recommendations,
		end_of_conversation=end_of_conversation,
	)


def main() -> None:
	"""Run the API with uvicorn when executed as a script."""
	import uvicorn

	uvicorn.run(
		"app.main:app",
		host="0.0.0.0",
		port=8000,
		reload=False,
		log_level="info",
	)


if __name__ == "__main__":
	main()
