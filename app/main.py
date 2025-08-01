from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from contextlib import asynccontextmanager

# Configurations & Metrics
from core.config import settings
from core.monitoring import metrics
from core.logging import logger

# API Routes
from api.routes import router

# Import your LLM class
from models.llm import LLM

# Global variable to store the model instance
# llm_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_model

    # Startup: initialize LLM model
    try:
        logger.info("Initializing LLM model...")

        # Initialize LLM with your desired configuration
        app.state.llm_model = LLM(
            base_model="meta-llama/Llama-3.2-3B-Instruct",
            peft_dir="./models/qlora_adapter/checkpoint-18/",  # Use your checkpoint path
            vectorstore_dir="./database/chroma_db",
            load_existing_adapter=True,  # Set to True if you have trained adapters
        )

        # Optionally load data into vector store if needed
        # llm_model.load_data_to_vector_store("../data/prompt_response/")

        logger.info("LLM model initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize LLM model: {e}")
        raise RuntimeError(f"LLM model initialization failed: {e}")

    yield  # Application runs here

    # Shutdown: cleanup
    logger.info("Shutting down LLM model...")
    app.state.llm_model = None


# def get_llm_model():
#     """Get the global LLM model instance"""
#     global llm_model
#     if llm_model is None:
#         raise RuntimeError("LLM model not initialized")
#     return llm_model


app = FastAPI(title="Custom LLM API", root_path="/custom-llm-api", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/llm", tags=["LLM"])


@app.get("/")
async def root(request: Request):
    return RedirectResponse(url=request.scope.get("root_path", "") + "/docs")


@app.get("/health")
async def health():
    metrics.health_requests.inc()
    return {"status": "healthy"}


@app.get("/metrics")
def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.FASTAPI_PORT)
