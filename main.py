from contextlib import asynccontextmanager
import os
import asyncio
import time
import uuid
import json
import itertools
import argparse
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
import sys

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, APIError, RateLimitError, BadRequestError

# --- Logging Configuration ---
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RA-1")

# --- Pydantic Models ---
class TextContentPart(BaseModel):
    type: str = "text"
    text: str
class ImageUrl(BaseModel):
    url: str
class ImageContentPart(BaseModel):
    type: str = "image_url"
    image_url: ImageUrl
class ChatMessage(BaseModel):
    role: str
    content: str | List[TextContentPart | ImageContentPart]
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
class OpenAIResponseMessage(BaseModel):
    role: str = "assistant"
    content: str
class OpenAIChoice(BaseModel):
    index: int = 0
    message: OpenAIResponseMessage
    finish_reason: str = "stop"
class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage
class StreamDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None
class StreamChoice(BaseModel):
    index: int = 0
    delta: StreamDelta
    finish_reason: Optional[str] = None
class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]
class Model(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "Mothr"
class ModelList(BaseModel):
    object: str = "list"
    data: List[Model]

# --- App State & Helper Functions ---
AVAILABLE_MODELS: List[Model] = []

def update_available_models(config: dict):
    global AVAILABLE_MODELS
    base_models = [Model(id="ra-1"), Model(id="ra-1-pro")]
    palette_models = []
    for m in config.get("MODEL_PALETTE", []):
        model_id = m.get("model_name")
        if model_id:
            display_id = config.get("MODEL_ALIASES", {}).get(model_id, model_id)
            palette_models.append(Model(id=display_id, owned_by="provider"))
    combined_models = base_models + palette_models
    model_dict = {model.id: model for model in combined_models}
    AVAILABLE_MODELS = list(model_dict.values())
    logger.info(f"Finalized model list. Total models available: {len(AVAILABLE_MODELS)}")

def setup_session_logger(session_id: str, proxy_key: str) -> logging.Logger:
    # ... (implementation unchanged)
    return logger

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup events after config is validated."""
    logger.info("Application startup...")
    update_available_models(app.state.config)
    yield
    logger.info("Application shutdown.")

app = FastAPI(lifespan=lifespan)

# --- Core Logic ---
async def run_tasks_with_rate_limit(tasks: List[Any], limit: int, session_logger: logging.Logger) -> List[Any]:
    # ... (implementation unchanged)
    pass
async def call_openai_agent(session_logger: logging.Logger, agent_name: str, api_key: str, model_name: str, system_prompt: str, user_messages: List[ChatMessage], generation_config: Dict[str, Any], config: dict):
    # ... (implementation unchanged)
    pass
async def call_openai_agent_stream(session_logger: logging.Logger, agent_name: str, api_key: str, model_name: str, system_prompt: str, user_messages: List[ChatMessage], generation_config: Dict[str, Any], config: dict) -> AsyncGenerator[bytes, None]:
    # ... (implementation unchanged)
    pass
async def get_agent_model_config(session_logger: logging.Logger, user_question: str, config: dict) -> Dict[str, str]:
    # ... (implementation unchanged)
    pass

# --- API Endpoints ---
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def get_api_key(request: Request, api_key_header: str = Security(api_key_header)) -> str:
    if not api_key_header or not api_key_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    token = api_key_header.split(" ")[1]
    if token not in request.app.state.config.get("VALID_PROXY_KEYS", set()):
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return token

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=AVAILABLE_MODELS)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authenticated_proxy_key: str = Security(get_api_key)):
    config = request.app.state.config
    session_id = str(uuid.uuid4())
    session_logger = setup_session_logger(session_id, authenticated_proxy_key)
    # ... (rest of endpoint logic is largely the same, but uses `config` dict)
    body = await request.json()
    try:
        chat_request = ChatCompletionRequest.model_validate(body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    # ... the rest of your chat completion logic here
    return {"message": "Endpoint placeholder"}


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to Mothr API Endpoint."}

# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Parse args, letting --help exit immediately.
    parser = argparse.ArgumentParser(description="Run the Mothr API FastAPI server.", add_help=True)
    parser.add_argument("--port", type=int, default=None, help="REQUIRED: Port to run the API server on.")
    parser.add_argument("--rpm", type=int, default=0, help="Requests Per Minute limit.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for failed API calls.")
    parser.add_argument("--retry-delay", type=int, default=10, help="Seconds to wait between retries.")
    args = parser.parse_args()

    # 2. Collect all startup errors before trying to run.
    error_messages = []
    config = {}

    # 2a. Check for required .env configuration
    load_dotenv()
    try:
        missing_keys = []
        env_vars = {
            "PROXY_AUTH_KEY": os.getenv("PROXY_AUTH_KEY"), "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"), "MODEL_PALETTE": os.getenv("MODEL_PALETTE")
        }
        for key, value in env_vars.items():
            if not value: missing_keys.append(key)
        if missing_keys:
            raise ValueError(f"Required environment variables are not set: {', '.join(missing_keys)}")
        
        config["OPENAI_BASE_URL"] = env_vars["OPENAI_BASE_URL"]
        config["VALID_PROXY_KEYS"] = {key.strip() for key in env_vars["PROXY_AUTH_KEY"].split(',') if key.strip()}
        config["OPENAI_API_KEYS_LIST"] = [key.strip() for key in env_vars["OPENAI_API_KEY"].split(',') if key.strip()]
        config["openai_api_key_rotator"] = itertools.cycle(config["OPENAI_API_KEYS_LIST"])
        config["MODEL_PALETTE"] = json.loads(env_vars["MODEL_PALETTE"])
        
        aliases_string = os.getenv("MODEL_ALIASES")
        aliases, reverse_aliases = {}, {}
        if aliases_string:
            for pair in aliases_string.split(','):
                if ':' in pair:
                    original, alias = pair.rsplit(':', 1)
                    aliases[original.strip()] = alias.strip()
                    reverse_aliases[alias.strip()] = original.strip()
        config["MODEL_ALIASES"] = aliases
        config["REVERSE_MODEL_ALIASES"] = reverse_aliases

    except (ValueError, json.JSONDecodeError) as e:
        error_messages.append(f"Configuration Error: {e}")

    # 2b. Check for required --port argument
    if args.port is None:
        error_messages.append("Error: The following argument is required: --port")

    # 3. Report all errors at once and exit if any exist
    if error_messages:
        for error in error_messages: logger.critical(error)
        logger.info("Run with --help to see all available options.")
        sys.exit(1)

    # 4. If all checks pass, finalize config and run server
    config["RATE_LIMIT_PER_MINUTE"] = args.rpm
    config["MAX_RETRIES"] = args.max_retries
    config["RETRY_DELAY"] = args.retry_delay
    config["OPENAI_DEFAULT_MODEL_NAME"] = os.getenv("OPENAI_DEFAULT_MODEL_NAME", "google/gemini-2.5-pro")
    config["ROUTING_MODEL"] = os.getenv("ROUTING_MODEL", "x-ai/grok-4-fast:free")
    app.state.config = config

    # 5. Final validation of argument values
    if args.port < 1024 and os.geteuid() != 0:
        logger.error(f"Port {args.port} is a privileged port. Please run with sudo or as root.")
        sys.exit(1)
    if not 1 <= args.max_retries <= 100:
        logger.error(f"Max retries must be between 1 and 100. You provided: {args.max_retries}")
        sys.exit(1)

    # 6. Log final settings and run
    logger.info(f"Loaded {len(config['OPENAI_API_KEYS_LIST'])} OpenAI API keys and {len(config['VALID_PROXY_KEYS'])} proxy keys.")
    if config["RATE_LIMIT_PER_MINUTE"] > 0:
        logger.info(f"🚀 Rate limiting enabled: {config['RATE_LIMIT_PER_MINUTE']} requests per minute.")
    else:
        logger.info("🚀 Rate limiting is disabled.")
        logger.warning("Running without a local RPM limit may cause you to hit provider rate limits.")
    logger.info(f"🔁 Agent retry policy: {config['MAX_RETRIES']} max retries with a {config['RETRY_DELAY']}-second delay.")
    
    import uvicorn
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_config=log_config)