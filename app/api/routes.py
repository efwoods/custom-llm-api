from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, RedirectResponse
from pydantic import BaseModel
from typing import Optional
import json
from core.monitoring import metrics
from core.config import settings
from core.logging import logger
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Import the function to get the LLM model
# Uncomment when ready to use
# from main import get_llm_model

router = APIRouter()


# Pydantic models for request/response validation
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 50
    top_k: Optional[int] = 10
    use_context: Optional[bool] = True
    do_sample: Optional[bool] = False
    temperature: Optional[float] = 0.7


class GenerateResponse(BaseModel):
    response: str
    prompt: str
    model_info: Optional[dict] = None


# Mock function for testing - remove when using real model
# def get_llm_model():
#     class MockLLM:
#         def __init__(self):
#             self.base_model = "meta-llama/Llama-3.2-3B-Instruct"
#             self.peft_dir = "./qlora_adapter"
#             self.vectorstore_dir = "./chroma_db"
#             self.device = "cpu"
#             self.load_existing_adapter = True

#         def generate_with_context(self, user_input, **kwargs):
#             return f"Mock response with context for: {user_input}"

#         def generate(self, prompt, **kwargs):
#             return f"Mock response for: {prompt}"

#         def get_vector_store_stats(self):
#             return {"total_documents": 100}

#     return MockLLM()


@router.websocket("/generate")
async def generate_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for text generation with full configuration options.

    Expected input format:
    {
        "prompt": "Your text here",
        "max_new_tokens": 50,
        "top_k": 10,
        "use_context": true,
        "do_sample": false,
        "temperature": 0.7
    }
    """
    await websocket.accept()
    metrics.active_websockets.inc()

    try:
        llm = websocket.app.state.llm_model
        logger.info("Generate WebSocket connection established")

        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                logger.info("Generate WebSocket message received")

                # Parse JSON message
                message = json.loads(data)

                # Validate required fields
                if "prompt" not in message:
                    await websocket.send_json(
                        {"error": "Missing 'prompt' field", "status": "error"}
                    )
                    continue

                # Extract parameters with defaults
                prompt = message["prompt"]
                max_new_tokens = message.get("max_new_tokens", 50)
                top_k = message.get("top_k", 10)
                use_context = message.get("use_context", True)
                do_sample = message.get("do_sample", False)
                temperature = message.get("temperature", 0.7)

                logger.info(f"Generating response for prompt: {prompt[:50]}...")

                # Generate response with or without context
                if use_context:
                    response = llm.generate_with_context(
                        user_input=prompt,
                        top_k=top_k,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                    )
                else:
                    response = llm.generate(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                    )

                # Update metrics
                metrics.health_requests.inc()
                logger.info("Response generated successfully")

                # Send response
                await websocket.send_json(
                    {
                        "response": response,
                        "prompt": prompt,
                        "model_info": {
                            "base_model": llm.base_model,
                            "use_context": use_context,
                            "vector_store_stats": llm.get_vector_store_stats(),
                        },
                        "status": "success",
                    }
                )

            except json.JSONDecodeError:
                await websocket.send_json(
                    {"error": "Invalid JSON format", "status": "error"}
                )
            except Exception as gen_error:
                logger.error(f"Generation error in WebSocket: {gen_error}")
                await websocket.send_json(
                    {"error": f"Generation failed: {str(gen_error)}", "status": "error"}
                )

    except WebSocketDisconnect:
        logger.info("Generate WebSocket disconnected")
    except Exception as e:
        logger.error(f"Generate WebSocket error: {e}")
        metrics.websocket_errors.inc()
    finally:
        metrics.active_websockets.dec()


@router.websocket("/chat")
async def chat_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for simplified chat interface.

    Expected input format:
    {
        "prompt": "Your message here",
        "max_new_tokens": 50 (optional)
    }
    """
    await websocket.accept()
    metrics.active_websockets.inc()

    try:
        llm = websocket.app.state.llm_model
        logger.info("Chat WebSocket connection established")

        # Send welcome message
        await websocket.send_json(
            {
                "message": "Chat WebSocket connected. Send your messages!",
                "status": "connected",
            }
        )

        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                logger.info("Chat WebSocket message received")

                # Parse JSON message
                message = json.loads(data)

                # Validate required fields
                if "prompt" not in message:
                    await websocket.send_json(
                        {"error": "Missing 'prompt' field", "status": "error"}
                    )
                    continue

                prompt = message["prompt"]
                max_new_tokens = message.get("max_new_tokens", 50)
                top_k = message.get("top_k", 10)
                do_sample = message.get("do_sample", False)
                temperature = message.get("temperature", 0.7)

                logger.info(f"Chat request: {prompt[:50]}...")

                response = llm.generate_with_context(
                    user_input=prompt,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                )

                await websocket.send_json(
                    {"message": response, "prompt": prompt, "status": "success"}
                )

            except json.JSONDecodeError:
                await websocket.send_json(
                    {"error": "Invalid JSON format", "status": "error"}
                )
            except Exception as chat_error:
                logger.error(f"Chat error in WebSocket: {chat_error}")
                await websocket.send_json(
                    {"error": f"Chat failed: {str(chat_error)}", "status": "error"}
                )

    except WebSocketDisconnect:
        logger.info("Chat WebSocket disconnected")
    except Exception as e:
        logger.error(f"Chat WebSocket error: {e}")
        metrics.websocket_errors.inc()
    finally:
        metrics.active_websockets.dec()


@router.websocket("/model-info")
async def model_info_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for getting model information.

    Send any message to get model info, or send:
    {
        "action": "get_info"
    }
    """
    await websocket.accept()
    metrics.active_websockets.inc()

    try:
        llm = websocket.app.state.llm_model
        logger.info("Model info WebSocket connection established")

        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                logger.info("Model info WebSocket message received")

                # Send model information
                model_info = {
                    "base_model": llm.base_model,
                    "peft_dir": llm.peft_dir,
                    "vectorstore_dir": llm.vectorstore_dir,
                    "device": llm.device,
                    "load_existing_adapter": llm.load_existing_adapter,
                    "vector_store_stats": llm.get_vector_store_stats(),
                    "status": "success",
                }

                await websocket.send_json(model_info)

            except json.JSONDecodeError:
                # Even if JSON is invalid, still send model info
                await websocket.send_json(
                    {
                        "base_model": llm.base_model,
                        "peft_dir": llm.peft_dir,
                        "vectorstore_dir": llm.vectorstore_dir,
                        "device": llm.device,
                        "load_existing_adapter": llm.load_existing_adapter,
                        "vector_store_stats": llm.get_vector_store_stats(),
                        "status": "success",
                    }
                )
            except Exception as info_error:
                logger.error(f"Model info error: {info_error}")
                await websocket.send_json(
                    {
                        "error": f"Failed to get model info: {str(info_error)}",
                        "status": "error",
                    }
                )

    except WebSocketDisconnect:
        logger.info("Model info WebSocket disconnected")
    except Exception as e:
        logger.error(f"Model info WebSocket error: {e}")
        metrics.websocket_errors.inc()
    finally:
        metrics.active_websockets.dec()


@router.websocket("/ws")
async def general_websocket(websocket: WebSocket):
    """
    General WebSocket endpoint for multi-purpose LLM interaction.

    Expected input format:
    {
        "action": "generate|chat|model_info",
        "prompt": "Your text here" (for generate/chat),
        "max_new_tokens": 50,
        "use_context": true,
        ... other parameters
    }
    """
    await websocket.accept()
    metrics.active_websockets.inc()

    try:
        llm = websocket.app.state.llm_model
        logger.info("General WebSocket connection established")

        # Send welcome message
        await websocket.send_json(
            {
                "message": "WebSocket connected. Available actions: 'generate', 'chat', 'model_info'",
                "status": "connected",
            }
        )

        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                logger.info(f"General WebSocket message received: {data}")

                # Parse JSON message
                message = json.loads(data)
                action = message.get("action", "generate")

                if action == "model_info":
                    # Send model information
                    model_info = {
                        "base_model": llm.base_model,
                        "peft_dir": llm.peft_dir,
                        "vectorstore_dir": llm.vectorstore_dir,
                        "device": llm.device,
                        "load_existing_adapter": llm.load_existing_adapter,
                        "vector_store_stats": llm.get_vector_store_stats(),
                        "status": "success",
                    }
                    await websocket.send_json(model_info)

                elif action in ["generate", "chat"]:
                    # Validate required fields
                    if "prompt" not in message:
                        await websocket.send_json(
                            {"error": "Missing 'prompt' field", "status": "error"}
                        )
                        continue

                    # Extract parameters
                    prompt = message["prompt"]
                    max_new_tokens = message.get("max_new_tokens", 50)
                    top_k = message.get("top_k", 10)
                    use_context = message.get("use_context", True)
                    do_sample = message.get("do_sample", False)
                    temperature = message.get("temperature", 0.7)

                    # Generate response
                    if use_context:
                        response = llm.generate_with_context(
                            user_input=prompt,
                            top_k=top_k,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature,
                        )
                    else:
                        response = llm.generate(
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature if do_sample else None,
                        )

                    # Send response
                    await websocket.send_json(
                        {
                            "response": response,
                            "prompt": prompt,
                            "action": action,
                            "status": "success",
                        }
                    )

                else:
                    await websocket.send_json(
                        {
                            "error": f"Unknown action: {action}. Available: 'generate', 'chat', 'model_info'",
                            "status": "error",
                        }
                    )

            except json.JSONDecodeError:
                await websocket.send_json(
                    {"error": "Invalid JSON format", "status": "error"}
                )
            except Exception as gen_error:
                logger.error(f"General WebSocket error: {gen_error}")
                await websocket.send_json(
                    {"error": f"Operation failed: {str(gen_error)}", "status": "error"}
                )

    except WebSocketDisconnect:
        logger.info("General WebSocket disconnected")
    except Exception as e:
        logger.error(f"General WebSocket error: {e}")
        metrics.websocket_errors.inc()
    finally:
        metrics.active_websockets.dec()


# Keep one GET endpoint for documentation
@router.get("/ws-info")
async def websocket_info():
    """
    Information about available WebSocket endpoints.
    """
    return {
        "endpoints": {
            "/generate": {
                "description": "Full-featured text generation",
                "input_format": {
                    "prompt": "Your text prompt (required)",
                    "max_new_tokens": "Maximum tokens to generate (optional, default: 50)",
                    "top_k": "Top-k for context retrieval (optional, default: 10)",
                    "use_context": "Whether to use vector store context (optional, default: true)",
                    "do_sample": "Whether to use sampling (optional, default: false)",
                    "temperature": "Temperature for sampling (optional, default: 0.7)",
                },
            },
            "/chat": {
                "description": "Simplified chat interface",
                "input_format": {
                    "prompt": "Your message (required)",
                    "max_new_tokens": "Maximum tokens to generate (optional, default: 50)",
                },
            },
            "/model-info": {
                "description": "Get model information",
                "input_format": "Any message or {'action': 'get_info'}",
            },
            "/ws": {
                "description": "General multi-purpose endpoint",
                "input_format": {
                    "action": "generate|chat|model_info",
                    "prompt": "Required for generate/chat actions",
                    "other_params": "Same as specific endpoints",
                },
            },
        },
        "example_usage": {
            "generate": {
                "prompt": "What is machine learning?",
                "max_new_tokens": 100,
                "use_context": True,
            },
            "chat": {"prompt": "Hello, how are you?"},
            "model_info": {"action": "model_info"},
        },
    }
