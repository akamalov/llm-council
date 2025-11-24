"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys - Direct provider access (falls back to OpenRouter if not provided)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of model configurations
# Each model has: model (identifier), provider (openai/anthropic/google/openrouter), display_name (for UI)
COUNCIL_MODELS = [
    {
        "model": "gpt-5.1",
        "provider": "openai",
        "display_name": "openai/gpt-5.1"
    },
    {
        "model": "gemini-3-pro-preview",
        "provider": "google",
        "display_name": "google/gemini-3-pro-preview"
    },
    {
        "model": "claude-sonnet-4.5-20250929",
        "provider": "anthropic",
        "display_name": "anthropic/claude-sonnet-4.5"
    },
    {
        "model": "x-ai/grok-4",
        "provider": "openrouter",
        "display_name": "x-ai/grok-4"
    },
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = {
    "model": "gemini-3-pro-preview",
    "provider": "google",
    "display_name": "google/gemini-3-pro-preview"
}

# Legacy OpenRouter API endpoint (for fallback)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
