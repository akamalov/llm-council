"""Configuration for the LLM Council."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from llm-council directory, not from cwd
_config_dir = Path(__file__).parent.parent
load_dotenv(_config_dir / ".env")

# API Keys - Direct provider access (falls back to OpenRouter if not provided)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of model configurations
# Each model has: model (identifier), provider (openai/anthropic/google/openrouter), display_name (for UI)
COUNCIL_MODELS = [
    {
        "model": "gpt-4-turbo",
        "provider": "openai",
        "display_name": "openai/gpt-4-turbo"
    },
    {
        "model": "gemini-2.5-flash",
        "provider": "google",
        "display_name": "google/gemini-2.5-flash"
    },
    {
        "model": "claude-sonnet-4-5-20250929",
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
    "model": "gemini-2.5-flash",
    "provider": "google",
    "display_name": "google/gemini-2.5-flash"
}

# Legacy OpenRouter API endpoint (for fallback)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage (absolute path)
DATA_DIR = str(_config_dir / "data" / "conversations")
