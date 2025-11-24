"""Unified LLM client supporting multiple providers (OpenAI, Anthropic, Google) with OpenRouter fallback."""

import httpx
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    Unified client for querying LLMs across different providers.

    Supports:
    - Direct API calls to OpenAI, Anthropic, Google Gemini
    - OpenRouter fallback for models without direct API keys
    """

    def __init__(self):
        """Initialize API keys from environment variables."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        # API endpoints
        self.openai_url = "https://api.openai.com/v1/chat/completions"
        self.anthropic_url = "https://api.anthropic.com/v1/messages"
        self.google_url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"

    async def query_model(
        self,
        model: str,
        provider: str,
        messages: List[Dict[str, str]],
        timeout: float = 120.0
    ) -> Optional[Dict[str, Any]]:
        """
        Query a model using the appropriate provider.

        Args:
            model: Model identifier (provider-specific format)
            provider: Provider name ("openai", "anthropic", "google", "openrouter")
            messages: List of message dicts with 'role' and 'content'
            timeout: Request timeout in seconds

        Returns:
            Response dict with 'content' and optional metadata, or None if failed
        """
        try:
            if provider == "openai":
                return await self._query_openai(model, messages, timeout)
            elif provider == "anthropic":
                return await self._query_anthropic(model, messages, timeout)
            elif provider == "google":
                return await self._query_google(model, messages, timeout)
            elif provider == "openrouter":
                return await self._query_openrouter(model, messages, timeout)
            else:
                print(f"Unknown provider: {provider}")
                return None
        except Exception as e:
            print(f"Error querying {provider}/{model}: {e}")
            return None

    async def _query_openai(
        self,
        model: str,
        messages: List[Dict[str, str]],
        timeout: float
    ) -> Optional[Dict[str, Any]]:
        """Query OpenAI API."""
        if not self.openai_api_key:
            print("OPENAI_API_KEY not found, falling back to OpenRouter")
            return await self._query_openrouter(f"openai/{model}", messages, timeout)

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self.openai_url, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()
            message = data['choices'][0]['message']

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details')
            }

    async def _query_anthropic(
        self,
        model: str,
        messages: List[Dict[str, str]],
        timeout: float
    ) -> Optional[Dict[str, Any]]:
        """Query Anthropic API."""
        if not self.anthropic_api_key:
            print("ANTHROPIC_API_KEY not found, falling back to OpenRouter")
            return await self._query_openrouter(f"anthropic/{model}", messages, timeout)

        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        # Convert messages format: Anthropic doesn't use 'system' role in messages array
        # Instead, system messages go in a separate 'system' parameter
        system_content = None
        anthropic_messages = []

        for msg in messages:
            if msg['role'] == 'system':
                system_content = msg['content']
            else:
                anthropic_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })

        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": 4096,
        }

        if system_content:
            payload["system"] = system_content

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self.anthropic_url, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()
            content_block = data['content'][0]

            return {
                'content': content_block.get('text', ''),
            }

    async def _query_google(
        self,
        model: str,
        messages: List[Dict[str, str]],
        timeout: float
    ) -> Optional[Dict[str, Any]]:
        """Query Google Gemini API."""
        if not self.google_api_key:
            print("GOOGLE_API_KEY not found, falling back to OpenRouter")
            return await self._query_openrouter(f"google/{model}", messages, timeout)

        url = self.google_url.format(model=model) + f"?key={self.google_api_key}"

        # Convert messages to Gemini format
        # Gemini uses 'contents' with 'parts' instead of OpenAI's message format
        contents = []
        system_instruction = None

        for msg in messages:
            if msg['role'] == 'system':
                system_instruction = msg['content']
            else:
                # Map 'assistant' to 'model' for Gemini
                role = 'model' if msg['role'] == 'assistant' else 'user'
                contents.append({
                    'role': role,
                    'parts': [{'text': msg['content']}]
                })

        payload = {
            "contents": contents,
        }

        if system_instruction:
            payload["system_instruction"] = {
                "parts": [{"text": system_instruction}]
            }

        headers = {
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()
            # Extract text from the first candidate's content parts
            candidate = data['candidates'][0]
            content_parts = candidate['content']['parts']
            text = content_parts[0].get('text', '')

            return {
                'content': text,
            }

    async def _query_openrouter(
        self,
        model: str,
        messages: List[Dict[str, str]],
        timeout: float
    ) -> Optional[Dict[str, Any]]:
        """Query OpenRouter API (fallback)."""
        if not self.openrouter_api_key:
            print("OPENROUTER_API_KEY not found, cannot query model")
            return None

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(self.openrouter_url, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()
            message = data['choices'][0]['message']

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details')
            }


# Global client instance
_client = LLMClient()


async def query_model(
    model: str,
    provider: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via the appropriate provider.

    Args:
        model: Model identifier
        provider: Provider name ("openai", "anthropic", "google", "openrouter")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional metadata, or None if failed
    """
    return await _client.query_model(model, provider, messages, timeout)


async def query_models_parallel(
    model_configs: List[Dict[str, str]],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        model_configs: List of dicts with 'model', 'provider', and 'display_name' keys
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping display_name to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    tasks = [
        query_model(config['model'], config['provider'], messages)
        for config in model_configs
    ]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map display names to their responses
    return {
        config['display_name']: response
        for config, response in zip(model_configs, responses)
    }
