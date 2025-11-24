"""MCP Server for LLM Council deliberation."""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

from .council import run_full_council
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL

# Initialize FastMCP server
mcp = FastMCP("LLM Council")

# Ephemeral cache for deliberations (stored in memory with TTL)
_deliberation_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL_MINUTES = 60  # 1 hour TTL


def _cleanup_expired_cache():
    """Remove expired deliberations from cache."""
    current_time = datetime.now()
    expired_ids = [
        deliberation_id
        for deliberation_id, data in _deliberation_cache.items()
        if current_time - data["timestamp"] > timedelta(minutes=_CACHE_TTL_MINUTES)
    ]
    for deliberation_id in expired_ids:
        del _deliberation_cache[deliberation_id]


def _store_deliberation(
    stage1_results: list,
    stage2_results: list,
    stage3_result: dict,
    metadata: dict
) -> str:
    """
    Store deliberation results in ephemeral cache.

    Returns:
        deliberation_id: UUID string for retrieving the full breakdown
    """
    _cleanup_expired_cache()

    deliberation_id = str(uuid.uuid4())
    _deliberation_cache[deliberation_id] = {
        "timestamp": datetime.now(),
        "stage1": stage1_results,
        "stage2": stage2_results,
        "stage3": stage3_result,
        "metadata": metadata
    }

    return deliberation_id


@mcp.tool()
async def llm_council_deliberate(question: str) -> Dict[str, Any]:
    """
    Run a complete LLM Council deliberation on the given question.

    The council consists of multiple LLMs that:
    1. Provide individual responses (Stage 1)
    2. Anonymously rank each other's responses (Stage 2)
    3. A chairman synthesizes the final answer (Stage 3)

    Args:
        question: The question or decision to deliberate on

    Returns:
        Dict containing:
        - final_answer: The chairman's synthesized conclusion (markdown formatted)
        - deliberation_id: UUID for retrieving detailed stage breakdown
        - summary: Metadata about the deliberation (models used, stages completed)
    """
    # Run the full 3-stage council process
    stage1_results, stage2_results, stage3_result, metadata = await run_full_council(question)

    # Check for errors
    if stage3_result.get("model") == "error":
        return {
            "error": stage3_result.get("response", "Unknown error occurred"),
            "final_answer": None,
            "deliberation_id": None,
            "summary": None
        }

    # Store in cache for later inspection
    deliberation_id = _store_deliberation(
        stage1_results,
        stage2_results,
        stage3_result,
        metadata
    )

    # Return primary result with reference to full breakdown
    return {
        "final_answer": stage3_result.get("response", ""),
        "deliberation_id": deliberation_id,
        "summary": {
            "council_models": [m['display_name'] for m in COUNCIL_MODELS],
            "chairman_model": CHAIRMAN_MODEL['display_name'],
            "stages_completed": 3,
            "models_responded": len(stage1_results),
            "cache_ttl_minutes": _CACHE_TTL_MINUTES
        }
    }


@mcp.tool()
async def llm_council_inspect(deliberation_id: str) -> Dict[str, Any]:
    """
    Retrieve detailed breakdown of a previous deliberation by its ID.

    Shows all three stages:
    - Stage 1: Individual responses from each council model
    - Stage 2: Peer evaluations and rankings (with anonymization mapping)
    - Stage 3: Final synthesized answer from the chairman

    Args:
        deliberation_id: UUID returned from llm_council_deliberate

    Returns:
        Dict containing full stage breakdown and metadata, or error if not found/expired
    """
    _cleanup_expired_cache()

    if deliberation_id not in _deliberation_cache:
        return {
            "error": f"Deliberation ID '{deliberation_id}' not found or expired (TTL: {_CACHE_TTL_MINUTES} minutes)",
            "stage1": None,
            "stage2": None,
            "stage3": None,
            "metadata": None
        }

    data = _deliberation_cache[deliberation_id]

    return {
        "stage1": data["stage1"],
        "stage2": data["stage2"],
        "stage3": data["stage3"],
        "metadata": data["metadata"],
        "cached_at": data["timestamp"].isoformat(),
        "expires_at": (data["timestamp"] + timedelta(minutes=_CACHE_TTL_MINUTES)).isoformat()
    }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
