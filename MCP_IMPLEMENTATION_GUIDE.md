# MCP Implementation Guide for Developers

This guide explains the technical implementation of the LLM Council MCP server for developers who want to understand, modify, or extend the codebase.

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│  MCP Client (Claude Code, etc.)                 │
│  - Discovers tools via MCP protocol             │
│  - Invokes tools with JSON parameters           │
│  - Receives structured JSON responses           │
└─────────────────┬───────────────────────────────┘
                  │ stdio transport
                  ▼
┌─────────────────────────────────────────────────┐
│  backend/mcp_server.py (FastMCP)                │
│  ┌───────────────────────────────────────────┐  │
│  │  @mcp.tool() llm_council_deliberate       │  │
│  │  - Validates input                        │  │
│  │  - Calls run_full_council()               │  │
│  │  - Stores result in ephemeral cache       │  │
│  │  - Returns {final_answer, deliberation_id}│  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  @mcp.tool() llm_council_inspect          │  │
│  │  - Retrieves from ephemeral cache         │  │
│  │  - Returns full stage breakdown           │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  Ephemeral Cache (in-memory dict)               │
│  - TTL: 60 minutes                               │
│  - Auto-cleanup on access                        │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  backend/council.py (Core Logic)                │
│  ┌───────────────────────────────────────────┐  │
│  │  run_full_council()                       │  │
│  │  ├─ stage1_collect_responses()            │  │
│  │  ├─ stage2_collect_rankings()             │  │
│  │  ├─ stage3_synthesize_final()             │  │
│  │  └─ calculate_aggregate_rankings()        │  │
│  └───────────────────────────────────────────┘  │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  backend/openrouter.py (API Client)             │
│  - query_model() - Single async query           │
│  - query_models_parallel() - Parallel queries   │
│  - Error handling with graceful degradation     │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  OpenRouter API                                 │
│  - Routes to: GPT-5.1, Gemini-3-Pro,            │
│    Claude-Sonnet-4.5, Grok-4                    │
└─────────────────────────────────────────────────┘
```

## Code Walkthrough

### 1. MCP Server Setup (`backend/mcp_server.py`)

#### Initialization

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("LLM Council")
```

**Why FastMCP?**
- High-level Python interface to MCP protocol
- Automatic tool registration via decorators
- Built-in stdio transport handling
- Type validation from function signatures

#### Ephemeral Cache Design

```python
_deliberation_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL_MINUTES = 60

def _cleanup_expired_cache():
    current_time = datetime.now()
    expired_ids = [
        deliberation_id
        for deliberation_id, data in _deliberation_cache.items()
        if current_time - data["timestamp"] > timedelta(minutes=_CACHE_TTL_MINUTES)
    ]
    for deliberation_id in expired_ids:
        del _deliberation_cache[deliberation_id]
```

**Design Decisions:**
- **In-memory storage**: No persistence layer needed for 60-min TTL
- **Dict-based**: O(1) lookups by UUID
- **Lazy cleanup**: Only runs on access, not via background thread
- **Thread-safe**: Python GIL protects dict operations for single-process server

**Trade-offs:**
- ✅ Simple implementation
- ✅ Fast access
- ❌ Lost on server restart
- ❌ Not suitable for multi-process deployment
- ❌ Memory grows with concurrent deliberations

**Alternative Approaches:**
- Redis: For multi-process/distributed deployments
- SQLite: For persistence across restarts
- LRU Cache: For bounded memory usage

### 2. Tool Implementation

#### Tool 1: `llm_council_deliberate`

```python
@mcp.tool()
async def llm_council_deliberate(question: str) -> Dict[str, Any]:
    """
    Run a complete LLM Council deliberation on the given question.
    ...
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
            "council_models": COUNCIL_MODELS,
            "chairman_model": CHAIRMAN_MODEL,
            "stages_completed": 3,
            "models_responded": len(stage1_results),
            "cache_ttl_minutes": _CACHE_TTL_MINUTES
        }
    }
```

**Key Implementation Details:**

1. **Decorator**: `@mcp.tool()` registers the function as an MCP tool
   - Function name becomes tool name
   - Docstring becomes tool description
   - Type hints define parameter schema

2. **Error Handling**: Returns structured error with null fields
   - MCP clients can check for `error` key
   - Graceful degradation preserves partial results

3. **Return Structure**: Optimized for primary use case
   - `final_answer`: What users care about most
   - `deliberation_id`: Optional deep-dive reference
   - `summary`: Metadata for transparency

#### Tool 2: `llm_council_inspect`

```python
@mcp.tool()
async def llm_council_inspect(deliberation_id: str) -> Dict[str, Any]:
    """
    Retrieve detailed breakdown of a previous deliberation by its ID.
    ...
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
```

**Design Rationale:**

1. **Separate Tool**: Follows MCP "single-purpose" best practice
   - Primary tool stays focused on decision delivery
   - Inspection is optional, invoked only when needed
   - Reduces token usage for typical use cases

2. **Explicit Expiration Info**: Returns `cached_at` and `expires_at`
   - Users understand time constraints
   - Enables proactive re-deliberation if needed

### 3. Reusing Existing Council Logic

```python
from .council import run_full_council
```

**Why Not Reimplement?**
- DRY principle: Web app and MCP use same logic
- Consistency: Same prompts, same anonymization
- Maintainability: Bug fixes benefit both interfaces
- Testing: Validate once, use everywhere

**Adapter Pattern:**
```
Web App (FastAPI) ─┐
                   ├──> council.py (Core Logic) ──> OpenRouter API
MCP Server ────────┘
```

### 4. Running the Server

```python
if __name__ == "__main__":
    mcp.run()
```

**What `mcp.run()` Does:**
1. Initializes stdio transport
2. Sends server capabilities to client
3. Enters event loop listening for tool calls
4. Handles protocol-level communication
5. Routes tool calls to decorated functions

**Process Model:**
- Single-threaded async event loop
- Each tool call runs as an async task
- Concurrent deliberations handled by asyncio
- No explicit multiprocessing (relies on OpenRouter concurrency)

## Extending the Implementation

### Adding a Third Tool

```python
@mcp.tool()
async def llm_council_compare(
    question1: str,
    question2: str
) -> Dict[str, Any]:
    """
    Compare council decisions on two related questions.
    """
    # Run deliberations in parallel
    results = await asyncio.gather(
        run_full_council(question1),
        run_full_council(question2)
    )

    stage1_results1, stage2_results1, stage3_result1, metadata1 = results[0]
    stage1_results2, stage2_results2, stage3_result2, metadata2 = results[1]

    return {
        "question1": {
            "final_answer": stage3_result1.get("response"),
            "aggregate_rankings": metadata1.get("aggregate_rankings")
        },
        "question2": {
            "final_answer": stage3_result2.get("response"),
            "aggregate_rankings": metadata2.get("aggregate_rankings")
        },
        "comparison": "..." # Add comparison logic
    }
```

### Custom Model Selection

To allow per-request model customization:

```python
@mcp.tool()
async def llm_council_deliberate_custom(
    question: str,
    council_models: Optional[List[str]] = None,
    chairman_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run deliberation with custom model selection.
    """
    # Use provided models or fall back to config
    models = council_models or COUNCIL_MODELS
    chairman = chairman_model or CHAIRMAN_MODEL

    # Would need to modify run_full_council to accept model parameters
    # Or duplicate logic with parameter injection
    ...
```

**Considerations:**
- Adds complexity to user interface
- Requires validation (models must exist in OpenRouter)
- Increases API costs unpredictably
- May violate expectations if results differ

### Persistent Storage

Replace in-memory cache with SQLite:

```python
import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db():
    conn = sqlite3.connect("deliberations.db")
    try:
        yield conn
    finally:
        conn.close()

def _store_deliberation(...):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO deliberations
            (id, timestamp, stage1, stage2, stage3, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            deliberation_id,
            datetime.now().isoformat(),
            json.dumps(stage1_results),
            json.dumps(stage2_results),
            json.dumps(stage3_result),
            json.dumps(metadata)
        ))
        conn.commit()
```

**Trade-offs:**
- ✅ Survives restarts
- ✅ Queryable history
- ❌ Disk I/O overhead
- ❌ Schema migrations needed
- ❌ More complex cleanup logic

## Testing

### Unit Testing Tools

```python
import pytest
from backend.mcp_server import llm_council_deliberate, llm_council_inspect

@pytest.mark.asyncio
async def test_deliberate_returns_structure():
    result = await llm_council_deliberate("What is 2+2?")

    assert "final_answer" in result
    assert "deliberation_id" in result
    assert "summary" in result
    assert result["summary"]["stages_completed"] == 3

@pytest.mark.asyncio
async def test_inspect_with_invalid_id():
    result = await llm_council_inspect("invalid-uuid")

    assert "error" in result
    assert result["stage1"] is None
```

### Integration Testing with MCP Inspector

FastMCP includes built-in inspector:

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Run server in one terminal
uv run python -m backend.mcp_server

# Connect inspector in another
mcp-inspector --connect stdio --command "uv run python -m backend.mcp_server"
```

**Inspector Features:**
- Interactive tool invocation
- JSON schema validation
- Response inspection
- Protocol debugging

### Load Testing

```python
import asyncio
import time

async def load_test():
    tasks = [
        llm_council_deliberate(f"Question {i}")
        for i in range(10)
    ]

    start = time.time()
    results = await asyncio.gather(*tasks)
    duration = time.time() - start

    print(f"10 concurrent deliberations: {duration:.2f}s")
    print(f"Average: {duration/10:.2f}s per deliberation")
```

**Expected Performance:**
- Stage 1: ~2-5s (parallel queries to 4 models)
- Stage 2: ~2-5s (parallel ranking queries)
- Stage 3: ~1-3s (single chairman query)
- **Total: ~5-13s** depending on model latency

## Security Considerations

### 1. API Key Protection

```python
# ✅ GOOD: Environment variable
from .config import OPENROUTER_API_KEY  # Loaded from .env

# ❌ BAD: Hardcoded
OPENROUTER_API_KEY = "sk-or-v1-..."
```

**MCP Client Configuration:**
```json
{
  "mcpServers": {
    "llm-council": {
      "env": {
        "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY}"
      }
    }
  }
}
```

### 2. Input Validation

Current implementation relies on type hints:

```python
async def llm_council_deliberate(question: str) -> Dict[str, Any]:
```

**Additional Validation:**
```python
async def llm_council_deliberate(question: str) -> Dict[str, Any]:
    # Length limits
    if len(question) > 10000:
        return {"error": "Question too long (max 10,000 chars)"}

    # Prompt injection detection (basic)
    dangerous_patterns = ["ignore previous", "disregard instructions"]
    if any(pattern in question.lower() for pattern in dangerous_patterns):
        return {"error": "Question contains disallowed patterns"}
```

### 3. Rate Limiting

```python
from collections import defaultdict
from datetime import datetime, timedelta

_rate_limit_tracker = defaultdict(list)
_MAX_CALLS_PER_HOUR = 10

def _check_rate_limit(client_id: str) -> bool:
    now = datetime.now()
    hour_ago = now - timedelta(hours=1)

    # Clean old entries
    _rate_limit_tracker[client_id] = [
        ts for ts in _rate_limit_tracker[client_id]
        if ts > hour_ago
    ]

    if len(_rate_limit_tracker[client_id]) >= _MAX_CALLS_PER_HOUR:
        return False

    _rate_limit_tracker[client_id].append(now)
    return True
```

### 4. Cost Controls

Each deliberation costs ~9 API calls. At $0.01/call average:
- **Single deliberation**: ~$0.09
- **10/hour**: ~$0.90/hour
- **100/day**: ~$9/day

**Mitigation:**
- Set OpenRouter budget limits
- Implement daily usage caps
- Add cost estimates to tool descriptions

## Deployment Patterns

### Development

```bash
uv run python -m backend.mcp_server
```

### Production (Claude Code)

```json
{
  "mcpServers": {
    "llm-council": {
      "command": "uv",
      "args": ["run", "python", "-m", "backend.mcp_server"],
      "cwd": "/absolute/path/to/llm-council"
    }
  }
}
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install uv
RUN uv sync

CMD ["uv", "run", "python", "-m", "backend.mcp_server"]
```

**Note**: stdio transport requires direct process communication; Docker mainly useful for packaging dependencies.

### Multi-User Server

For shared deployment, consider HTTP transport:

```python
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette

app = Starlette()
mcp_app = FastMCP("LLM Council")

# Mount MCP to HTTP endpoint
app.mount("/mcp", mcp_app.get_asgi_app())
```

Then deploy via Uvicorn/Gunicorn on a server.

## Monitoring & Debugging

### Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@mcp.tool()
async def llm_council_deliberate(question: str) -> Dict[str, Any]:
    logger.info(f"Deliberation requested: {question[:50]}...")

    try:
        result = await run_full_council(question)
        logger.info(f"Deliberation completed: {len(result[0])} models responded")
        return ...
    except Exception as e:
        logger.error(f"Deliberation failed: {e}", exc_info=True)
        return {"error": str(e)}
```

### Metrics

```python
from prometheus_client import Counter, Histogram

deliberations_total = Counter('deliberations_total', 'Total deliberations')
deliberation_duration = Histogram('deliberation_duration_seconds', 'Deliberation latency')

@mcp.tool()
async def llm_council_deliberate(question: str) -> Dict[str, Any]:
    deliberations_total.inc()

    with deliberation_duration.time():
        result = await run_full_council(question)

    return ...
```

## Common Pitfalls

### 1. Blocking I/O in Async Functions

```python
# ❌ BAD: Blocks event loop
def blocking_operation():
    time.sleep(5)

@mcp.tool()
async def llm_council_deliberate(question: str):
    blocking_operation()  # Freezes server!
```

```python
# ✅ GOOD: Use async operations
@mcp.tool()
async def llm_council_deliberate(question: str):
    await asyncio.sleep(5)  # Non-blocking
```

### 2. Unbounded Cache Growth

```python
# ❌ BAD: Never cleaned
_deliberation_cache[uuid] = data

# ✅ GOOD: TTL-based cleanup
_cleanup_expired_cache()
_deliberation_cache[uuid] = {"timestamp": datetime.now(), "data": data}
```

### 3. Exposing Internal Errors

```python
# ❌ BAD: Leaks stack traces
except Exception as e:
    return {"error": str(e)}  # May contain sensitive paths

# ✅ GOOD: Generic error messages
except Exception as e:
    logger.error(f"Internal error: {e}", exc_info=True)
    return {"error": "Internal server error"}
```

## Performance Optimization

### 1. Connection Pooling

```python
# In openrouter.py
import httpx

# ✅ Reuse client across requests
_http_client = httpx.AsyncClient(timeout=60.0)

async def query_model(model: str, messages: list):
    return await _http_client.post(...)
```

### 2. Caching Identical Questions

```python
import hashlib

def _hash_question(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()

@mcp.tool()
async def llm_council_deliberate(question: str) -> Dict[str, Any]:
    q_hash = _hash_question(question)

    # Check if identical question was asked recently
    if q_hash in _deliberation_cache:
        cached = _deliberation_cache[q_hash]
        if datetime.now() - cached["timestamp"] < timedelta(minutes=5):
            return cached["result"]

    # Run deliberation...
```

### 3. Partial Responses

For long-running deliberations, consider streaming:

```python
@mcp.tool()
async def llm_council_deliberate_stream(question: str):
    yield {"status": "stage1_starting"}
    stage1_results = await stage1_collect_responses(question)
    yield {"status": "stage1_complete", "models": len(stage1_results)}

    # Continue for stages 2 & 3...
```

**Note**: Requires MCP client support for streaming responses.

## Conclusion

The LLM Council MCP implementation demonstrates:
- **Simplicity**: ~150 lines for full MCP integration
- **Reusability**: Leverages existing council.py logic
- **Scalability**: Async operations support concurrent deliberations
- **Maintainability**: Clear separation of concerns

**Key Takeaways:**
1. FastMCP abstracts MCP protocol complexity
2. Ephemeral cache suits 60-min TTL use case
3. Decorator-based tools keep code clean
4. Async/await enables parallel model queries
5. Structured errors improve debugging

For questions or contributions, see main README.md and CLAUDE.md.
