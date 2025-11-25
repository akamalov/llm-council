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
│  backend/llm_client.py (Unified API Client)     │
│  - LLMClient class with provider adapters       │
│  - query_model() - Single async query           │
│  - query_models_parallel() - Parallel queries   │
│  - Automatic fallback to OpenRouter             │
│  - Error handling with graceful degradation     │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  Direct Provider APIs + OpenRouter Fallback     │
│  ┌────────────┬────────────┬────────────┬─────┐ │
│  │  OpenAI    │ Anthropic  │  Google    │ OR  │ │
│  │  API       │  API       │  Gemini    │     │ │
│  └────────────┴────────────┴────────────┴─────┘ │
│  - GPT-5.1, Gemini-3-Pro, Claude-Sonnet-4.5    │
│  - Grok-4 (via OpenRouter)                      │
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

### 3. Provider System Implementation

The unified `llm_client.py` abstracts provider differences while maintaining a consistent interface.

**LLMClient Architecture:**

```python
class LLMClient:
    def __init__(self):
        # Load API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    async def query_model(self, model, provider, messages, timeout):
        # Route to appropriate provider
        if provider == "openai":
            return await self._query_openai(...)
        elif provider == "anthropic":
            return await self._query_anthropic(...)
        # ... etc
```

**Provider-Specific Adapters:**

**OpenAI** - Standard chat format:
```python
payload = {
    "model": model,
    "messages": messages  # Direct pass-through
}
```

**Anthropic** - System message extraction:
```python
# Extract system messages to separate parameter
system_content = None
anthropic_messages = []

for msg in messages:
    if msg['role'] == 'system':
        system_content = msg['content']
    else:
        anthropic_messages.append(msg)

payload = {
    "model": model,
    "messages": anthropic_messages,
    "max_tokens": 4096,
    "system": system_content  # Separate field
}
```

**Google Gemini** - Contents/parts format:
```python
# Convert to Gemini format
contents = []
for msg in messages:
    role = 'model' if msg['role'] == 'assistant' else 'user'
    contents.append({
        'role': role,
        'parts': [{'text': msg['content']}]
    })

payload = {"contents": contents}
```

**Fallback Logic:**
```python
async def _query_openai(self, model, messages, timeout):
    if not self.openai_api_key:
        print("OPENAI_API_KEY not found, falling back to OpenRouter")
        return await self._query_openrouter(f"openai/{model}", messages, timeout)
    # ... proceed with direct API
```

**Benefits:**
- **Single Interface**: `query_model(model, provider, messages)`
- **Format Translation**: Automatic message format conversion
- **Graceful Degradation**: Missing keys → OpenRouter fallback
- **Provider Agnostic**: Council logic doesn't care about providers

### 4. Reusing Existing Council Logic

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
                   ├──> council.py (Core Logic) ──> llm_client.py ──> Providers
MCP Server ────────┘
```

### 5. Running the Server

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

**Important Note on stdio Transport:**
MCP servers use stdio (standard input/output) for communication. The server waits for JSON-RPC messages on stdin and writes responses to stdout. When running in Docker, stdin must remain open using the `-i` flag, otherwise the container exits immediately when stdin closes.

## Extending the Implementation

### Adding a New Provider

To add support for a new provider (e.g., xAI Grok, Cohere, Mistral):

**Step 1: Add API Key to Config**
```python
# backend/config.py
XAI_API_KEY = os.getenv("XAI_API_KEY")
```

**Step 2: Implement Provider Adapter**
```python
# backend/llm_client.py
async def _query_xai(
    self,
    model: str,
    messages: List[Dict[str, str]],
    timeout: float
) -> Optional[Dict[str, Any]]:
    """Query xAI API (OpenAI-compatible)."""
    if not self.xai_api_key:
        print("XAI_API_KEY not found, falling back to OpenRouter")
        return await self._query_openrouter(f"x-ai/{model}", messages, timeout)

    headers = {
        "Authorization": f"Bearer {self.xai_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        data = response.json()
        message = data['choices'][0]['message']

        return {
            'content': message.get('content'),
        }
```

**Step 3: Route in Main Query Method**
```python
async def query_model(self, model, provider, messages, timeout):
    if provider == "xai":
        return await self._query_xai(model, messages, timeout)
    # ... existing providers
```

**Step 4: Update Config**
```python
# backend/config.py
COUNCIL_MODELS = [
    # ... existing models
    {
        "model": "grok-4",
        "provider": "xai",
        "display_name": "x-ai/grok-4"
    },
]
```

**Considerations:**
- **API Compatibility**: OpenAI-compatible APIs easiest to integrate
- **Message Format**: May need custom conversion (see Anthropic/Google examples)
- **Error Handling**: Provider-specific error codes and messages
- **Rate Limits**: Implement provider-specific backoff strategies

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
# ✅ GOOD: Environment variables
from .config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    OPENROUTER_API_KEY
)  # All loaded from .env

# ❌ BAD: Hardcoded
OPENAI_API_KEY = "sk-..."
```

**MCP Client Configuration:**
```json
{
  "mcpServers": {
    "llm-council": {
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
        "GEMINI_API_KEY": "${GEMINI_API_KEY}",
        "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY}"
      }
    }
  }
}
```

**Provider-Specific Security:**

**OpenAI**
- Rotate keys regularly via dashboard
- Monitor usage at platform.openai.com
- Set spending limits

**Anthropic**
- Keys prefixed with `sk-ant-`
- Organization-level access control
- Audit logs available

**Google**
- API key restrictions (HTTP referrers, IPs)
- Quota limits per project
- Cloud console monitoring

**Key Rotation Strategy:**
```python
# Support multiple keys with priority
OPENAI_API_KEYS = [
    os.getenv("OPENAI_API_KEY_PRIMARY"),
    os.getenv("OPENAI_API_KEY_SECONDARY"),
]

# Try primary, fallback to secondary if rate limited
for key in OPENAI_API_KEYS:
    try:
        return await query_with_key(key)
    except RateLimitError:
        continue
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

**Image:** `akamalov/llm-council:1.0`

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install uv
RUN uv sync

CMD ["uv", "run", "python", "-m", "backend.mcp_server"]
```

**Important**: MCP servers use stdio transport and require stdin to remain open. Without the `-i` flag, the container exits immediately when stdin closes.

**Running the Container:**

```bash
# Detached mode (background)
docker run -d -i --name llm-council akamalov/llm-council:1.0

# With environment variables from host
docker run -d -i \
  -e OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY \
  -e GEMINI_API_KEY \
  -e OPENROUTER_API_KEY \
  --name llm-council \
  akamalov/llm-council:1.0

# With port mapping (if using HTTP variant)
docker run -d -i -p 8000:8000 --name llm-council akamalov/llm-council:1.0
```

**MCP Client Configuration for Docker:**

```json
{
  "mcpServers": {
    "llm-council": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e", "OPENAI_API_KEY=${OPENAI_API_KEY}",
        "-e", "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}",
        "-e", "GEMINI_API_KEY=${GEMINI_API_KEY}",
        "-e", "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}",
        "akamalov/llm-council:1.0"
      ]
    }
  }
}
```

**Key Considerations:**
- The `-i` flag keeps stdin open for stdio communication
- Use `--rm` flag for automatic cleanup after each session
- Environment variables passed via `-e` flags
- No need for volume mounts (stateless cache)

### Multi-User Server

For shared deployment, consider HTTP transport:

```python
from mcp.server.fastmcp import FastMCP
import uvicorn

mcp = FastMCP("LLM Council")

if __name__ == "__main__":
    # Run as HTTP server instead of stdio
    uvicorn.run(mcp.get_asgi_app(), host="0.0.0.0", port=8000)
```

Then deploy via Docker:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install uv
RUN uv sync

EXPOSE 8000
CMD [". /app/.venv/bin/activate && python -m backend.mcp_server"]
```

**Note**: HTTP transport requires client support and different configuration than stdio transport.

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
- **Simplicity**: ~150 lines for MCP integration + ~250 lines for provider abstraction
- **Flexibility**: Multi-provider support with automatic fallback
- **Reusability**: Leverages existing council.py logic across web and MCP interfaces
- **Scalability**: Async operations support concurrent deliberations
- **Maintainability**: Clear separation of concerns with provider adapters

**Key Takeaways:**
1. FastMCP abstracts MCP protocol complexity
2. Provider abstraction enables direct API access with OpenRouter fallback
3. Format translation handles provider-specific message structures
4. Ephemeral cache suits 60-min TTL use case
5. Decorator-based tools keep code clean
6. Async/await enables parallel model queries across multiple providers
7. Structured errors improve debugging

**Provider System Benefits:**
- Lower latency via direct API connections
- Better rate limits per provider
- Cost optimization through provider selection
- Graceful degradation maintains reliability

For questions or contributions, see main README.md and CLAUDE.md.
