# LLM Council MCP Server

This document explains how to use LLM Council as an MCP (Model Context Protocol) server in Claude Code or other MCP-compatible clients.

## Overview

The LLM Council MCP server exposes two tools that allow AI assistants to leverage multi-model deliberation for complex decisions:

1. **`llm_council_deliberate`** - Run a complete 3-stage deliberation process
2. **`llm_council_inspect`** - Retrieve detailed breakdown of a previous deliberation

**Location Independent:** The MCP server works from any directory. All paths (config, logs, data) are resolved relative to the llm-council installation directory, not your current working directory. This means you can use llm-council from any project.

## Installation & Setup

### Installation Options

Choose between local Python installation or Docker deployment:

#### Option A: Local Python Installation

**Prerequisites:**
- Python 3.10+ installed
- `uv` package manager installed
- API keys in `.env` file (multiple options)

**Install Dependencies:**
```bash
uv sync
```

#### Option B: Docker Deployment (Recommended)

**Prerequisites:**
- Docker installed
- API keys available as environment variables

**Pre-built Image:**
```bash
docker pull akamalov/llm-council:1.0
```

**Build from Source:**
```bash
docker build -t akamalov/llm-council:1.0 .
```

### API Keys Configuration

Get your API keys from:
- OpenAI: [platform.openai.com](https://platform.openai.com/)
- Anthropic: [console.anthropic.com](https://console.anthropic.com/)
- Google: [ai.google.dev](https://ai.google.dev/)
- OpenRouter: [openrouter.ai](https://openrouter.ai/)

**Option A: Direct Provider APIs (Recommended)**
```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
export OPENROUTER_API_KEY=sk-or-v1-...  # Optional fallback
```

**Option B: OpenRouter Only**
```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

### 3. Configure MCP Client

Add the LLM Council server to your MCP client configuration.

#### For Local Python Installation

**Claude Code (`.claude/mcp.json`):**

```json
{
  "mcpServers": {
    "llm-council": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "backend.mcp_server"
      ],
      "cwd": "/absolute/path/to/llm-council",
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

**Important**:
- Replace `/absolute/path/to/llm-council` with the actual absolute path to your project directory
- The `${VAR}` syntax pulls from your environment variables
- You can omit keys you don't have - the system will fall back to OpenRouter

#### For Docker Deployment

**Claude Code (`.claude/mcp.json`):**

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

**Docker Configuration Notes:**
- The `-i` flag keeps stdin open for stdio communication (required for MCP servers)
- The `--rm` flag automatically removes the container after each session
- Environment variables are passed via `-e` flags
- No volume mounts needed (cache is ephemeral in-memory)
- The `${VAR}` syntax pulls from your host environment variables

**Why `-i` flag is required:**
MCP servers use stdio (standard input/output) for communication. Without the `-i` flag, stdin closes immediately when the container starts, causing the server to exit. The `-i` flag keeps stdin open, allowing the MCP client to communicate with the server.

### 4. Restart Claude Code

After adding the configuration, restart Claude Code to load the new MCP server.

## Usage

### Tool 1: `llm_council_deliberate`

Runs the complete 3-stage LLM Council deliberation process.

**Parameters:**
- `question` (string, required): The question or decision to deliberate on

**Returns:**
```json
{
  "final_answer": "The chairman's synthesized conclusion (markdown)",
  "deliberation_id": "uuid-string",
  "summary": {
    "council_models": ["openai/gpt-5.1", "google/gemini-3-pro-preview", ...],
    "chairman_model": "google/gemini-3-pro-preview",
    "stages_completed": 3,
    "models_responded": 4,
    "cache_ttl_minutes": 60
  }
}
```

**Example Usage in Claude Code:**

```
User: Use llm-council to decide: Should I refactor this authentication module?

Claude: I'll consult the LLM Council for this architectural decision.
[Calls llm_council_deliberate with question]

Response shows:
- Final synthesized answer from the chairman
- deliberation_id for detailed inspection
- Summary of which models participated
```

### Tool 2: `llm_council_inspect`

Retrieves the detailed breakdown of all three stages from a previous deliberation.

**Parameters:**
- `deliberation_id` (string, required): UUID returned from `llm_council_deliberate`

**Returns:**
```json
{
  "stage1": [
    {
      "model": "openai/gpt-5.1",
      "response": "Individual model's response..."
    },
    ...
  ],
  "stage2": [
    {
      "model": "openai/gpt-5.1",
      "ranking": "Full evaluation text with FINAL RANKING: ...",
      "parsed_ranking": ["Response C", "Response A", "Response B"]
    },
    ...
  ],
  "stage3": {
    "model": "google/gemini-3-pro-preview",
    "response": "Chairman's final synthesis..."
  },
  "metadata": {
    "label_to_model": {
      "Response A": "openai/gpt-5.1",
      "Response B": "google/gemini-3-pro-preview",
      ...
    },
    "aggregate_rankings": [
      {
        "model": "openai/gpt-5.1",
        "average_rank": 1.5,
        "rankings_count": 4
      },
      ...
    ]
  },
  "cached_at": "2025-11-24T10:30:00",
  "expires_at": "2025-11-24T11:30:00"
}
```

**Example Usage:**

```
User: Show me the individual model responses from that deliberation

Claude: I'll retrieve the detailed breakdown.
[Calls llm_council_inspect with the deliberation_id]

Response shows:
- All individual model responses (Stage 1)
- Each model's peer evaluation and ranking (Stage 2)
- The de-anonymization mapping
- Aggregate rankings showing which models performed best
- Final chairman synthesis (Stage 3)
```

## How It Works

### Stage 1: Individual Responses
All council models (GPT-5.1, Gemini-3-Pro, Claude-Sonnet-4.5, Grok-4) receive the question simultaneously and provide independent responses.

### Stage 2: Anonymized Peer Review
Each model receives all Stage 1 responses with anonymized labels (Response A, B, C, D). They evaluate and rank the responses without knowing which model produced which answer, preventing bias.

### Stage 3: Chairman Synthesis
The chairman model (Gemini-3-Pro) receives all individual responses and peer evaluations, then synthesizes a final answer representing the council's collective wisdom.

## Ephemeral Cache

Deliberation results are cached in memory for **60 minutes** to allow inspection via `llm_council_inspect`. After expiration, detailed breakdowns are no longer available, but the final answer remains in the conversation history.

**Cache Cleanup:**
- Automatic cleanup runs before each new deliberation and inspection
- No persistent storage - cache resets when server restarts
- Thread-safe for concurrent deliberations

## Configuration

### Customizing Council Models

Edit `backend/config.py` to change which models participate. Each model now specifies its provider:

```python
COUNCIL_MODELS = [
    {
        "model": "gpt-5.1",
        "provider": "openai",  # Direct API or falls back to OpenRouter
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
        "provider": "openrouter",  # Always uses OpenRouter
        "display_name": "x-ai/grok-4"
    },
]

CHAIRMAN_MODEL = {
    "model": "gemini-3-pro-preview",
    "provider": "google",
    "display_name": "google/gemini-3-pro-preview"
}
```

**Provider Options:**
- `openai` - Direct OpenAI API (requires `OPENAI_API_KEY`)
- `anthropic` - Direct Anthropic API (requires `ANTHROPIC_API_KEY`)
- `google` - Direct Google Gemini API (requires `GEMINI_API_KEY`)
- `openrouter` - OpenRouter routing service (requires `OPENROUTER_API_KEY`)

**Automatic Fallback:** If a direct API key is missing, the system automatically falls back to OpenRouter for that provider.

### Adjusting Cache TTL

In `backend/mcp_server.py`, modify:

```python
_CACHE_TTL_MINUTES = 60  # Change to desired minutes
```

## Troubleshooting

### Server Not Loading

**Local Python Installation:**
1. Check that `.env` contains valid API keys
2. Verify absolute path in MCP configuration
3. Ensure `uv sync` completed successfully
4. Check logs in Claude Code's MCP server panel

**Docker Deployment:**
1. Verify Docker image exists: `docker images | grep llm-council`
2. Check API keys are exported as environment variables
3. Ensure `-i` flag is present in MCP configuration
4. Test container manually: `docker run -i --rm akamalov/llm-council:1.0`
5. Check logs: `docker logs <container-id>`

### Container Exits Immediately

**Symptom:** Docker container starts and exits right away without errors.

**Cause:** Missing `-i` flag in Docker run command. MCP servers require stdin to remain open.

**Solution:** Ensure MCP configuration includes `-i` flag:
```json
"args": [
  "run",
  "-i",      // ← This flag is required
  "--rm",
  ...
]
```

### All Models Failed Error

```json
{
  "error": "All models failed to respond. Please try again.",
  "final_answer": null
}
```

**Common causes:**
- Invalid API keys (check all provider keys in `.env`)
- Insufficient credits in provider accounts
- Network connectivity issues
- Model identifiers incorrect in `config.py`
- Provider-specific issues:
  - **OpenAI**: Model name typos (use "gpt-5.1" not "gpt5.1")
  - **Anthropic**: Model version required (e.g., "claude-sonnet-4.5-20250929")
  - **Google**: API key format issues

**Debugging:**
Check server logs for "falling back to OpenRouter" messages to identify which providers are using fallback.

### Deliberation ID Not Found

```json
{
  "error": "Deliberation ID '...' not found or expired (TTL: 60 minutes)"
}
```

**Causes:**
- More than 60 minutes have passed since deliberation
- MCP server was restarted (cache is in-memory only)
- Invalid UUID provided

## Example Workflows

### Architectural Decision

```
User: Use llm-council to decide: Should I use microservices or monolith for this project?

Claude: [Calls llm_council_deliberate]
Returns: Chairman's synthesis with architectural recommendation

User: What did the individual models say?

Claude: [Calls llm_council_inspect with deliberation_id]
Shows: All 4 model responses, their peer evaluations, and aggregate rankings
```

### Code Review Decision

```
User: Use llm-council to conclude: Is this optimization worth the added complexity?

Claude: [Calls llm_council_deliberate]
Returns: Final recommendation from the council

User: Which model ranked highest in this deliberation?

Claude: [Calls llm_council_inspect, examines aggregate_rankings]
Shows: "openai/gpt-5.1 had average_rank 1.25 across 4 peer evaluations"
```

## Provider System Benefits

### Why Use Direct APIs?

**Performance:**
- **Lower Latency**: Direct connection eliminates OpenRouter routing overhead
- **Better Rate Limits**: Provider-specific limits instead of shared OpenRouter quotas
- **Native Features**: Access provider-specific capabilities (reasoning_details, etc.)

**Cost:**
- **Potentially Lower**: Some providers offer better pricing than OpenRouter markup
- **Separate Billing**: Easier to track costs per provider
- **Credit Control**: Individual budget limits per provider

**Reliability:**
- **Reduced Single Point of Failure**: Not dependent on OpenRouter availability
- **Gradual Migration**: Can move providers one at a time
- **Fallback Safety**: Automatic OpenRouter fallback if direct API fails

### Migration Strategy

**Phase 1: Add Direct Keys**
```bash
# Add one provider at a time
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-v1-...  # Keep as fallback
```

**Phase 2: Monitor Logs**
Check for "falling back to OpenRouter" messages to verify direct API usage.

**Phase 3: Expand Coverage**
Add more providers as you test:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

**Phase 4: Optional OpenRouter Removal**
Once all models use direct APIs, `OPENROUTER_API_KEY` becomes optional (only needed for xAI Grok or other unsupported providers).

## Best Practices

1. **Use for Complex Decisions**: LLM Council is best for decisions with multiple valid perspectives (architecture, trade-offs, design patterns)

2. **Inspect When Uncertain**: If the final answer seems surprising, use `llm_council_inspect` to understand the reasoning and peer rankings

3. **Consider Cost**: Each deliberation queries 4 models twice (Stages 1 & 2) plus chairman once (Stage 3), totaling 9 API calls. Use judiciously for important decisions.

4. **Cache Awareness**: Inspect detailed results within 60 minutes. After expiration, only the final answer in conversation history remains accessible.

## Technical Details

- **Transport**: stdio (standard MCP deployment)
- **Concurrency**: Stage 1 and Stage 2 use parallel async queries for speed
- **Error Handling**: Graceful degradation - continues with successful model responses if some fail
- **Security**: API key passed via environment variable, not exposed in tool responses

## Version Information

- MCP Protocol: 2025-06-18 specification
- FastMCP SDK: Uses official `mcp` Python package
- Tool Output: Structured JSON with defined schemas
- Python: Requires 3.10+ for async/await support
