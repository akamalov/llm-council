# CLAUDE.md - Technical Notes for LLM Council

This file contains technical details, architectural decisions, and important implementation notes for future development sessions.

## Project Overview

LLM Council is a 3-stage deliberation system where multiple LLMs collaboratively answer user questions. The key innovation is anonymized peer review in Stage 2, preventing models from playing favorites.

## Architecture

### Backend Structure (`backend/`)

**`config.py`**
- Contains `COUNCIL_MODELS` (list of model configurations with provider info)
- Contains `CHAIRMAN_MODEL` (model that synthesizes final answer)
- Uses environment variables for API keys:
  - `OPENAI_API_KEY`: Direct OpenAI access
  - `ANTHROPIC_API_KEY`: Direct Anthropic access
  - `GOOGLE_API_KEY`: Direct Google Gemini access
  - `OPENROUTER_API_KEY`: OpenRouter fallback/alternative
- Backend runs on **port 8001** (NOT 8000 - user had another app on 8000)

**`llm_client.py`** (NEW - replaces direct OpenRouter usage)
- `LLMClient`: Unified client supporting multiple providers
- `query_model()`: Single async model query to any provider
- `query_models_parallel()`: Parallel queries using `asyncio.gather()`
- Returns dict with 'content' and optional metadata
- **Automatic fallback**: If direct API key not available, falls back to OpenRouter
- Supported providers: `openai`, `anthropic`, `google`, `openrouter`

**`openrouter.py`** (LEGACY - kept for backward compatibility)
- Now mostly replaced by `llm_client.py`
- Still used as fallback when direct API keys unavailable

**`council.py`** - The Core Logic
- `stage1_collect_responses()`: Parallel queries to all council models
- `stage2_collect_rankings()`:
  - Anonymizes responses as "Response A, B, C, etc."
  - Creates `label_to_model` mapping for de-anonymization
  - Prompts models to evaluate and rank (with strict format requirements)
  - Returns tuple: (rankings_list, label_to_model_dict)
  - Each ranking includes both raw text and `parsed_ranking` list
- `stage3_synthesize_final()`: Chairman synthesizes from all responses + rankings
- `parse_ranking_from_text()`: Extracts "FINAL RANKING:" section, handles both numbered lists and plain format
- `calculate_aggregate_rankings()`: Computes average rank position across all peer evaluations

**`storage.py`**
- JSON-based conversation storage in `data/conversations/`
- Each conversation: `{id, created_at, messages[]}`
- Assistant messages contain: `{role, stage1, stage2, stage3}`
- Note: metadata (label_to_model, aggregate_rankings) is NOT persisted to storage, only returned via API

**`main.py`**
- FastAPI app with CORS enabled for localhost:5173 and localhost:3000
- POST `/api/conversations/{id}/message` returns metadata in addition to stages
- Metadata includes: label_to_model mapping and aggregate_rankings

### Frontend Structure (`frontend/src/`)

**`App.jsx`**
- Main orchestration: manages conversations list and current conversation
- Handles message sending and metadata storage
- Important: metadata is stored in the UI state for display but not persisted to backend JSON

**`components/ChatInterface.jsx`**
- Multiline textarea (3 rows, resizable)
- Enter to send, Shift+Enter for new line
- User messages wrapped in markdown-content class for padding

**`components/Stage1.jsx`**
- Tab view of individual model responses
- ReactMarkdown rendering with markdown-content wrapper

**`components/Stage2.jsx`**
- **Critical Feature**: Tab view showing RAW evaluation text from each model
- De-anonymization happens CLIENT-SIDE for display (models receive anonymous labels)
- Shows "Extracted Ranking" below each evaluation so users can validate parsing
- Aggregate rankings shown with average position and vote count
- Explanatory text clarifies that boldface model names are for readability only

**`components/Stage3.jsx`**
- Final synthesized answer from chairman
- Green-tinted background (#f0fff0) to highlight conclusion

**Styling (`*.css`)**
- Light mode theme (not dark mode)
- Primary color: #4a90e2 (blue)
- Global markdown styling in `index.css` with `.markdown-content` class
- 12px padding on all markdown content to prevent cluttered appearance

## Key Design Decisions

### Stage 2 Prompt Format
The Stage 2 prompt is very specific to ensure parseable output:
```
1. Evaluate each response individually first
2. Provide "FINAL RANKING:" header
3. Numbered list format: "1. Response C", "2. Response A", etc.
4. No additional text after ranking section
```

This strict format allows reliable parsing while still getting thoughtful evaluations.

### De-anonymization Strategy
- Models receive: "Response A", "Response B", etc.
- Backend creates mapping: `{"Response A": "openai/gpt-5.1", ...}`
- Frontend displays model names in **bold** for readability
- Users see explanation that original evaluation used anonymous labels
- This prevents bias while maintaining transparency

### Error Handling Philosophy
- Continue with successful responses if some models fail (graceful degradation)
- Never fail the entire request due to single model failure
- Log errors but don't expose to user unless all models fail

### UI/UX Transparency
- All raw outputs are inspectable via tabs
- Parsed rankings shown below raw text for validation
- Users can verify system's interpretation of model outputs
- This builds trust and allows debugging of edge cases

## Important Implementation Details

### Relative Imports
All backend modules use relative imports (e.g., `from .config import ...`) not absolute imports. This is critical for Python's module system to work correctly when running as `python -m backend.main`.

### Port Configuration
- Backend: 8001 (changed from 8000 to avoid conflict)
- Frontend: 5173 (Vite default)
- Update both `backend/main.py` and `frontend/src/api.js` if changing

### Markdown Rendering
All ReactMarkdown components must be wrapped in `<div className="markdown-content">` for proper spacing. This class is defined globally in `index.css`.

### Model Configuration
Models are configured in `backend/config.py` with explicit provider fields. Each model specifies:
- `model`: Provider-specific model identifier (e.g., "gpt-5.1", "claude-sonnet-4.5-20250929")
- `provider`: Which API to use ("openai", "anthropic", "google", "openrouter")
- `display_name`: User-facing name shown in UI (e.g., "openai/gpt-5.1")

Chairman can be same or different from council members. The current default is Gemini as chairman per user preference.

## Provider System

### Direct API Integration
The system now supports direct API calls to OpenAI, Anthropic, and Google, bypassing OpenRouter for lower latency and better rate limits.

**Provider-Specific Adaptations:**

**OpenAI** (`openai`)
- Endpoint: `/v1/chat/completions`
- Uses standard message format
- Supports reasoning_details

**Anthropic** (`anthropic`)
- Endpoint: `/v1/messages`
- System messages extracted to separate `system` parameter
- Max tokens required (default: 4096)
- Response in `content[0].text`

**Google Gemini** (`google`)
- Endpoint: `/v1beta/models/{model}:generateContent`
- Messages converted to `contents` with `parts` structure
- Role mapping: `assistant` → `model`
- System instruction in separate parameter
- API key in query string

**OpenRouter** (`openrouter`)
- Used as fallback when direct API keys unavailable
- Unified format for all models
- Single billing across providers

### Automatic Fallback Logic
If a direct provider API key is not configured:
1. System prints: "PROVIDER_API_KEY not found, falling back to OpenRouter"
2. Queries via OpenRouter using format: `provider/model`
3. Continues gracefully without user intervention

This allows gradual migration from OpenRouter to direct APIs without breaking existing setups.

## Common Gotchas

1. **Module Import Errors**: Always run backend as `python -m backend.main` from project root, not from backend directory
2. **CORS Issues**: Frontend must match allowed origins in `main.py` CORS middleware
3. **Ranking Parse Failures**: If models don't follow format, fallback regex extracts any "Response X" patterns in order
4. **Missing Metadata**: Metadata is ephemeral (not persisted), only available in API responses
5. **Provider API Keys**: If direct API keys missing, system falls back to OpenRouter silently. Check logs for "falling back to OpenRouter" messages.
6. **Model Identifiers**: Use native format ("gpt-5.1") not OpenRouter format ("openai/gpt-5.1") in model field. OpenRouter format only for `display_name`.

## Future Enhancement Ideas

- Configurable council/chairman via UI instead of config file
- Streaming responses instead of batch loading
- Export conversations to markdown/PDF
- Model performance analytics over time
- Custom ranking criteria (not just accuracy/insight)
- Support for reasoning models (o1, etc.) with special handling

## MCP Integration

LLM Council is available as an MCP (Model Context Protocol) server for use in Claude Code and other MCP-compatible clients.

**Documentation:**
- `MCP.md` - User guide (setup, usage, troubleshooting)
- `MCP_IMPLEMENTATION_GUIDE.md` - Developer guide (architecture, extending, testing)

**Quick Setup:**

Add to `.claude/mcp.json`:
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

**Two Tools Exposed:**

1. **`llm_council_deliberate(question)`** - Run complete 3-stage deliberation, returns final answer + deliberation_id
2. **`llm_council_inspect(deliberation_id)`** - Retrieve detailed breakdown of all stages

**Usage Pattern:**
```
User: "Use llm-council to decide: Should I refactor this module?"
→ Claude calls llm_council_deliberate
→ Returns chairman's synthesis + deliberation_id

User: "Show me what each model said"
→ Claude calls llm_council_inspect
→ Returns all stage1/stage2/stage3 details
```

**Technical Notes:**
- Deliberations cached in memory for 60 minutes
- Stateless design (no conversation IDs)
- Uses hardcoded models from `config.py`
- Returns structured JSON with all stage data
- Runs via stdio transport

## Testing Notes

Use `test_openrouter.py` to verify API connectivity and test different model identifiers before adding to council. The script tests both streaming and non-streaming modes.

## Data Flow Summary

```
User Query
    ↓
Stage 1: Parallel queries → [individual responses]
    ↓
Stage 2: Anonymize → Parallel ranking queries → [evaluations + parsed rankings]
    ↓
Aggregate Rankings Calculation → [sorted by avg position]
    ↓
Stage 3: Chairman synthesis with full context
    ↓
Return: {stage1, stage2, stage3, metadata}
    ↓
Frontend: Display with tabs + validation UI
```

The entire flow is async/parallel where possible to minimize latency.
