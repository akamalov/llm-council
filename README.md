# LLM Council

![llmcouncil](header.jpg)

The idea of this repo is that instead of asking a question to your favorite LLM provider (e.g. OpenAI GPT 5.1, Google Gemini 3.0 Pro, Anthropic Claude Sonnet 4.5, xAI Grok 4, eg.c), you can group them into your "LLM Council". This repo is a simple, local web app that essentially looks like ChatGPT except it uses OpenRouter to send your query to multiple LLMs, it then asks them to review and rank each other's work, and finally a Chairman LLM produces the final response.

In a bit more detail, here is what happens when you submit a query:

1. **Stage 1: First opinions**. The user query is given to all LLMs individually, and the responses are collected. The individual responses are shown in a "tab view", so that the user can inspect them all one by one.
2. **Stage 2: Review**. Each individual LLM is given the responses of the other LLMs. Under the hood, the LLM identities are anonymized so that the LLM can't play favorites when judging their outputs. The LLM is asked to rank them in accuracy and insight.
3. **Stage 3: Final response**. The designated Chairman of the LLM Council takes all of the model's responses and compiles them into a single final answer that is presented to the user.

## Vibe Code Alert

This project was 99% vibe coded as a fun Saturday hack because I wanted to explore and evaluate a number of LLMs side by side in the process of [reading books together with LLMs](https://x.com/karpathy/status/1990577951671509438). It's nice and useful to see multiple responses side by side, and also the cross-opinions of all LLMs on each other's outputs. I'm not going to support it in any way, it's provided here as is for other people's inspiration and I don't intend to improve it. Code is ephemeral now and libraries are over, ask your LLM to change it in whatever way you like.

## Setup

### 1. Install Dependencies

The project uses [uv](https://docs.astral.sh/uv/) for project management.

**Backend:**
```bash
uv sync
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

### 2. Configure API Keys

Create a `.env` file in the project root. You have two options:

**Option A: Direct Provider APIs (Recommended)**
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=sk-or-v1-...  # Optional fallback for models without direct keys
```

**Option B: OpenRouter Only**
```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

Get your API keys:
- OpenAI: [platform.openai.com](https://platform.openai.com/)
- Anthropic: [console.anthropic.com](https://console.anthropic.com/)
- Google Gemini: [ai.google.dev](https://ai.google.dev/)
- OpenRouter: [openrouter.ai](https://openrouter.ai/) (fallback for models without direct API keys)

**Benefits of Direct APIs:**
- Lower latency (direct connection)
- Better rate limits
- Native model access
- Automatic fallback to OpenRouter if keys not provided

### 3. Configure Models (Optional)

Edit `backend/config.py` to customize the council. Each model specifies its provider:

```python
COUNCIL_MODELS = [
    {
        "model": "gpt-5.1",
        "provider": "openai",  # Uses OPENAI_API_KEY, falls back to OpenRouter
        "display_name": "openai/gpt-5.1"
    },
    {
        "model": "gemini-3-pro-preview",
        "provider": "google",  # Uses GEMINI_API_KEY, falls back to OpenRouter
        "display_name": "google/gemini-3-pro-preview"
    },
    {
        "model": "claude-sonnet-4.5-20250929",
        "provider": "anthropic",  # Uses ANTHROPIC_API_KEY, falls back to OpenRouter
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

**Provider Options:** `openai`, `anthropic`, `google`, `openrouter`

## Running the Application

**Option 1: Use the start script**
```bash
./start.sh
```

**Option 2: Run manually**

Terminal 1 (Backend):
```bash
uv run python -m backend.main
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

## Tech Stack

- **Backend:** FastAPI (Python 3.10+), async httpx, OpenRouter API
- **Frontend:** React + Vite, react-markdown for rendering
- **Storage:** JSON files in `data/conversations/`
- **Package Management:** uv for Python, npm for JavaScript

## MCP Integration

LLM Council is available as an **MCP (Model Context Protocol) server** for Claude Code and other MCP clients. This allows you to invoke the council directly from within your AI assistant:

```
User: "Use llm-council to decide: Should I use microservices or monolith?"
Claude: [Invokes council via MCP, returns synthesized decision]
```

**Quick Setup:** See `MCP.md` for full documentation.

**Two MCP Tools:**
- `llm_council_deliberate(question)` - Complete 3-stage deliberation
- `llm_council_inspect(deliberation_id)` - Detailed stage breakdown

**Location Independent:** Works from any directory - you can use llm-council from any project without being in the llm-council directory.

This enables AI assistants to leverage multi-model consensus for complex architectural decisions, code reviews, and technical trade-offs.
