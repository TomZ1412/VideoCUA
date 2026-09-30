# Gemini Agents

This package contains Google GenAI SDK based Gemini vision agents for OSWorld.

## Agents

- `GeminiAgent`: Gemini computer-use agent using the same XML tool-call scaffold as the Qwen agent.

Preferred imports for new code:

```python
from mm_agents.gemini import GeminiAgent
```

## API Configuration

The agent uses the `google-genai` SDK.

```bash
export GEMINI_API_KEY=...
```

`GEMINI_AK` is also accepted for compatibility with older internal scripts.

Useful runtime knobs:

- `OSWORLD_MAX_RETRY_TIMES`
- `OSWORLD_GEMINI_TIMEOUT`
- `GEMINI_API_VERSION`

Screenshot inputs are sent without Qwen-specific resizing. Use `--media_resolution low|medium|high` to control Gemini's media processing detail level.

## Runner Example

```bash
python scripts/eval/run_multienv_gemini.py \
  --model gemini-3.1-p \
  --media_resolution high \
  --api_key "$GEMINI_API_KEY"
```
