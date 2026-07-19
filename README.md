# inkling

A [Claude Code skill](https://code.claude.com/docs/en/skills) for setting up cheap specialized AI on a hobby budget (single-digit $/month), built around [Tinker](https://thinkingmachines.ai/tinker/) — the fine-tuning and sampling API from [Thinking Machines](https://thinkingmachines.ai) — plus local embeddings with pgvector for semantic search/RAG, and an optional one-time fine-tune exported and served locally via Ollama.

## Install

```bash
mkdir -p ~/.claude/skills/inkling
curl -o ~/.claude/skills/inkling/SKILL.md \
  https://raw.githubusercontent.com/zbcoding/inkling-skill/main/SKILL.md
```

Then invoke with `/inkling` in Claude Code.

## What it covers

- A cost ladder: free-tier LLM chain → local embeddings + pgvector → Tinker base-model sampling → one-time SFT, stopping at the first rung that meets accuracy needs
- Tinker SDK sampling (Thinking Machines base models like `thinkingmachines/Inkling`), billing levers (cached prefill), and model selection
- Idempotent pgvector migrations, e5 embedding conventions, backfill patterns
- RAG few-shot injection for tagging/translation/classification
- Training-data curation heuristics and verification habits
- LoRA export → GGUF → quantize → Ollama serving toolchain

## License

MIT
