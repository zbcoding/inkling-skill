---
name: inkling
description: Set up cheap specialized AI for a project using Tinker (Thinking Machines) base models, pgvector semantic search/RAG, and an optional one-time fine-tune served locally. Use when adding tagging, translation, classification, semantic search, or a budget custom model to any project.
---

# Inkling / Tinker specialized-model setup

Goal: specialized model capability on a hobby budget (single-digit $/month). The
ladder — stop at the first rung that meets accuracy needs:

1. **Free-tier LLM chain** for generation tasks (tag/translate/classify). Order
   providers free→paid; every provider is a client with the same
   `chat_completion(prompt, max_tokens, temperature)` interface so fallbacks are
   one call each.
2. **Local embeddings + pgvector** for search and RAG. No LLM involved in search.
3. **Base-model sampling via Tinker** as a paid fallback or quality upgrade.
4. **One-time SFT on Tinker → export → serve locally** only when prompting with
   RAG demonstrably fails, or when the prompt overhead (taxonomy/schema) makes
   CPU serving infeasible without baking it into weights.

## Tinker essentials

- SDK: `pip install tinker tinker-cookbook`; auth via `TINKER_API_KEY`
  (console: tinker-console.thinkingmachines.ai).
- Billing meters are per million tokens: prefill / sample / train. Cached
  prefill is heavily discounted — keep static prompt prefixes (taxonomy,
  schema, instructions) byte-identical across calls. Checkpoint storage is
  per GB/month (LoRA checkpoints = cents). Check current rates in the docs;
  they change.
- Sampling a base model (no training):

```python
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

model = "thinkingmachines/Inkling"  # or a cheaper base like openai/gpt-oss-20b
renderer = renderers.get_renderer(model_info.get_recommended_renderer_name(model),
                                  get_tokenizer(model))
client = tinker.ServiceClient().create_sampling_client(base_model=model)
# build_generation_prompt returns a ModelInput (already tokenized) — pass it
# straight to sample(); no extra tokenizer.encode() needed.
prompt = renderer.build_generation_prompt(
    [renderers.Message(role="user", content=text)])
resp = client.sample(prompt=prompt, num_samples=1,
                     sampling_params=tinker.SamplingParams(
                         max_tokens=1024, stop=renderer.get_stop_sequences()))
resp = resp.result()  # sample() returns a Future
message, _ = renderer.parse_response(resp.sequences[0].tokens)
```

- Renderers differ per model family (GPT-OSS uses the `gpt_oss`/Harmony
  renderer). Always resolve via `model_info`, never hardcode.
- Inkling has controllable reasoning effort (0.0–0.99), but the exact API
  surface for passing it (renderer kwarg vs SamplingParams vs model-level)
  is undocumented — verify in the Tinker docs/console before relying on it.
- Verify exact `base_model=` ID strings against the Tinker model catalog
  (tinker-docs.thinkingmachines.ai/tinker/models/) — mismatched IDs fail at
  the SDK level.
- Keep reasoning effort minimal for structured output tasks; effort scales
  sample-token spend directly.
- Inkling accepts vision input — usable for captioning product/note images into
  text that then feeds tagging and embeddings. Cheaper alternative: run the
  captioning once through a small local or free-tier VLM; store captions as a
  text column and treat them as ordinary source text thereafter. Validate
  caption quality on a sample first — bad captions poison downstream tagging
  and embeddings silently.

## pgvector setup (per project)

Migration template (idempotent, safe to re-run):

```sql
CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE items
    ADD COLUMN IF NOT EXISTS embedding vector(384),   -- dims must match the model
    ADD COLUMN IF NOT EXISTS embedded_at timestamptz;
CREATE INDEX IF NOT EXISTS idx_items_embedding
    ON items USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_items_needs_embedding
    ON items (id) WHERE embedding IS NULL;             -- cheap backfill targeting
```

- On Supabase, DDL usually needs the table owner (often `supabase_admin`),
  not `postgres`.
- Verify the real column names against `information_schema.columns` before
  writing anything; never trust remembered schema.
- Embedding model default: `intfloat/multilingual-e5-small` (384 dims,
  CPU-friendly, multilingual). e5 requires role prefixes: `passage: ` for
  stored documents, `query: ` for searches; normalize embeddings.
- Embed a composite: title + translated title + tag names + head of
  description (~1500 chars; the model truncates ~512 tokens anyway).
- Backfill script pattern: loop `WHERE embedding IS NULL AND deleted_at IS NULL
  LIMIT batch`, encode batch, `UPDATE ... SET embedding=%s::vector,
  embedded_at=now()`. NULL-targeting makes it idempotent and doubles as the
  catch-up job for new rows. With concurrent writers, add `FOR UPDATE SKIP
  LOCKED` to the select.

## RAG for generation tasks

Before calling the LLM on a new item, retrieve the K≈5 nearest already-labeled
rows (`ORDER BY embedding <=> :vec LIMIT 5`) and inject their labels/
translations as few-shot examples. This is the main quality lever before any
training.

## Training-data curation

Curate with a **view over the source table**, not copied columns — views stay
in sync and re-curating is a redefinition, not a re-copy. Filter heuristics
that earn their place:

- required fields non-empty;
- output identical to input is an echo-failure **only when the source is
  actually in the language being translated from** — check the source
  language, not just presence of CJK; same-language sources (e.g. Spanish →
  Spanish) legitimately map to themselves;
- length ratio bounds (¼×–4×) catch truncated and runaway outputs.

Always sample ~10 accepted AND ~10 rejected rows and read them before trusting
a filter — the rejected set is where filter bugs hide. Content-level errors
(wrong-but-fluent output) pass every heuristic; accept a small error rate for
SFT rather than hand-cleaning.

## When training happens

- Wait for a few thousand clean labeled rows; the free pipeline generates them
  as a side effect, so waiting is free and improves the dataset.
- Pick the base model by where it will be **served**: check the target box's
  free RAM first. CPU serving at Q4 needs roughly GB ≈ params(B) × 0.5–0.6
  (approximate; Q4_K_M ≈ 0.5); MoE models count **total** params for RAM but
  active params for speed — Inkling-class MoE won't fit a 32 GB box; use a
  20B-class base for local serving.
- SFT cost ≈ dataset tokens × epochs × train rate — typically single-digit
  dollars for a few-thousand-row dataset. Export the LoRA, merge, convert to
  GGUF, quantize, serve via Ollama; the provider-chain interface means the
  served model is just another client entry. The merge→GGUF→quantize steps
  are external to the Tinker SDK: merge with `peft` (or Tinker export
  utilities), convert with llama.cpp's convert script, quantize with
  llama.cpp `quantize`, then `ollama create` or `llama-server`.
- The fine-tune's biggest win on CPU is prompt deletion: taxonomy/schema baked
  into weights turns a ~10K-token prompt into just the item text.

## Verification habits

- Dry-run every view/filter as a SELECT count against the live DB before
  shipping the migration.
- After backfill, prove search works with one nearest-neighbor query on a
  known item, not just row counts.
- Never print DB credentials off the host; read them from the container env at
  use time.
