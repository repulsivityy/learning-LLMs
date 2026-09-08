# AI & Agentic Workflows — Hands-On Lesson Plan

**Goal:** Go from first-principles transformer mechanics to building, fine-tuning, instrumenting, and adversarial stress-testing a multi-agent runtime harness. Practitioner depth — enough to build real things and hold your own in a conversation with researchers, not academic-level theory. No fixed timeline: work in bursts, pick up wherever you left off.

**How we'll work through this:** step by step, module by module, with Claude as tutor, one chat session per module for fresh context. Each module below has a normal technical explanation plus a plain-English (**ELI5**) line so nothing assumes prior knowledge. At the end of each module, expect 3-5 checkpoint questions to solidify what you learned before moving on.

---

## Working With This Plan (Claude Project Setup)

Recommended: run this as a **Claude Project**, one chat per module.

1. Create a Project (e.g. "AI & Agentic Workflows")
2. Upload this file as Project knowledge
3. Paste the instructions below into the Project's custom instructions
4. Start a new chat per module, named after the module (e.g. "Module 0 — Mechanics")
5. At the start of each chat, say which module you're starting or resuming
6. Tick off the Progress Checklist at the bottom as you finish each module — re-upload the file, or just tell Claude "I finished Module X" at the start of the next chat

**Suggested Project custom instructions (paste as-is):**
> You are my hands-on AI/ML tutor, working through the attached lesson plan one module per chat. Assume I'm new to whichever concept comes up — explain normally, plus a plain-English (ELI5) summary. Teach hands-on and step by step, don't dump everything at once. At the end of the module, ask me 3-5 checkpoint questions to check my understanding before considering it done.

---

## Environment, Portability & Tooling (Cloud-first — any device)

* **Primary approach: cloud notebook/IDE, not local setup.** Fully solves portability — identical experience on Windows, Mac, Linux, or a borrowed laptop, no local install required.
* **Code:** Single GitHub repo (`learning-LLMs`), `/notebooks` for exploration, `/src` for reusable code. The repo is the only thing that needs to survive between sessions.
* **Notebook modules (0-4, 6):** Google Colab. Free GPU tier covers most of it; step up to Colab Pro or RunPod/Lambda only if Module 1 (pretraining) or the heavier parts of Module 6 (multi-epoch DPO/PPO, larger LoRA) need more.
  * Each session: clone the repo into the Colab runtime → install deps (`uv`/`pip`) in a cell → work in `notebooks/` → commit and push back to GitHub before closing. Colab runtimes are ephemeral and reset — nothing survives unless it's in Git or Drive.
  * Mount Google Drive for large files that shouldn't live in Git (datasets, checkpoints).
* **Script modules (5, 7, 8, 9):** notebooks get clunky once you're building a CLI agent/harness. Use **GitHub Codespaces** instead — full VS Code + terminal in the browser, git-native, no local setup, same portability benefit.
* **Local fallback (optional):** if you have a machine set up for it on a given day, WSL2 on Windows / native on Mac & Linux still works — just no longer required.
* **Secrets:** `.env` via `python-dotenv`, never committed. Checkpoints go to external storage (Hugging Face Hub, Drive, or object storage) — not Git.

---

## Module 0 — Mechanics of a Language Model (CPU/MPS, Jupyter)
**Why:** Understand tokenization, attention, and inference cost before treating models as black boxes.

**ELI5:** Before you can teach a robot to talk, you need to see how it chops sentences into pieces and decides which earlier words matter most when guessing the next one.

- Build a BPE tokenizer from scratch *(a way of chopping text into small reusable chunks, like Lego bricks, instead of whole words)*
- Implement core transformer pieces yourself: attention *(deciding which earlier words matter most)*, multi-head attention *(doing that several times at once, each "head" looking for a different pattern)*, residual stream, layernorm *(keeps the numbers from growing too big or small as they flow through)*, feed-forward
- Implement a naive KV-cache *(remembering work you already did, instead of redoing it for every new word)* — see the difference between naive and cached generation
- **Deliverable:** a small model trained on toy text, generating token-by-token with attention you can inspect

**Eval:** loss trending down, visible perplexity improvement across training

---

## Module 1 — Train a Real Small Language Model (Cloud GPU, Jupyter)
**Why:** Real compute budgets, real training stability issues.

**ELI5:** You're baking your own tiny robot brain from scratch — pick a recipe (architecture), feed it stories (data), and watch it slowly get better at finishing sentences.

- Architecture: nanoGPT/Llama-style decoder — RoPE *(tells the model how far apart words are)*, SwiGLU *(a layer shape that tends to train better)*, RMSNorm *(a lighter version of the number-scaling trick from Module 0)*
- Dataset: small, clean, curated (TinyStories or a Markdown/docs subset) — curation is part of the lesson
- Cosine warmup *(start the learning speed low, ramp up, then ease back down — like warming up before a run)*, gradient clipping *(cap how big one update can be, so one weird example can't wreck the model)*, mixed precision *(smaller numbers = faster and lighter on memory, with barely any accuracy cost)*, checkpointing
- **Deliverable:** a trained SLM checkpoint (<100M params) that produces coherent paragraphs

**Eval:** perplexity *(a score for how "surprised" the model is by real text — lower is better)* on held-out data, tokens/sec and memory logged

---

## Module 2 — Prompt Engineering & Structured Output (Local, Jupyter)
**Why:** Get reliable behavior out of a model before you add external context or tools.

**ELI5:** Learning the magic words that make the robot answer the way you want — and how to force it to always answer in a format you can actually use, instead of just hoping.

- Prompt patterns: role/task/constraint framing, few-shot examples *(showing a couple of examples before asking your real question)*, chain-of-thought *(asking it to show its reasoning step by step instead of jumping to an answer)*
- Context window limits and attention degradation on long inputs
- Grammar-constrained decoding: enforce JSON/schema compliance at the sampling level *(physically blocking the model from breaking your format, instead of just asking nicely)*
- **Deliverable:** a small benchmark script measuring schema-compliance rate across prompt strategies and temperatures

**Eval:** schema validation pass rate; best strategy becomes your baseline going forward

---

## Module 3 — Embeddings & RAG (Local, Jupyter)
**Why:** Understand retrieval, not just call a vector DB.

**ELI5:** Teaching the robot to look things up in a library before answering, instead of only using what it memorized — and understanding how that "library search" actually works.

- Embeddings: bi-encoders *(turns text into a point in space; similar meaning = nearby points)* vs. cross-encoders *(looks at two texts together, slower but more accurate)*, contrastive training *(training with "these are similar" and "these are different" pairs)*, distance metrics, visualize clusters (t-SNE/UMAP — *ways to squash high-dimensional points down to 2D so a human can actually look at them*)
- Chunking strategy: semantic chunking, document structure, token-aware windows
- Vector DB (FAISS/Chroma/Qdrant local)
- Full loop: query → retrieve → inject context → generate (using your SLM)
- Deliberately surface failure modes: irrelevant top-k, lost-in-the-middle *(models tend to ignore stuff buried in the middle of a long prompt)*, semantic drift

**Eval:** Precision@k, Recall@k, MRR *(different ways of scoring "did search return the right documents, and how high up")*, factual grounding on a small QA set

---

## Module 4 — Frontier Models & Hybrid Retrieval (Local/API, Jupyter)
**Why:** Measure the real gap between your SLM and a frontier model, and improve retrieval quality.

**ELI5:** Comparing your homemade robot brain to a much bigger, professionally-built one — and giving your search system a second pair of eyes so it misses less.

- Swap in a frontier API (Claude/Gemini) on the same RAG setup
- Hybrid search: dense retrieval + BM25 keyword search *(old-school keyword search, like a smarter Ctrl+F — kept around because it catches things semantic search misses)*, combined via reciprocal rank fusion *(a simple way to merge two ranked lists into one)*
- Add a reranking pass *(a second, more careful pass that re-orders your first results)* before final generation
- **Deliverable:** comparison table — quality, latency, cost — SLM vs. frontier, with and without hybrid retrieval

**Eval:** LLM-as-judge *(using another AI to score/compare two answers instead of doing it by hand)* pairwise scoring + latency/cost curves

---

## Module 5 — Tool Use (Transition to Scripts)
**Why:** Turn generation into action.

**ELI5:** Giving the robot hands — letting it use a calculator, search the web, or run code, instead of only being able to talk.

- Function calling: schema declarations, parsing, error recovery when the model gets it wrong *(letting the model output a structured "call this tool with these inputs" request instead of just text)*
- Build a ReAct-style loop yourself, no framework *(the model alternates between reasoning — "what should I do next" — and acting, looping until done)*
- Tool suite: calculator, file reader, one API client, sandboxed code execution
- Safety: isolate execution (Docker or restricted subprocess), timeouts, output size limits

**Eval:** tool selection accuracy, schema adherence, recovery from failed tool calls

---

## Module 6 — Fine-Tuning Ladder (Cloud GPU/MPS)
**Why:** Adapt a model's behavior instead of relying on prompting alone.

**ELI5:** Teaching your robot new habits three different ways: showing it lots of examples, telling it "this answer beats that one," or rewarding it like a pet for good behavior.

*Uses an existing small open pretrained model (Qwen 0.5B–1.5B, Llama 3.2 1B) — your Module 1 SLM is too undertrained for this to teach anything.*

- **SFT** — supervised fine-tuning: training directly on examples of the exact behavior you want
- **LoRA/QLoRA** — fine-tune by training a small add-on instead of the whole model — much cheaper, single consumer GPU
- **DPO** — training directly on "this answer is better than that one" pairs
- **RLHF (small hands-on task)** — using TRL's PPO trainer + LoRA, run a toy sentiment-steering task with an off-the-shelf sentiment classifier as the reward *(training with a reward score instead of direct examples — more like training a dog with treats than handing it a rulebook)*, for a small number of steps, not to convergence. Watch the reward curve go up and the KL-divergence *(how far the model has drifted from where it started, kept in check so it doesn't go off the rails chasing reward)* stay bounded — and feel directly why PPO (4 models in memory at once, fussier to stabilize) is heavier than DPO for a similar result

**Eval:** before/after task performance for SFT/LoRA/DPO; reward + KL curves and side-by-side samples for the RLHF task, compared against the DPO run

---

## Module 7 — Agent Harness & Observability (Scripts)
**Why:** Turn models into a scalable, observable system.

**ELI5:** Building the robot's nervous system — the part that decides what to do next, remembers what already happened, and keeps a diary of everything it did.

- Execution graph *(a flowchart the agent follows: this step, then that step, with branches for different situations)*: nodes, edges, conditional branches, state persisted across turns
- **Model router:** a clean interface/adapter pattern — a middleman that lets your code say "generate text" without caring which actual model does the work. Plug in whichever backend runs on your current machine (Ollama/llama.cpp anywhere, vLLM as an optional adapter when you're on Linux/WSL2/cloud), plus frontier APIs. The lesson is the interface design, not any specific product.
- Resilience: retries/backoff, fallback models, context truncation, loop-termination guards
- Observability: structured tracing — logging every step in detail so you can see exactly what the agent did and why, after the fact
- **Deliverable:** a runtime engine with a clean programmatic API, swappable backend, and full call tracing

**Eval:** automated regression suite across a set of deterministic test scenarios

---

## Module 8 — Security-Testing & Red-Teaming Your Agent (Scripts)
**Why:** You built a harness with tools and RAG — now attack it.

**ELI5:** Trying to trick your own robot the way a scammer would, so you find the weak spots before someone else does.

- Direct prompt injection / jailbreaking *(overriding the system prompt, bypassing guardrails)*
- Indirect prompt injection: planting instructions inside RAG documents or tool outputs *(hiding instructions in text the model reads, tricking it into doing something it shouldn't)*
- Tool-use exploitation: parameter manipulation, unauthorized access, unintended state changes
- Data exfiltration paths: can the agent be tricked into leaking context through outputs or tool calls?
- Defenses: input/output filtering, tighter tool permission scoping, a second lightweight LLM check on risky outputs
- Re-run your Module 7 eval suite before/after hardening

**Eval:** Attack Success Rate *(what percentage of your attack attempts actually worked)* before vs. after hardening

---

## Module 9 (Stretch) — Multi-Agent Orchestration (Scripts)
**Why:** Natural extension of the harness, builds on your existing CrewAI/work experience.

**ELI5:** Instead of one robot doing everything, you build a small team of robots with different jobs who talk to each other to get things done.

- Topologies: hierarchical *(one boss, several workers)*, sequential *(an assembly line)*, peer-to-peer *(agents talking as equals)*
- Shared state/blackboard memory *(a shared notebook all the agents can read and write to)*, delegation, conflict resolution between agents
- Compare: one generalist agent vs. several specialists on the same task set
- Bonus: re-run your Module 8 red-team suite across agent trust boundaries — does coordination open new attack surface?

**Eval:** task success rate and efficiency, multi-agent vs. single-agent harness

---

## Progress Checklist
No calendar — just tick these off as you go, in bursts whenever you have time.

- [ ] Module 0 — Mechanics (tokenizer, attention, KV-cache from scratch)
- [ ] Module 1 — Train your SLM on GPU
- [ ] Module 2 — Prompt engineering + structured output
- [ ] Module 3 — Embeddings + RAG
- [ ] Module 4 — Frontier model comparison + hybrid retrieval
- [ ] Module 5 — Tool use
- [ ] Module 6 — Fine-tuning ladder (SFT → LoRA → DPO → small RLHF task)
- [ ] Module 7 — Harness + observability
- [ ] Module 8 — Security testing / red-teaming
- [ ] Module 9 (stretch) — Multi-agent orchestration

---

**One-liner:** you build the engine (0-1), learn to steer it reliably (2), give it a memory (3), see how a bigger engine compares (4), give it hands (5), teach it new tricks including the hard way researchers do it (6), give it a proper steering wheel that works with any engine (7), try to steal the car yourself before anyone else can (8), then build a whole convoy (9).
