# The Architecture of Not Knowing: How ARC-AGI-3 Agents Learn Game Rules from Pixels Without Being Told

## Why This Repo Deserves a Deep Read

The [ARC-AGI-3-Agents](https://github.com/symbolica-ai/ARC-AGI-3-Agents) repository by Symbolica AI is not another LLM wrapper that calls `gpt-4` in a loop. It's a **competition-grade agent harness** — a framework where a dozen different agent architectures race to solve unknown puzzle games from raw pixel grids, without being told the rules.

The problem is deceptively hard: the agent receives a 64×64 integer grid (values 0–15) representing a bird's-eye view of a game level. It can take one of seven actions per turn (four directional moves, an "enter" action, a coordinate-click, and a reset). It doesn't know what walls are. It doesn't know what keys do. It has 80 actions and 25 energy units to figure it out and **win**. The codebase contains twelve distinct agent implementations — from pure random baselines to multi-LLM-call visual reasoning pipelines — all plugging into the same abstract `Agent` interface. The design decisions visible in this code are a masterclass in agent architecture trade-offs.

What follows are the key patterns I found by reading every source file, not the README.

---

## Insight 1: The "Two-Phase Turn" — Separate Observation from Action Selection

**The most important architectural decision in the codebase is splitting each turn into two distinct LLM calls: one to observe, one to act.**

In `agents/templates/llm_agents.py`, the `LLM` base class implements this via the `DO_OBSERVATION` flag. When `True` (the default), each turn proceeds in two phases:

1. **Observation phase**: The LLM receives the game state as a function response and is asked to reply with "a few sentences of plain-text strategy observation about the frame to inform your next action" (see `build_func_resp_prompt()`).
2. **Action phase**: A second LLM call receives the conversation including the observation and must return exactly one function call (game action).

The `FastLLM` subclass sets `DO_OBSERVATION = False` and skips the first call entirely. This isn't just a speed optimization — it's an explicit experiment testing whether chain-of-thought observation improves game performance.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Frame Data   │────▶│ Observe     │────▶│ Choose      │
│ (game state) │     │ (free-text) │     │ (tool call) │
└─────────────┘     └─────────────┘     └─────────────┘
      LLM.DO_OBSERVATION=True              Always runs
      Skipped if False
```

**Why this matters**: Many agent builders fuse reasoning and action into a single prompt. Splitting them forces the model to commit to an interpretation of the game state *before* it selects an action. This creates an explicit reasoning trace that's recorded, debuggable, and — crucially — feeds back into the sliding context window for future turns. The `FastLLM` variant exists precisely to measure whether this overhead pays off.

---

## Insight 2: Sliding-Window Message Management with Tool-Call Boundary Awareness

**The codebase solves context-window management with a FIFO message buffer that respects tool-call message boundaries — a detail most agent frameworks get wrong.**

In `LLM.push_message()` (`agents/templates/llm_agents.py`), messages are appended to a list capped at `MESSAGE_LIMIT` (default 10). When the limit is exceeded, old messages are dropped from the front. But there's a critical guard:

```python
def push_message(self, message):
    self.messages.append(message)
    if len(self.messages) > self.MESSAGE_LIMIT:
        self.messages = self.messages[-self.MESSAGE_LIMIT:]
    if self.MODEL_REQUIRES_TOOLS:
        # can't clip the message list between tool and tool_call
        while self.messages[0].get("role") == "tool":
            self.messages.pop(0)
    return self.messages
```

If the window boundary lands in the middle of a tool-call/tool-response pair (as happens with models that use the `tools` API format rather than the older `functions` format), the orphaned `tool` message is stripped. Without this, the LLM would see a tool response with no preceding tool call — causing API errors or hallucinated context.

**Why this matters**: The `MODEL_REQUIRES_TOOLS` flag (set `True` for models like `o4-mini` and `o3`) is doing double duty: it switches the API format *and* activates this boundary-aware trimming. This is a real-world lesson in how context window management differs between function-calling and tool-calling models. Most frameworks either don't trim at all (hitting token limits) or trim naively (breaking tool-call pairs). This pattern is directly reusable.

---

## Insight 3: The MultiModal Agent's "Three-LLM-Call" Architecture — Observe, Decide as Human, Translate to Machine

**The `MultiModalLLM` agent in `agents/templates/multimodal.py` uses three separate LLM calls per turn, with the middle call asking the model to describe actions in human language before a third call translates them to game commands.**

The architecture is:

1. **Analysis call**: Given previous action, expected outcome, actual visual result (PNG images + diff image), and a self-programming memory prompt — produce an analysis and update the memory.
2. **Human action call**: Given the current frame as PNG images and the updated memory — describe the next action as a human would: `"Click on the red square near the bottom left corner"`.
3. **Translation call**: Given the human description and the current frame image — output the exact `ACTION1`–`ACTION6` with coordinates.

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│ Analyze  │───▶│ Human Action │───▶│ Translate to │
│ Outcome  │    │ Description  │    │ Game Command │
│ + Diff   │    │ (natural lang)│   │ (ACTION1-6)  │
└──────────┘    └──────────────┘    └──────────────┘
  Updates          "Click on the       {"action":
  memory           red square..."       "ACTION3"}
```

The `image_diff()` function computes a pixel-level diff between the previous and current frame, highlighting changed pixels in red on a black background. This diff image is passed alongside the actual frames.

**The self-programming memory is the most remarkable part.** The `_memory_prompt` starts as a template with `{{human.inputs}}` placeholder and evolves through the game. After each analysis call, the LLM returns a response split by `---`: the part before is the analysis, the part after *replaces* the memory prompt going forward. The LLM is literally rewriting its own system instructions each turn:

```python
before, _, after = analysis_message.partition("---")
analysis = before.strip()
self._memory_prompt = after.strip()  # LLM rewrites its own memory
```

**Why this matters**: This is a concrete implementation of "LLM self-programming" — the agent's understanding of game rules, action logs, and strategy are stored in a prompt that the LLM itself modifies. It's like giving the agent a scratchpad it can read and rewrite each turn. The human-language intermediate step also acts as a form of chain-of-thought that decouples *intent* from *execution*, making the translation step simpler and more reliable.

---

## Insight 4: The LangGraph Thinking Agent — Persistent Observation Journal with Long-Term Memory

**The `langgraph_thinking` package implements a full agentic loop with persistent observations stored in a SQLite-backed `Store`, creating durable memory that survives across game sessions.**

In `agents/templates/langgraph_thinking/tools.py`, the agent has four tools: `act`, `think`, `observe`, and `delete_observation`. The `observe` tool writes to a LangGraph `Store`:

```python
@tool
def observe(observation: str) -> str:
    """Stores an observation about the game in your journal.
    These observations are long-lived and will persist between game sessions."""
    store = get_store()
    store.put(("observations"), uuid.uuid4(), observation)
    return f"Observation stored with ID: {id}"
```

The `think` tool, by contrast, only appends to an in-memory `thoughts` list in the agent state. This creates a two-tier memory system:

| Memory Type | Storage | Lifetime | Mechanism |
|---|---|---|---|
| Thoughts | In-memory list | Single decision cycle | `think()` tool → `state["thoughts"]` |
| Observations | SQLite Store | Across sessions | `observe()` tool → `store.put()` |

The `act` node in `nodes.py` regenerates the system prompt every iteration by loading all stored observations and current thoughts:

```python
observations = [
    Observation(id=item.key, observation=item.value)
    for item in store.search(("observations"), limit=100)
]
system_message = SystemMessage(
    content=build_system_prompt(observations, state["thoughts"])
)
```

The workflow in `agent.py` chains four nodes: `init` → `check_key` → `analyze_frame_delta` → `act`. The `analyze_frame_delta` node does pixel-level comparison between frames and uses an LLM to interpret what changed, appending the interpretation to the context. The `check_key` node uses structured output (`KeyCheck` schema) to determine if the agent's current key matches the door pattern.

**Why this matters**: Most agent memory implementations are either pure context-window (loses old info) or vector-store retrieval (lossy, unstructured). This design gives the agent explicit control over what to remember via the `observe` tool and what to forget via `delete_observation`. The agent decides what's worth persisting. The two-tier system (ephemeral thoughts vs. persistent observations) mirrors how humans distinguish working memory from long-term knowledge.

---

## Insight 5: The Swarm Orchestrator — Thread-Per-Game Parallelism with Shared Scorecard

**The `Swarm` class in `agents/swarm.py` runs one agent instance per game in separate threads, all sharing a single scorecard ID for scoring.**

The orchestration is straightforward but reveals a key design choice:

```python
def main(self) -> EnvironmentScorecard | None:
    self.card_id = self.open_scorecard()
    for i in range(len(self.GAMES)):
        g = self.GAMES[i % len(self.GAMES)]
        a = self.agent_class(
            card_id=self.card_id, game_id=g, ...
        )
        self.agents.append(a)
    for a in self.agents:
        self.threads.append(Thread(target=a.main, daemon=True))
    for t in self.threads:
        t.start()
    for t in self.threads:
        t.join()
    scorecard = self.close_scorecard(card_id)
```

Each agent gets its own `arc_env` (via `self._arc.make(g, scorecard_id=self.card_id)`), own thread, own recorder — but they share the scorecard. The `i % len(self.GAMES)` modulo indexing suggests the framework supports running more agents than games (multiple attempts per game).

The `Playback` subclass of `Agent` replays recorded `.recording.jsonl` files, and the `__init__.py` registration system auto-discovers these files and registers them as valid "agents":

```python
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback
```

**Why this matters**: The architecture cleanly separates *what the agent does* (the `Agent` subclass) from *how many run in parallel* (the `Swarm`). Adding a new agent means implementing two methods: `is_done()` and `choose_action()`. Everything else — threading, recording, scoring, cleanup — is inherited. The playback mechanism means every run is reproducible, which is essential for iterating on agent strategies.

---

## Insight 6: The ReasoningAgent's Hypothesis-Driven Exploration with Structured Tool Output

**The `ReasoningAgent` in `agents/templates/reasoning_agent.py` forces the LLM to output structured hypotheses alongside every action, using Pydantic-validated tool calls.**

The `ReasoningActionResponse` schema requires five fields for every action:

```python
class ReasoningActionResponse(BaseModel):
    name: Literal["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"]
    reason: str          # 10-2000 chars
    short_description: str  # 5-500 chars
    hypothesis: str      # Current hypothesis about game mechanics
    aggregated_findings: str  # Summary of discoveries so far
```

The agent doesn't just pick `ACTION3` — it must articulate *why*, what it thinks the game mechanics are, and a running summary of everything it's learned. This schema is passed to the LLM as tool parameters, so each action function (`ACTION1`, `ACTION2`, etc.) expects these structured fields.

The agent also generates richly annotated grid images with zone coordinates overlaid via PIL (`generate_grid_image_with_zone()`), dividing the 64×64 grid into 16×16 zones with gold borders and coordinate labels. Both the image and raw grid text are sent to the LLM, giving it dual modalities for spatial reasoning.

With `MAX_ACTIONS = 400` (5x the default 80) and `REASONING_EFFORT = "high"`, this agent trades speed and cost for depth of understanding. It maintains a full `history` of `ReasoningActionResponse` objects and a `screen_history` buffer (capped at 10 to prevent memory bloat).

**Why this matters**: Structured outputs aren't just for clean logging — they *change how the model reasons*. Forcing the LLM to articulate a hypothesis and aggregated findings means it can't just pattern-match: it must maintain a coherent theory of the game that evolves over time. The Pydantic schema with `min_length` constraints prevents degenerate outputs like single-word reasoning. This is a design pattern worth stealing for any agent that needs to build understanding over time.

---

## Lessons for Builders

1. **Split observation from action into separate LLM calls.** The `LLM` base class's `DO_OBSERVATION` pattern (in `llm_agents.py`) creates an explicit reasoning step that becomes part of the conversation history. Your agent's future decisions improve because past observations are in-context. The `FastLLM` variant proves this is a toggleable experiment, not a fixed requirement — measure whether the extra latency pays off for your domain.

2. **Manage your message buffer at tool-call boundaries, not just token counts.** The `push_message()` FIFO in `llm_agents.py` with its `MODEL_REQUIRES_TOOLS` guard prevents a class of silent failures where orphaned tool messages cause API errors. If you're building agents with sliding context windows over tool-calling models, you need this pattern.

3. **Give your agent two tiers of memory: ephemeral thoughts and persistent observations.** The `langgraph_thinking` module's `think()` vs. `observe()` tools (in `tools.py`) let the agent choose what to commit to long-term storage. Combined with `delete_observation()`, the agent curates its own knowledge base. This is more robust than stuffing everything into context or hoping retrieval will surface the right memory.

4. **Use human-language intermediates before machine-executable actions.** The `MultiModalLLM`'s three-call architecture (in `multimodal.py`) — analyze, describe as human, translate to command — decouples intent from execution. When your agent's action space is small but the observation space is complex (images, grids, spatial reasoning), this intermediate step dramatically improves reliability.

5. **Force structured hypotheses in tool call schemas, not just action names.** The `ReasoningAgent`'s `ReasoningActionResponse` Pydantic model (in `reasoning_agent.py`) requires the LLM to articulate *why* it's acting and *what it expects* with enforced length constraints. This isn't overhead — it's a forcing function that prevents the model from falling into reactive loops. Apply this whenever your agent needs to build cumulative understanding.

---

## What This Signals

This codebase is a snapshot of where agent design is heading: away from monolithic prompt-and-pray loops and toward **decomposed cognitive architectures** where observation, reasoning, memory management, and action selection are separate, testable, and independently improvable components. The twelve agent variants aren't just different prompts — they represent fundamentally different theories about how an AI should approach an unknown environment. The shared `Agent` interface and `Swarm` orchestrator make these theories directly comparable on the same benchmark.

The most telling detail is what's *not* in the code: there's no task-specific game knowledge baked into the base framework. The `GuidedLLM` variant hardcodes LockSmith rules in its prompt and uses `o3` with `REASONING_EFFORT = "high"` — but it exists as an explicit contrast to agents that must discover rules from scratch. The competition isn't just about solving puzzles; it's about solving the meta-problem of *how to build agents that learn to solve puzzles*. That's the problem worth studying.

---
## Analysis Report

### Reasoning Chain
1. Fetched the GitHub repository main page to map overall file structure and understand the project (fork of arcprize/ARC-AGI-3-Agents by symbolica-ai)
2. Fetched the `agents/` directory listing to identify core module files: `agent.py`, `swarm.py`, `recorder.py`, `tracing.py`, `__init__.py`
3. Fetched the `agents/templates/` directory listing to identify all agent implementations: 8 files + 1 subdirectory (`langgraph_thinking/`)
4. Fetched the `agents/templates/langgraph_thinking/` directory listing: 8 files forming a complete LangGraph-based agent package
5. Read `agents/swarm.py` — complete source code. Identified thread-per-game orchestration pattern, scorecard lifecycle, and the Arcade SDK integration
6. Read `agents/recorder.py` — complete source code. JSONL recording with UUID-based filenames, prefix parsing for replay
7. Read `agents/tracing.py` — complete source code. AgentOps integration with NoOp fallback pattern and decorator-based session tracing
8. Read `agents/__init__.py` — complete source code. Auto-discovery of Agent subclasses, recording file registration as Playback agents
9. Read `agents/templates/llm_agents.py` — complete source code (~400 lines). Core LLM agent with observation/action split, FIFO message buffer, tool-call boundary handling. Four subclasses: `ReasoningLLM`, `FastLLM`, `GuidedLLM`, `MyCustomLLM`
10. Read `agents/templates/multimodal.py` — complete source code (~450 lines). Three-LLM-call architecture, self-programming memory prompt, image diff, human-language intermediate actions
11. Read `agents/templates/reasoning_agent.py` — complete source code (~300 lines). Pydantic-structured hypothesis output, grid visualization with zone overlays, screen history management
12. Read `agents/templates/langgraph_random_agent.py` — complete source code. LangGraph StateGraph wrapper around random action selection
13. Read `agents/templates/langgraph_functional_agent.py` — summary (full code not returned by fetch). LangGraph functional API with dual rendering modes (image/text)
14. Read all 7 files in `agents/templates/langgraph_thinking/`: `__init__.py`, `agent.py`, `schema.py`, `llm.py`, `tools.py`, `nodes.py`, `prompts.py`, `vision.py`. Complete source for each
15. Read `SYMBOLICA_README.md` (from symbolica/arcgentica branch) — project overview mentioning orchestrator + specialized subagents
16. Read `pyproject.toml` — dependencies including `arc-agi`, `langgraph`, `openai`, `smolagents`, `PIL`, `pydantic`
17. Read `llms.txt` — structured project summary for LLM consumption
18. Fetched `agents/templates/random_agent.py` — summary of random baseline agent
19. Fetched `agents/templates/smolagents.py` — summary of SmolCodingAgent and SmolVisionAgent implementations
20. Synthesized all findings into the article, cross-referencing specific files, functions, and code patterns

### Files Processed
- Total files examined: 25
- Key files:
  - `agents/templates/llm_agents.py` — Core LLM agent architecture; observation/action split; FIFO message buffer with tool-call boundary handling; four variant subclasses
  - `agents/templates/multimodal.py` — Three-LLM-call architecture; self-programming memory; image diff computation; human-language intermediate actions
  - `agents/templates/reasoning_agent.py` — Hypothesis-driven exploration; Pydantic-structured tool outputs; grid visualization with zone overlays
  - `agents/templates/langgraph_thinking/nodes.py` — Four-node workflow (init, check_key, analyze_frame_delta, act); tool-based action loop with retry
  - `agents/templates/langgraph_thinking/tools.py` — Two-tier memory system (think vs observe); persistent SQLite-backed observation journal
  - `agents/templates/langgraph_thinking/prompts.py` — System prompt construction with injected observations and thoughts; frame delta analysis prompts
  - `agents/templates/langgraph_thinking/vision.py` — Frame rendering with grid overlay, coordinate labels, and object highlighting (player, door, rotator, key)
  - `agents/templates/langgraph_thinking/schema.py` — AgentState TypedDict; KeyCheck structured output; LLM enum
  - `agents/swarm.py` — Thread-per-game orchestration; shared scorecard lifecycle
  - `agents/recorder.py` — JSONL event recording with UUID filenames
  - `agents/tracing.py` — AgentOps integration with NoOp fallback and decorator tracing
  - `agents/__init__.py` — Agent auto-discovery and registration; recording file → Playback mapping
  - `agents/templates/langgraph_random_agent.py` — LangGraph StateGraph-based random agent
  - `agents/templates/langgraph_functional_agent.py` — LangGraph functional API agent with dual rendering
  - `agents/templates/random_agent.py` — Pure random baseline agent
  - `agents/templates/smolagents.py` — HuggingFace smolagents integration (coding + vision variants)
  - `agents/templates/langgraph_thinking/llm.py` — LLM factory (GPT-4.1)
  - `agents/templates/langgraph_thinking/agent.py` — LangGraphThinking agent class with workflow compilation
  - `agents/templates/langgraph_thinking/__init__.py` — Package export
  - `main.py` — CLI entry point with argument parsing, game fetching, signal handling
  - `pyproject.toml` — Dependencies and tool configuration
  - `SYMBOLICA_README.md` — Symbolica-specific project documentation
  - `llms.txt` — Structured project summary for LLM consumption
  - `README.md` — Main project README (fetched via GitHub page)
  - `tests/` directory structure — conftest.py, unit/ subdirectory

### Token Metrics
- Input tokens (estimated): ~85,000 (25 web fetches returning full source files + conversation context)
- Output tokens (estimated): ~8,000 (article + report)
- Total tokens: ~93,000

### Agents / Tools Used
- **WebFetch**: Used 22 times to fetch raw source files from GitHub (`raw.githubusercontent.com`) and directory listings from GitHub UI pages
- **Agent (general-purpose)**: Launched 1 background agent to explore repository via `gh` CLI (failed due to `gh` not being installed, but web fetches succeeded independently)
- **TodoWrite**: Used to track analysis progress through 6 task items
- **Write**: Used to write the final article to disk
- **Bash**: Used for git branch checkout

### Confidence Notes
- `agents/agent.py`: Full source code was not returned verbatim by the web fetch tool (got a summary instead). The abstract interface (`is_done()`, `choose_action()`, `MAX_ACTIONS=80`) was inferred from its subclasses which reference these methods/attributes extensively. The `main()` game loop, `append_frame()`, `cleanup()`, and `Playback` class behavior were confirmed through cross-references in `swarm.py`, `llm_agents.py`, and `recorder.py`.
- `agents/templates/langgraph_functional_agent.py`: Got a summary rather than full source. The dual rendering (image/text), LangSmith tracing, and tool-calling patterns were confirmed from the summary and are consistent with the other LangGraph implementations.
- `agents/templates/random_agent.py`: Got a summary rather than full source. Core pattern (random action selection, seed-based RNG) confirmed from the similar `LangGraphRandom` implementation which was obtained in full.
- `agents/templates/smolagents.py`: Got a summary rather than full source. SmolCodingAgent and SmolVisionAgent patterns inferred from summary — these wrap HuggingFace's `smolagents` library.
- The Symbolica-specific `arcgentica` agent mentioned in `SYMBOLICA_README.md` was not found in the `main` branch — it likely lives on the `symbolica/arcgentica` branch. The analysis focused on the `main` branch contents.
- Token estimates are approximate based on typical web fetch response sizes and output length.
---
