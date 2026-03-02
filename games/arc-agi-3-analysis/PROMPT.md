# ARC-AGI-3 Agent Analysis Prompt

Run this prompt verbatim. Do not modify or add instructions.

---

## Task

Analyze the GitHub repository at **https://github.com/symbolica-ai/ARC-AGI-3-Agents** and produce two deliverables:

1. An **X Article** — long-form, publication-ready — surfacing the key insights and lessons from the agent design patterns, methods, and architecture used in this codebase.
2. A **self-report** documenting exactly how you performed the analysis.

---

## Part 1: X Article

Write a long-form X Article on the insights and lessons extracted from this codebase. X Articles support full Markdown: headers, bold, code blocks, lists, and ASCII diagrams. Use them. Write for an AI/ML builder audience — the kind of person who builds agents and wants to learn from production-quality implementations.

**Article structure:**

### Title
A specific, compelling title. Not "Lessons from ARC-AGI-3" — something that names the actual insight.

### Hook (1–2 paragraphs)
Why this repo is worth reading. What problem it solves, and what makes the approach distinctive. Ground it in the code, not the README summary.

### Key Insights (4–6 sections)
Each section covers one distinct insight with:
- A bold insight headline
- 2–4 paragraphs of explanation, grounded in specific code evidence (file name, function, pattern)
- Where relevant: a code snippet, ASCII diagram, or comparison table illustrating the point

**Insight quality bar:**
- Every claim must be traceable to a specific file or pattern you verified exists
- Prefer novel or counterintuitive observations over obvious ones
- Name the design decisions and trade-offs the authors made — not just what the code does, but why it was built that way

### Lessons for Builders (1 section)
3–5 concrete takeaways that an AI/ML engineer can apply directly to their own agent implementations. Be specific — not "use modular design" but "separate your scoring loop from your generation loop as seen in `X` because..."

### Closing
1 paragraph. What this repo signals about the direction of agent design.

---

## Part 2: Analysis Self-Report

Append this block after the article. Do not mix it with the article content.

```
---
## Analysis Report

### Reasoning Chain
1. [First action — e.g., fetched repo root to map file structure]
2. [Second action — e.g., read README to identify entry points]
3. [Continue step-by-step]
...

### Files Processed
- Total files examined: N
- Key files:
  - path/to/file.py — [why it mattered]
  - ...

### Token Metrics
- Input tokens (estimated): N
- Output tokens (estimated): N
- Total tokens: N

### Agents / Tools Used
- [Tool or agent]: [what it was used for]
- ...

### Confidence Notes
- [Anything uncertain, inferred, or inaccessible]
---
```

---

## Constraints

- Do not summarize the README alone — read the actual source files
- Do not fabricate file names, functions, or patterns; only reference what you verified
- If you cannot access the repository, state that clearly rather than inferring content
- The article and the self-report are separate — do not blend them
