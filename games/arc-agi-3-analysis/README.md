# ARC-AGI-3 Agent Analysis

**Status:** Open for submissions

Analyze a GitHub repo with the AI of your choice. Post your output, track your usage, submit here.

---

## The Rules

- Run the prompt exactly as written — no added instructions, no modifications
- Use any model or inference platform you want (OpenAI, Anthropic, local, Qwen, homebrew — all valid)
- Submit as many times as you want with different models or methods
- Track your usage (screenshot before/after, or note token counts/cost)

---

## How to Play

1. Copy the prompt from [PROMPT.md](PROMPT.md) — run it verbatim, no modifications
2. Post your output as an X Article or GitHub URL
3. Note your usage (screenshot before/after, or token/cost estimate)
4. Submit here via PR with your model, method, link, and usage

The only rule: same prompt every time. Any model, any platform, as many runs as you want.

---

## How to Submit

1. Fork `40verse/GAiME`
2. Create `games/arc-agi-3-analysis/submissions/<your-handle>_<model>.md`
3. Fill in the required fields and paste your full output
4. Open a PR to `40verse/GAiME`

### Submission template

```markdown
**Handle:** your-handle
**Model:** claude-opus-4-6
**Method:** Claude Code with sub-agents and custom tools
**Link:** https://x.com/... or https://github.com/...
**Usage:** ~200k tokens / $3.20 (screenshot: link-to-screenshot)

---

[Your full output here]
```

---

## Review Process

1. A bot checks your PR automatically — wrong paths get closed, missing fields get flagged with a comment explaining what to fix
2. Once the check passes, a maintainer reviews and merges
3. On merge, your submission is automatically appended to the registry

Typical turnaround: within a day or two. Fix any bot feedback and push to the same branch — no need to open a new PR.

---

## Submissions

See [SUBMISSIONS.md](SUBMISSIONS.md) for the full registry.
