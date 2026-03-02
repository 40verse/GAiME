# GAiME
A place for random AI games. Each game lives in its own folder. Anyone can submit runs.

---

## How to Submit

1. **Fork** this repository
2. Navigate to the target game's folder (e.g., `games/arc-agi-3-analysis/`)
3. Add your submission file inside `submissions/` — name it `<your-handle>_<model>.md` (e.g., `alice_opus-4-6.md`)
4. **Open a Pull Request** back to `40verse/GAiME` targeting the game's folder

That's it. No special tooling required.

---

## Submission File Format

Each submission is a Markdown file inside the game's `submissions/` folder.

```
games/<game-name>/submissions/<handle>_<model>.md
```

Minimum required fields at the top of the file:

```markdown
**Handle:** your-handle
**Model:** claude-opus-4-6
**Method:** Claude Code with sub-agents and custom tools
**Link:** https://x.com/... or https://github.com/...
**Usage:** screenshot or description of token/cost usage
```

Followed by your full output below.

---

## Game Folders

| Game | Description |
|------|-------------|
| [arc-agi-3-analysis](games/arc-agi-3-analysis/) | Analyze a GitHub repo and publish findings to X or GitHub |

---

## Adding a New Game

Want to host a new game? Fork, create `games/<your-game-name>/` with a `README.md` describing the rules and prompt, and PR it in. Keep it simple — one prompt, one results format.
