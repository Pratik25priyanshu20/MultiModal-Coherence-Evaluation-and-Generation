# Preferences

User's workflow, coding, and communication preferences.

---

## Communication
- Prefers concise, direct responses
- Likes detailed technical exploration when asked to "go through" code
- Comfortable with parallel agent workflows for speed

## Coding Style
- Python-based project (research/ML codebase)
- Uses type hints and docstrings in source code
- Prefers editing existing files over creating new ones
- Settings centralized in `src/config/settings.py`

## Workflow
- Uses VS Code as IDE
- Git-based version control (current branch: clean-for-push)
- Jupyter notebooks for GPU training tasks
- Streamlit for interactive demos/dashboards
- Ollama for local LLM inference

## Tools & Environment
- macOS (Darwin 25.3.0)
- Shell: zsh
- Python with PyTorch, transformers, librosa, scipy stack
- Claude Code CLI for development assistance
- University GPU servers (compute.data-lab.site) for training — A6000/H200 nodes
- HuggingFace Spaces for deployments (demo + eval interface)

---

_Update this file as new preferences are discovered during sessions._

## Git / GitHub (2026-08-02)
- **NEVER push to GitHub** — the user ALWAYS pushes manually themselves
- Keep new deliverables in dedicated folders (e.g. cmsci-package/) so they do not mix with research code
