# Agent Notes

This project uses Pixi. Prefer Pixi for Python package installs, environment commands, and reusable project commands.

- Use `pixi add` or `pixi add --pypi` instead of calling `pip` directly.
- Run project commands through `pixi run` when practical.
- Add reusable commands as Pixi tasks in `pixi.toml`; use task arguments for paths or modes that should be configurable.
- Keep generated datasets and large archives out of Git unless the project explicitly chooses a large-data mechanism.
- Use `jaxtyping` annotations for tensor-facing APIs. Prefer explicit semantic dimension names such as `batch`, `n`, `buckets`, `channel`, `height`, and `width`, and include tensor dtype families such as `Float`, `Int`, or `UInt8` when practical.
- See `.github/skills/pixi-workflow/SKILL.md` for the local Pixi workflow skill.
