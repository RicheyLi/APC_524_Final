You are creating a final project for a Software Engineering course.

Push changes to this repo: https://github.com/RicheyLi/APC_524_Final

Create documentation as you go, use Pytest for test-driven development for automated testing of the repository. Establish version control with git, use cookie cutter to initiate project and use logical package structures to seperate files. Use a virtual environment for package management. Use any other software engineering best practices as you build.


## Project-specific guidance

- Read `docs/plan.md` for the work-package breakdown and `CONTRIBUTING.md` for the rules
  every work package must follow. Implement exactly one work package per branch.
- Package import name is `wildfire_sim` (under `src/`). CLI entry point is `wildfire-sim`.
- Contracts you must not change without sign-off: `wildfire_sim.types`, `wildfire_sim.grid`,
  `wildfire_sim.spread.base`, `wildfire_sim.config`.
- Tests must be offline and deterministic; use the fixtures in `tests/conftest.py`.
- Quality gate before every push: `just check` (ruff format, ruff lint, ty, pytest).
- Primary Python is 3.12. Do not use `--python=3.14`.
