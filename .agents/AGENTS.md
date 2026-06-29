# Agent Guidelines

## Critical Rules
- **Do not execute any Git commands** (e.g., `git commit`, `git push`, `git stash`, etc.) unless explicitly instructed to do so by the user.
- **Code Comments**: Always add comments to explain any complex logic.
- **Lint Vefify**: 
  - Remove temporary files after use
  - Remove unused imports
  - Use import instead of fully qualified names.
  - Check for common lint issues