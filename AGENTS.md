## Repo Specific Instructions

When making code edits in your folder, consider using `scripts/copy_workspace.sh --dst=...` to create a copy of the code.

The following dirs are not copied by default:

- Heavy/runtime dirs: `.git`, `.venv`, `archive`, `exp`, `wandb`, `logs`, caches
- Files larger than `8MB` (override with `--max-file-size-mb`)

Example usage:

```bash
# Copy into an explicit destination path
bash scripts/copy_workspace.sh --dst="$(pwd)/archive/.../<HEX_FOLDER_NAME>"

# Include PolaRiS-Hub and allow larger files
bash scripts/copy_workspace.sh --dst="$(pwd)/archive/.../<HEX_FOLDER_NAME>" --include-root=PolaRiS/PolaRiS-Hub --max-file-size-mb=64
```