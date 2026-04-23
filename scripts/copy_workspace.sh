#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/copy_workspace.sh [--src=PATH] --dst=PATH [--max-file-size-mb=N] [--include-root=DIR]

Create a lightweight working copy for parallel agent edits.

Options:
  --src=PATH              Source repo root (default: this repo root)
  --dst=PATH              Destination directory (required)
  --max-file-size-mb=N    Skip files larger than N MB (default: 8)
  --include-root=DIR      Additional top-level directory to copy (repeatable)
  --dry-run               Show rsync actions without writing files
  --help                  Show this help message
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SRC="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC="${DEFAULT_SRC}"
DST=""
MAX_FILE_SIZE_MB="8"
DRY_RUN=0

EXTRA_INCLUDE_ROOTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src=*)
      SRC="${1#*=}"
      shift
      ;;
    --dst=*)
      DST="${1#*=}"
      shift
      ;;
    --max-file-size-mb=*)
      MAX_FILE_SIZE_MB="${1#*=}"
      shift
      ;;
    --include-root=*)
      EXTRA_INCLUDE_ROOTS+=("${1#*=}")
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "[copy_workspace] unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "${MAX_FILE_SIZE_MB}" =~ ^[0-9]+$ ]]; then
  echo "[copy_workspace] --max-file-size-mb must be a non-negative integer" >&2
  exit 2
fi

SRC="$(cd "${SRC}" && pwd)"
if [[ ! -d "${SRC}" ]]; then
  echo "[copy_workspace] source directory not found: ${SRC}" >&2
  exit 2
fi

if [[ -z "${DST}" ]]; then
  echo "[copy_workspace] --dst=PATH is required" >&2
  usage >&2
  exit 2
fi
mkdir -p "${DST}"
DST="$(cd "${DST}" && pwd)"

if [[ "${DST}" == "${SRC}" ]]; then
  echo "[copy_workspace] destination must differ from source" >&2
  exit 2
fi

COMMON_FILTER="$(mktemp)"
ROOT_FILE_LIST="$(mktemp)"
INCLUDE_ROOTS_FILE="$(mktemp)"
FILTER_FILE="$(mktemp)"
trap 'rm -f "${COMMON_FILTER}" "${ROOT_FILE_LIST}" "${INCLUDE_ROOTS_FILE}" "${FILTER_FILE}"' EXIT

cat > "${COMMON_FILTER}" <<'FILTERS'
- /.git/
- /.venv/
- /.uv/
- /.pytest_cache/
- /.ruff_cache/
- /__pycache__/
- /archive/
- /exp/
- /wandb/
- /logs/
- /tmp/
- /outputs/
- /slurm/
- /slurm_out/
- /slurm_auto/
- /node_modules/
- /dist/
- /build/
- /datasets_scripted/
- /results/
- /vis/
- /**/.git/
- /**/.venv/
- /**/.pytest_cache/
- /**/.ruff_cache/
- /**/__pycache__/
- /**/.ipynb_checkpoints/
- /**/*.pyc
- /**/*.pyo
FILTERS

INCLUDE_ROOTS=(
  "configs"
  "rlpd"
  "scripts"
  "assets"
)
for root in "${EXTRA_INCLUDE_ROOTS[@]}"; do
  INCLUDE_ROOTS+=("${root}")
done

if [[ -f "${SRC}/.gitmodules" ]]; then
  while IFS= read -r module_path; do
    module_path="${module_path##* = }"
    if [[ "${module_path}" != "" ]]; then
      INCLUDE_ROOTS+=("${module_path}")
    fi
  done < <(grep -E '^\s*path\s*=\s*' "${SRC}/.gitmodules")
fi

declare -A seen
for root in "${INCLUDE_ROOTS[@]}"; do
  if [[ "${root}" == "" ]]; then
    continue
  fi
  if [[ -n "${seen["${root}"]+x}" ]]; then
    continue
  fi
  seen["${root}"]=1
  if [[ -e "${SRC}/${root}" ]]; then
    echo "${root}" >> "${INCLUDE_ROOTS_FILE}"
  fi
done

find "${SRC}" -mindepth 1 -maxdepth 1 -type f \( \
  -name '*.py' -o -name '*.sh' -o -name '*.toml' -o -name '*.yaml' -o -name '*.yml' -o \
  -name '*.json' -o -name '*.md' -o -name '*.txt' -o -name '*.lock' -o -name '*.ini' -o \
  -name '.env' -o -name '.gitignore' -o -name '.python-version' \
\) -size -"${MAX_FILE_SIZE_MB}"M | sort > "${ROOT_FILE_LIST}"

while IFS= read -r root; do
  cat "${COMMON_FILTER}" > "${FILTER_FILE}"

  echo "[copy_workspace] copying root dir: ${root}"
  rsync_cmd=(
    rsync -a --prune-empty-dirs
    --max-size="${MAX_FILE_SIZE_MB}m"
    --filter="merge ${FILTER_FILE}"
    "${SRC}/${root}/"
    "${DST}/${root}/"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    rsync_cmd+=(--dry-run --itemize-changes)
  fi
  "${rsync_cmd[@]}"
done < "${INCLUDE_ROOTS_FILE}"

while IFS= read -r top_file; do
  rel_path="${top_file#${SRC}/}"
  if [[ "${rel_path}" == "" ]]; then
    continue
  fi
  echo "[copy_workspace] copying root file: ${rel_path}"
  rsync_cmd=(
    rsync -a
    --max-size="${MAX_FILE_SIZE_MB}m"
    "${top_file}"
    "${DST}/"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    rsync_cmd+=(--dry-run --itemize-changes)
  fi
  "${rsync_cmd[@]}"
done < "${ROOT_FILE_LIST}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[copy_workspace] dry run complete"
  exit 0
fi

FILE_COUNT="$(find "${DST}" -type f | wc -l | tr -d ' ')"
BYTE_COUNT="$(du -sb "${DST}" | awk '{print $1}')"

echo "[copy_workspace] done"
echo "[copy_workspace] source: ${SRC}"
echo "[copy_workspace] destination: ${DST}"
echo "[copy_workspace] files: ${FILE_COUNT}"
echo "[copy_workspace] size_bytes: ${BYTE_COUNT}"
