import os
from fnmatch import fnmatch
from pathlib import Path
import wandb


_REPO_ROOT = Path(__file__).resolve().parent.parent


def robomimic_datasets_root(default_root):
    if "ROBOMIMIC_DATASETS_PATH" in os.environ:
        return Path(os.environ["ROBOMIMIC_DATASETS_PATH"])
    fallback_root = _REPO_ROOT / "datasets" / "robomimic"
    if fallback_root.is_dir():
        return fallback_root
    default_root = Path(default_root)
    if default_root.is_absolute():
        return default_root
    return _REPO_ROOT / default_root


def _load_gitignore_patterns(gitignore_path):
    patterns = []
    for raw_line in gitignore_path.read_text().splitlines():
        line = raw_line.strip()
        if line == "":
            continue
        if line.startswith("#"):
            continue
        if line.startswith("!"):
            continue
        if line.startswith("./"):
            line = line[2:]
        anchored = line.startswith("/")
        if anchored:
            line = line[1:]
        directory = line.endswith("/")
        if directory:
            line = line[:-1]
        if line == "":
            continue
        patterns.append((line, anchored, directory))
    return patterns


def _build_gitignore_exclude_fn(repo_root):
    gitignore_path = Path(repo_root) / ".gitignore"
    patterns = _load_gitignore_patterns(gitignore_path)

    def _matches_pattern(rel_path, pattern, anchored, directory):
        if directory:
            if anchored:
                return rel_path.startswith(f"{pattern}/")
            if rel_path.startswith(f"{pattern}/"):
                return True
            return f"/{pattern}/" in f"/{rel_path}"
        if "/" not in pattern:
            if anchored:
                first_part = rel_path.split("/", 1)[0]
                return fnmatch(first_part, pattern)
            for part in rel_path.split("/"):
                if fnmatch(part, pattern):
                    return True
            return False
        if anchored:
            return fnmatch(rel_path, pattern)
        if fnmatch(rel_path, pattern):
            return True
        parts = rel_path.split("/")
        for start in range(1, len(parts)):
            if fnmatch("/".join(parts[start:]), pattern):
                return True
        return False

    def exclude_fn(path, _root):
        rel_path = os.path.relpath(path, repo_root)
        rel_posix = Path(rel_path).as_posix()
        for pattern, anchored, directory in patterns:
            if _matches_pattern(rel_posix, pattern, anchored, directory):
                return True
        return False

    return exclude_fn


_SOURCE_CODE_INCLUDE_ROOTS = ("rlpd", "configs")


def _build_source_code_include_fn(repo_root):
    include_roots = tuple(Path(root).as_posix().strip("/") for root in _SOURCE_CODE_INCLUDE_ROOTS)
    for include_root in include_roots:
        include_root_path = Path(repo_root) / include_root
        assert include_root_path.is_dir(), include_root_path

    def include_fn(path, _root):
        rel_path = Path(os.path.relpath(path, repo_root)).as_posix()
        for include_root in include_roots:
            if rel_path == include_root or rel_path.startswith(f"{include_root}/"):
                return True
        return False

    return include_fn


def _dedupe_config_overrides(argv):
    prefix = "--config"
    items = []
    config_tokens = None
    config_idx = None
    last_override_idx = {}
    i = 1
    while i < len(argv):
        arg = argv[i]
        kind = "other"
        key = None
        tokens = [arg]
        step = 1

        if arg == prefix:
            if i + 1 < len(argv) and argv[i + 1] != "":
                kind = "config"
                tokens = [arg, argv[i + 1]]
                step = 2
        elif arg.startswith(prefix + "="):
            if arg[len(prefix) + 1:] != "":
                kind = "config"
        elif arg.startswith(prefix + "."):
            tail = arg[len(prefix) + 1:]
            if tail != "":
                if "=" in tail:
                    key = tail.split("=", 1)[0]
                    if key != "":
                        kind = "override"
                    else:
                        key = None
                else:
                    kind = "override"
                    key = tail
                    if i + 1 < len(argv):
                        tokens = [arg, argv[i + 1]]
                        step = 2

        idx = len(items)
        items.append((kind, key, tokens))
        if kind == "config":
            config_tokens = tokens
            config_idx = idx
        elif kind == "override":
            last_override_idx[key] = idx
        i += step

    config_insert_idx = None
    if config_idx is not None:
        if len(last_override_idx) > 0:
            first_override_idx = min(last_override_idx.values())
            if config_idx > first_override_idx:
                config_insert_idx = first_override_idx
            else:
                config_insert_idx = config_idx
        else:
            config_insert_idx = config_idx

    out = [argv[0]]
    config_inserted = False
    for idx, (kind, key, tokens) in enumerate(items):
        if config_insert_idx is not None and not config_inserted and idx == config_insert_idx:
            out.extend(config_tokens)
            config_inserted = True
        if kind == "config":
            continue
        if kind == "override":
            if last_override_idx[key] != idx:
                continue
        out.extend(tokens)
    if config_tokens is not None and not config_inserted:
        out.extend(config_tokens)
    return out

class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row['step'] = step
        if self.file is None:
            self.file = open(self.path, 'w')
            if self.header is None:
                self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
                self.file.write(','.join(self.header) + '\n')
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        else:
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
