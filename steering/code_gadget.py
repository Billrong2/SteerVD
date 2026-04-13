from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import time
from fnmatch import fnmatchcase
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .joern_slice import (
    DEFAULT_CACHE_DIR as DEFAULT_JOERN_CACHE_DIR,
    DEFAULT_JOERN_CLI_DIR,
    extract_joern_call_rows,
    extract_joern_external_call_argument_flows,
    extract_joern_method_rows,
    extract_joern_user_symbol_rows,
    generate_joern_slice_graph,
)


DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "code_gadget"
DEFAULT_PROJECT_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "project_context"
DEFAULT_PROJECT_REPO_MAP_PATH = Path(__file__).resolve().parents[1] / "Source" / "project_repo_urls.json"
CODE_GADGET_CACHE_VERSION = "v17_joern_only_revert"

CONTROL_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "catch",
}
IDENTIFIER_BLACKLIST = CONTROL_KEYWORDS | {
    "true",
    "false",
    "null",
    "nullptr",
    "const",
    "static",
    "volatile",
    "unsigned",
    "signed",
    "struct",
    "class",
    "enum",
    "union",
    "void",
    "char",
    "short",
    "int",
    "long",
    "float",
    "double",
    "bool",
}
PAPER_FORWARD_API_PATTERNS = {
    "cin",
    "getenv",
    "getenv_s",
    "wgetenv",
    "wgetenv_s",
    "catgets",
    "gets",
    "getchar",
    "getc",
    "getch",
    "getche",
    "kbhit",
    "getdlgtext",
    "getpass",
    "scanf",
    "fscanf",
    "vscanf",
    "vfscanf",
    "get",
    "getline",
    "peek",
    "read*",
    "putback",
    "sbumpc",
    "sgetc",
    "sgetn",
    "snextc",
    "sputbackc",
    "sendmessage",
    "sendmessagecallback",
    "sendnotifymessage",
    "postmessage",
    "postthreadmessage",
    "recv",
    "recvfrom",
    "recvmsg",
    "receive",
    "receivefrom",
    "receivefromex",
    "receive*",
    "fgets",
    "sscanf",
    "swscanf",
    "sscanf_s",
    "swscanf_s",
    "winmain",
    "getrawinput*",
    "getcomboboxinfo",
    "getwindowtext",
    "getkeynametext",
    "dde*",
    "getfilemui*",
    "getlocaleinfo*",
    "getstring*",
    "getcursor*",
    "getscroll*",
    "getdlgitem*",
    "getmenuitem*",
}
FORWARD_BUFFER_ARG_PATTERNS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("recv", (2,)),
    ("recvfrom", (2,)),
    ("recvmsg", (2,)),
    ("receive", (1,)),
    ("receivefrom", (2,)),
    ("receivefromex", (2,)),
    ("receive*", (1, 2)),
    ("read", (2,)),
    ("read*", (2,)),
    ("fread", (1,)),
    ("gets", (1,)),
    ("fgets", (1,)),
    ("getline", (1, 2)),
    ("getdelim", (1, 2)),
    ("scanf", (2,)),
    ("fscanf", (3,)),
    ("vscanf", (2,)),
    ("vfscanf", (3,)),
    ("sscanf", (3,)),
    ("swscanf", (3,)),
    ("sscanf_s", (3,)),
    ("swscanf_s", (3,)),
    ("getdlgtext", (2,)),
    ("getwindowtext", (2,)),
    ("getkeynametext", (2,)),
    ("getlocaleinfo*", (3,)),
    ("getrawinput*", (3,)),
)
FORWARD_RETURN_SOURCE_PATTERNS = {
    "catgets",
    "getc",
    "getchar",
    "getch",
    "getche",
    "getenv",
    "getenv_s",
    "wgetenv",
    "wgetenv_s",
    "getpass",
    "peek",
    "sbumpc",
    "sgetc",
    "sgetn",
    "snextc",
    "putback",
    "get",
    "getstring*",
}

CALL_NAME_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*(?:(?:::|->|\.)[A-Za-z_][A-Za-z0-9_]*)*)\s*\(")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
LINE_COMMENT_RE = re.compile(r"//.*?$", flags=re.MULTILINE)
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(
    r"""
    [A-Za-z_][A-Za-z0-9_]* |
    0x[0-9A-Fa-f]+ |
    \d+\.\d+ |
    \d+ |
    ==|!=|<=|>=|->|::|\+\+|--|&&|\|\||<<|>>|[-+*/%&|^~!=<>?:;.,()[\]{}]
    """,
    flags=re.VERBOSE,
)
CPP_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
GRAPH_VARIABLE_LABELS = {"IDENTIFIER", "METHOD_PARAMETER_IN", "LOCAL"}
GRAPH_FLOW_LABEL = "REACHING_DEF"


@dataclass(frozen=True)
class CallCandidate:
    raw_name: str
    normalized_name: str
    line_number: int
    statement_text: str
    arg_texts: List[str]
    arg_identifiers: List[List[str]]
    caller_method_name: str = ""
    target_method_name: str = ""
    method_full_name: str = ""
    dispatch_type: str = ""
    is_external_target: bool = False


@dataclass(frozen=True)
class UserSymbol:
    kind: str
    name: str
    line_number: int
    parent_method_name: str
    order: int = -1


@dataclass(frozen=True)
class FunctionScope:
    name: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class ProjectContext:
    project_name: str
    commit_id: str
    checkout_root: Path
    source_path: Path
    source_text: str
    snippet_start_line: int
    snippet_end_line: int


def _cache_key(
    *,
    code_text: str,
    project_name: Optional[str],
    commit_id: Optional[str],
    strict_project_context: bool,
    max_hops: Optional[int],
    slice_depth: int,
    parallelism: int,
    timeout_sec: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(code_text or "").encode("utf-8"))
    digest.update(str(project_name or "").encode("utf-8"))
    digest.update(str(commit_id or "").encode("utf-8"))
    digest.update(str(bool(strict_project_context)).encode("utf-8"))
    digest.update(str(max_hops).encode("utf-8"))
    digest.update(str(int(slice_depth)).encode("utf-8"))
    digest.update(str(int(parallelism)).encode("utf-8"))
    digest.update(str(int(timeout_sec)).encode("utf-8"))
    digest.update(CODE_GADGET_CACHE_VERSION.encode("utf-8"))
    return digest.hexdigest()


def _strip_comments(text: str) -> str:
    without_block = BLOCK_COMMENT_RE.sub("", str(text or ""))
    return LINE_COMMENT_RE.sub("", without_block)


def _source_lines(code_text: str) -> Dict[int, str]:
    return {idx: line.rstrip("\n") for idx, line in enumerate(str(code_text or "").splitlines(), start=1)}


def _run_checked(cmd: Sequence[str], *, cwd: Optional[Path] = None, timeout_sec: int = 180) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Command timed out: {' '.join(map(str, cmd))}\nstdout:\n{exc.stdout or ''}\nstderr:\n{exc.stderr or ''}"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(map(str, cmd))}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def _safe_project_key(project_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(project_name or "").strip().lower())
    return safe or "project"


def _load_project_repo_map(path: Optional[Path]) -> Dict[str, str]:
    resolved = Path(path).expanduser() if path is not None else DEFAULT_PROJECT_REPO_MAP_PATH
    if not resolved.is_file():
        return {}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(value, str) or not value.strip():
            continue
        out[_safe_project_key(str(key))] = value.strip()
    return out


def _candidate_local_repos(project_name: str, project_source_root: Optional[Path]) -> List[Path]:
    if project_source_root is None:
        return []
    root = Path(project_source_root).expanduser()
    if not root.is_dir():
        return []
    safe_key = _safe_project_key(project_name)
    wanted = {str(project_name or "").strip().lower(), safe_key}
    candidates: List[Path] = []
    direct = root / str(project_name or "")
    if direct.is_dir():
        candidates.append(direct)
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name.lower() in wanted or _safe_project_key(child.name) == safe_key:
            candidates.append(child)
    unique: List[Path] = []
    seen = set()
    for path in candidates:
        real = path.resolve()
        if real in seen:
            continue
        seen.add(real)
        unique.append(real)
    return unique


def _git_has_commit(repo_root: Path, commit_id: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{commit_id}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def _ensure_project_repo(
    *,
    project_name: str,
    commit_id: str,
    project_source_root: Optional[Path],
    project_cache_dir: Path,
    project_repo_map_path: Optional[Path],
    timeout_sec: int,
) -> Path:
    for candidate in _candidate_local_repos(project_name, project_source_root):
        if (candidate / ".git").exists() or subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
        ).returncode == 0:
            if _git_has_commit(candidate, commit_id):
                return candidate

    repo_map = _load_project_repo_map(project_repo_map_path)
    repo_url = repo_map.get(_safe_project_key(project_name))
    if not repo_url:
        raise RuntimeError(
            f"Strict project context requires a local repo or repo URL mapping for project={project_name!r}."
        )

    project_cache_dir.mkdir(parents=True, exist_ok=True)
    repo_root = project_cache_dir / "repos" / _safe_project_key(project_name)
    if not repo_root.exists():
        repo_root.parent.mkdir(parents=True, exist_ok=True)
        _run_checked(["git", "clone", "--no-checkout", repo_url, str(repo_root)], timeout_sec=timeout_sec)
    _run_checked(["git", "-C", str(repo_root), "fetch", "--all", "--tags", "--prune"], timeout_sec=timeout_sec)
    if not _git_has_commit(repo_root, commit_id):
        _run_checked(["git", "-C", str(repo_root), "fetch", "origin", commit_id], timeout_sec=timeout_sec)
    if not _git_has_commit(repo_root, commit_id):
        raise RuntimeError(f"Commit {commit_id} not found for project={project_name!r}.")
    return repo_root


def _ensure_detached_worktree(repo_root: Path, *, project_name: str, commit_id: str, project_cache_dir: Path, timeout_sec: int) -> Path:
    worktree_root = project_cache_dir / "worktrees" / _safe_project_key(project_name) / str(commit_id)
    if worktree_root.is_dir():
        return worktree_root
    worktree_root.parent.mkdir(parents=True, exist_ok=True)
    _run_checked(
        ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(worktree_root), str(commit_id)],
        timeout_sec=timeout_sec,
    )
    return worktree_root


def _normalized_nonempty_lines(text: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for line_no, raw in enumerate(str(text or "").splitlines(), start=1):
        cleaned = _ascii_no_comments(raw)
        collapsed = WHITESPACE_RE.sub(" ", cleaned).strip()
        if not collapsed:
            continue
        out.append((line_no, collapsed))
    return out


def _match_snippet_in_text(text: str, snippet: str) -> Optional[Tuple[int, int]]:
    body = str(snippet or "").strip("\n")
    if not body.strip():
        return None
    raw_idx = str(text or "").find(body)
    if raw_idx >= 0:
        start_line = str(text or "")[:raw_idx].count("\n") + 1
        line_count = body.count("\n") + 1
        return start_line, start_line + line_count - 1

    file_lines = _normalized_nonempty_lines(text)
    snippet_lines = [line for _, line in _normalized_nonempty_lines(body)]
    if not snippet_lines or len(snippet_lines) > len(file_lines):
        return None
    wanted = "\n".join(snippet_lines)
    for idx in range(0, len(file_lines) - len(snippet_lines) + 1):
        candidate = "\n".join(line for _, line in file_lines[idx : idx + len(snippet_lines)])
        if candidate != wanted:
            continue
        return file_lines[idx][0], file_lines[idx + len(snippet_lines) - 1][0]
    return None


def _locate_snippet_in_checkout(checkout_root: Path, code_text: str) -> Tuple[Path, str, int, int]:
    matches: List[Tuple[Path, str, int, int]] = []
    for path in checkout_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in CPP_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        span = _match_snippet_in_text(text, code_text)
        if span is None:
            continue
        start_line, end_line = span
        matches.append((path, text, start_line, end_line))
    if not matches:
        raise RuntimeError("Unable to locate target snippet in the resolved project checkout.")
    if len(matches) > 1:
        matches.sort(key=lambda item: (len(item[0].parts), len(item[1]), str(item[0])))
    return matches[0]


def _resolve_project_context(
    *,
    code_text: str,
    project_name: str,
    commit_id: str,
    project_source_root: Optional[Path],
    project_cache_dir: Path,
    project_repo_map_path: Optional[Path],
    timeout_sec: int,
) -> ProjectContext:
    repo_root = _ensure_project_repo(
        project_name=project_name,
        commit_id=commit_id,
        project_source_root=project_source_root,
        project_cache_dir=project_cache_dir,
        project_repo_map_path=project_repo_map_path,
        timeout_sec=timeout_sec,
    )
    checkout_root = _ensure_detached_worktree(
        repo_root,
        project_name=project_name,
        commit_id=commit_id,
        project_cache_dir=project_cache_dir,
        timeout_sec=timeout_sec,
    )
    source_path, source_text, start_line, end_line = _locate_snippet_in_checkout(checkout_root, code_text)
    return ProjectContext(
        project_name=str(project_name),
        commit_id=str(commit_id),
        checkout_root=checkout_root,
        source_path=source_path,
        source_text=source_text,
        snippet_start_line=int(start_line),
        snippet_end_line=int(end_line),
    )


def _ascii_no_comments(text: str) -> str:
    cleaned = _strip_comments(str(text or ""))
    return cleaned.encode("ascii", errors="ignore").decode("ascii")


def _normalize_call_name(raw_name: str) -> str:
    parts = re.split(r"::|->|\.", str(raw_name or ""))
    return parts[-1].strip() if parts else str(raw_name or "").strip()


def _normalize_identifier(raw_name: str) -> str:
    parts = re.split(r"::|->|\.", str(raw_name or ""))
    return parts[-1].strip().lower() if parts else str(raw_name or "").strip().lower()


def _find_matching_paren(text: str, open_idx: int) -> Optional[int]:
    depth = 0
    for idx in range(open_idx, len(text)):
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return idx
    return None


def _split_top_level_args(text: str) -> List[str]:
    parts: List[str] = []
    depth_paren = depth_bracket = depth_brace = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        elif ch == "," and depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
            part = text[start:idx].strip()
            if part:
                parts.append(part)
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_identifiers(expr: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for match in IDENT_RE.finditer(str(expr or "")):
        ident = match.group(0)
        lowered = ident.lower()
        if lowered in IDENTIFIER_BLACKLIST:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(ident)
    return out


def _lexical_tokens(text: str) -> List[str]:
    return [match.group(0) for match in TOKEN_RE.finditer(str(text or ""))]


def _statement_lookup_by_line(code_text: str) -> Dict[int, str]:
    return {
        line_no: line.rstrip("\n")
        for line_no, line in enumerate(str(code_text or "").splitlines(), start=1)
    }


def _line_to_function_map(scopes: Sequence[FunctionScope]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for scope in scopes:
        for line_no in range(int(scope.start_line), int(scope.end_line) + 1):
            mapping[int(line_no)] = str(scope.name)
    return mapping


def _build_function_call_graph(
    all_calls: Sequence[CallCandidate],
    *,
    user_defined_functions: Sequence[str],
) -> Dict[str, List[str]]:
    user_defined = {str(name) for name in user_defined_functions if str(name)}
    outgoing: Dict[str, set[str]] = defaultdict(set)
    for candidate in all_calls:
        caller = str(candidate.caller_method_name or "")
        callee = str(candidate.target_method_name or candidate.normalized_name or "")
        if not caller or not callee or callee not in user_defined or caller == callee:
            continue
        outgoing[str(caller)].add(callee)
    return {name: sorted(targets) for name, targets in outgoing.items()}


def _ordered_function_pieces(
    function_names: Sequence[str],
    *,
    function_call_graph: Dict[str, List[str]],
    seed: int,
) -> List[str]:
    nodes = [str(name) for name in function_names if str(name)]
    node_set = set(nodes)
    indegree: Dict[str, int] = {name: 0 for name in node_set}
    outgoing: Dict[str, set[str]] = {name: set() for name in node_set}
    for src in node_set:
        for dst in function_call_graph.get(src, []):
            if dst not in node_set or dst == src:
                continue
            if dst in outgoing[src]:
                continue
            outgoing[src].add(dst)
            indegree[dst] += 1

    rng = random.Random(int(seed))
    ready = [name for name in node_set if indegree[name] == 0]
    rng.shuffle(ready)
    ordered: List[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        next_ready: List[str] = []
        for dst in sorted(outgoing[current]):
            indegree[dst] -= 1
            if indegree[dst] == 0:
                next_ready.append(dst)
        rng.shuffle(next_ready)
        ready.extend(next_ready)

    remaining = [name for name in nodes if name not in ordered]
    if remaining:
        rng.shuffle(remaining)
        ordered.extend(remaining)
    return ordered


def _assemble_code_gadget(
    *,
    line_numbers: Sequence[int],
    source_map: Dict[int, str],
    line_to_function: Dict[int, str],
    function_call_graph: Dict[str, List[str]],
    seed: int,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    deduped_lines = sorted({int(line_no) for line_no in line_numbers if int(line_no) > 0})
    pieces_by_function: Dict[str, List[int]] = defaultdict(list)
    unscoped_lines: List[int] = []
    for line_no in deduped_lines:
        owner = line_to_function.get(int(line_no))
        if owner:
            pieces_by_function[str(owner)].append(int(line_no))
        else:
            unscoped_lines.append(int(line_no))

    ordered_functions = _ordered_function_pieces(
        sorted(pieces_by_function.keys()),
        function_call_graph=function_call_graph,
        seed=seed,
    )

    assembled_sequence: List[int] = []
    piece_summaries: List[Dict[str, Any]] = []
    if unscoped_lines:
        assembled_sequence.extend(unscoped_lines)
        piece_summaries.append(
            {
                "function_name": None,
                "line_sequence": list(unscoped_lines),
                "text": "\n".join(source_map.get(line_no, "") for line_no in unscoped_lines),
            }
        )

    for function_name in ordered_functions:
        piece_lines = sorted({int(line_no) for line_no in pieces_by_function.get(function_name, []) if int(line_no) > 0})
        if not piece_lines:
            continue
        assembled_sequence.extend(piece_lines)
        piece_summaries.append(
            {
                "function_name": str(function_name),
                "line_sequence": list(piece_lines),
                "text": "\n".join(source_map.get(line_no, "") for line_no in piece_lines),
            }
        )

    return assembled_sequence, piece_summaries


def _ordered_symbol_names(lines: Sequence[str], candidates: Sequence[str], *, prefix: str) -> Dict[str, str]:
    normalized_candidates = [str(name) for name in candidates if str(name)]
    mapping: Dict[str, str] = {}
    for line in lines:
        for match in IDENT_RE.finditer(str(line or "")):
            token = match.group(0)
            if token not in normalized_candidates:
                continue
            if token in mapping:
                continue
            mapping[token] = f"{prefix}{len(mapping) + 1}"
    return mapping


def _symbolic_code_gadget(
    *,
    assembled_lines: Sequence[int],
    source_map: Dict[int, str],
    user_defined_functions: Sequence[str],
    api_call_names: Sequence[str],
    user_defined_variable_names: Sequence[str],
) -> Tuple[str, Dict[str, Any]]:
    raw_lines = [_ascii_no_comments(source_map.get(int(line_no), "")) for line_no in assembled_lines]
    cleaned_lines = [line.rstrip() for line in raw_lines if line.strip()]

    function_mapping = _ordered_symbol_names(
        cleaned_lines,
        user_defined_functions,
        prefix="FUN",
    )

    api_names = {str(name) for name in api_call_names if str(name)}
    allowed_variables = {str(name) for name in user_defined_variable_names if str(name)}
    variable_candidates: List[str] = []
    seen_variables = set()
    for line in cleaned_lines:
        for match in IDENT_RE.finditer(line):
            token = match.group(0)
            lowered = token.lower()
            if lowered in IDENTIFIER_BLACKLIST:
                continue
            if token in function_mapping:
                continue
            if token in api_names:
                continue
            if token not in allowed_variables:
                continue
            if token in seen_variables:
                continue
            seen_variables.add(token)
            variable_candidates.append(token)

    variable_mapping = {token: f"VAR{idx}" for idx, token in enumerate(variable_candidates, start=1)}

    rendered_lines: List[str] = []
    for line in cleaned_lines:
        rendered = line
        for original, replacement in function_mapping.items():
            rendered = re.sub(rf"\b{re.escape(original)}\b", replacement, rendered)
        for original, replacement in variable_mapping.items():
            rendered = re.sub(rf"\b{re.escape(original)}\b", replacement, rendered)
        rendered_lines.append(rendered)

    return "\n".join(rendered_lines), {
        "function_mapping": function_mapping,
        "variable_mapping": variable_mapping,
    }


def _parse_call_text(raw_call_name: str, call_code: str) -> Tuple[List[str], List[List[str]]]:
    cleaned = _strip_comments(str(call_code or "")).strip()
    if not cleaned:
        return [], []
    match = CALL_NAME_RE.search(cleaned)
    if not match:
        return [], []
    open_idx = cleaned.find("(", match.start())
    if open_idx < 0:
        return [], []
    close_idx = _find_matching_paren(cleaned, open_idx)
    if close_idx is None:
        return [], []
    arg_blob = cleaned[open_idx + 1 : close_idx]
    arg_texts = _split_top_level_args(arg_blob)
    return arg_texts, [_extract_identifiers(part) for part in arg_texts]


def _call_candidate_from_joern_row(
    row: Dict[str, Any],
    *,
    statement_lookup: Dict[int, str],
) -> Optional[CallCandidate]:
    raw_name = str(row.get("name") or "")
    normalized_name = _normalize_call_name(raw_name)
    if not raw_name or not normalized_name:
        return None
    if normalized_name.lower() in CONTROL_KEYWORDS:
        return None
    line_number = int(row.get("line") or -1)
    if line_number <= 0:
        return None
    call_code = str(row.get("code") or "")
    statement_text = str(statement_lookup.get(line_number) or call_code or "")
    raw_arg_texts = row.get("argTexts")
    if isinstance(raw_arg_texts, list):
        arg_texts = [str(item) for item in raw_arg_texts]
        arg_identifiers = [_extract_identifiers(part) for part in arg_texts]
    else:
        arg_texts, arg_identifiers = _parse_call_text(raw_name, call_code)
    return CallCandidate(
        raw_name=raw_name,
        normalized_name=normalized_name,
        line_number=line_number,
        statement_text=statement_text,
        arg_texts=arg_texts,
        arg_identifiers=arg_identifiers,
        caller_method_name=str(row.get("callerMethodName") or ""),
        target_method_name=str(row.get("targetMethodName") or ""),
        method_full_name=str(row.get("methodFullName") or ""),
        dispatch_type=str(row.get("dispatchType") or ""),
        is_external_target=bool(row.get("isExternalTarget")),
    )


def _select_api_call_candidates(
    all_calls: Sequence[CallCandidate],
    *,
    declaration_only_targets: Optional[set[str]] = None,
) -> List[CallCandidate]:
    declaration_only_targets = {
        str(name)
        for name in (declaration_only_targets or set())
        if str(name)
    }
    selected: List[CallCandidate] = []
    for candidate in all_calls:
        if (
            not candidate.is_external_target
            and str(candidate.target_method_name or "") not in declaration_only_targets
        ):
            continue
        if str(candidate.raw_name).startswith("<operator>."):
            continue
        if not candidate.arg_texts and "(" not in candidate.statement_text:
            continue
        selected.append(candidate)
    return selected


def _user_symbols_from_joern_rows(rows: Sequence[Dict[str, Any]]) -> List[UserSymbol]:
    out: List[UserSymbol] = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("kind") or "")
        name = str(row.get("name") or "")
        parent_method_name = str(row.get("parentMethodName") or "")
        line_number = int(row.get("line") or -1)
        if not kind or not name or not parent_method_name:
            continue
        key = (kind, name, parent_method_name, line_number)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            UserSymbol(
                kind=kind,
                name=name,
                line_number=line_number,
                parent_method_name=parent_method_name,
                order=int(row.get("order") or -1),
            )
        )
    return out


def _function_scopes_from_joern_rows(rows: Sequence[Dict[str, Any]]) -> List[FunctionScope]:
    scopes: List[FunctionScope] = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if bool(row.get("isExternal")):
            continue
        name = str(row.get("name") or "")
        if not name or name.startswith("<"):
            continue
        start_line = int(row.get("lineStart") or -1)
        end_line = int(row.get("lineEnd") or -1)
        if start_line <= 0 or end_line < start_line:
            continue
        key = (name, start_line, end_line)
        if key in seen:
            continue
        seen.add(key)
        scopes.append(FunctionScope(name=name, start_line=start_line, end_line=end_line))
    scopes.sort(key=lambda scope: (int(scope.start_line), int(scope.end_line), str(scope.name)))
    return scopes


def _declaration_only_method_names(
    rows: Sequence[Dict[str, Any]],
    *,
    source_map: Dict[int, str],
) -> set[str]:
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        if not name or name.startswith("<"):
            continue
        start_line = int(row.get("lineStart") or -1)
        end_line = int(row.get("lineEnd") or -1)
        if start_line <= 0 or end_line <= 0:
            continue
        if end_line != start_line:
            continue
        line_text = str(source_map.get(start_line) or "").strip()
        if not line_text:
            continue
        if line_text.endswith(";") and "{" not in line_text:
            names.add(name)
    return names


def _call_direction(candidate: CallCandidate) -> str:
    normalized = str(candidate.normalized_name or "").strip().lower()
    if any(fnmatchcase(normalized, pattern) for pattern in PAPER_FORWARD_API_PATTERNS):
        return "forward"
    return "backward"


def _entry_arg_index(entry: Dict[str, Any]) -> int:
    if "argIndex" in entry:
        try:
            return int(entry.get("argIndex"))
        except Exception:
            return -1
    if "arg_index" in entry:
        try:
            return int(entry.get("arg_index"))
        except Exception:
            return -1
    return -1


def _forward_source_arg_indices(candidate: CallCandidate) -> List[int]:
    normalized = str(candidate.normalized_name or "").strip().lower()
    arg_count = int(len(candidate.arg_texts))
    if arg_count <= 0:
        return []
    for pattern, indices in FORWARD_BUFFER_ARG_PATTERNS:
        if fnmatchcase(normalized, pattern):
            out = [idx for idx in indices if 1 <= int(idx) <= arg_count]
            if out:
                return out
    if _forward_has_return_source(candidate):
        return []
    return list(range(1, arg_count + 1))


def _forward_has_return_source(candidate: CallCandidate) -> bool:
    normalized = str(candidate.normalized_name or "").strip().lower()
    return any(fnmatchcase(normalized, pattern) for pattern in FORWARD_RETURN_SOURCE_PATTERNS)


def _extract_assignment_lhs_identifiers(statement_text: str, call_name: str) -> List[str]:
    text = str(statement_text or "")
    raw_name = str(call_name or "")
    if not text or not raw_name:
        return []
    if raw_name not in text:
        return []
    prefix = text.split(raw_name, 1)[0]
    if "=" not in prefix:
        return []
    lhs = prefix.rsplit("=", 1)[0]
    idents = _extract_identifiers(lhs)
    if not idents:
        return []
    return [idents[-1]]


def _summarize_arg_flow_rows(rows: Sequence[Dict[str, Any]], *, direction: str) -> Dict[str, Any]:
    return {
        "direction": str(direction),
        "flow_row_count": int(len(rows)),
        "path_count": int(sum(int(row.get("pathCount", 0)) for row in rows if isinstance(row, dict))),
    }


def _arg_flow_group_summary(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "arg_code": str(entry.get("argCode") or entry.get("arg_code") or ""),
        "arg_index": _entry_arg_index(entry),
        "caller_method": str(entry.get("callerMethodName") or entry.get("caller_method") or ""),
        "line_sequences": [
            [int(line_no) for line_no in sequence if int(line_no) > 0]
            for sequence in (entry.get("lineSequences") or entry.get("line_sequences") or [])
            if isinstance(sequence, list)
        ],
        "direction": str(entry.get("direction") or ""),
    }


def _call_key(candidate: CallCandidate) -> Tuple[str, int, str]:
    return (
        str(candidate.caller_method_name or ""),
        int(candidate.line_number),
        str(candidate.raw_name or ""),
    )


def _flow_row_call_key(entry: Dict[str, Any]) -> Tuple[str, int, str]:
    return (
        str(entry.get("callerMethodName") or entry.get("caller_method") or ""),
        int(entry.get("line") or entry.get("call_line") or -1),
        str(entry.get("name") or entry.get("call_name") or entry.get("arg_code") or ""),
    )


def _match_argument_flow_rows(
    flow_rows: Sequence[Dict[str, Any]],
    *,
    candidate: CallCandidate,
    arg_index: int,
) -> List[Dict[str, Any]]:
    wanted_key = _call_key(candidate)
    matches: List[Dict[str, Any]] = []
    seen = set()
    for entry in flow_rows:
        if not isinstance(entry, dict):
            continue
        if _flow_row_call_key(entry) != wanted_key:
            continue
        if _entry_arg_index(entry) != int(arg_index):
            continue
        key = (
            _flow_row_call_key(entry),
            _entry_arg_index(entry),
            tuple(
                tuple(int(line_no) for line_no in seq if int(line_no) > 0)
                for seq in (entry.get("lineSequences") or entry.get("line_sequences") or [])
                if isinstance(seq, list)
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        matches.append(entry)
    return matches


def _line_numbers_from_flow_rows(
    *,
    call_line: int,
    flow_rows: Sequence[Dict[str, Any]],
) -> List[int]:
    line_numbers = {int(call_line)}
    for entry in flow_rows:
        for sequence in (entry.get("lineSequences") or entry.get("line_sequences") or []):
            if not isinstance(sequence, list):
                continue
            for line_no in sequence:
                parsed = int(line_no)
                if parsed > 0:
                    line_numbers.add(parsed)
    return sorted(line_numbers)


def _has_nonempty_line_sequences(flow_rows: Sequence[Dict[str, Any]]) -> bool:
    for entry in flow_rows:
        if not isinstance(entry, dict):
            continue
        for sequence in (entry.get("lineSequences") or entry.get("line_sequences") or []):
            if not isinstance(sequence, list):
                continue
            if any(int(line_no) > 0 for line_no in sequence):
                return True
    return False


def _compact_line_sequence(values: Sequence[int]) -> List[int]:
    out: List[int] = []
    for value in values:
        parsed = int(value)
        if parsed <= 0:
            continue
        if out and out[-1] == parsed:
            continue
        out.append(parsed)
    return out


def _extract_graph_argument_flow_groups(
    graph_payload: Dict[str, Any],
    *,
    candidate: CallCandidate,
    arg_index: int,
    arg_text: str,
    arg_identifiers: Sequence[str],
    max_hops: Optional[int],
    direction: str,
) -> List[Dict[str, Any]]:
    graph = dict(graph_payload.get("graph") or {})
    nodes = [dict(item) for item in (graph.get("nodes") or []) if isinstance(item, dict)]
    edges = [dict(item) for item in (graph.get("edges") or []) if isinstance(item, dict)]
    wanted = {_normalize_identifier(name) for name in arg_identifiers if _normalize_identifier(name)}
    if not wanted:
        return []

    node_map = {int(node.get("id")): node for node in nodes if int(node.get("id") or -1) >= 0}
    out_edges: Dict[int, List[int]] = defaultdict(list)
    in_edges: Dict[int, List[int]] = defaultdict(list)
    for edge in edges:
        if str(edge.get("label") or "") != GRAPH_FLOW_LABEL:
            continue
        src = int(edge.get("src") or -1)
        dst = int(edge.get("dst") or -1)
        if src < 0 or dst < 0:
            continue
        out_edges[src].append(dst)
        in_edges[dst].append(src)

    anchor_ids: List[int] = []
    for node in nodes:
        node_id = int(node.get("id") or -1)
        if node_id < 0:
            continue
        if str(node.get("label") or "") not in GRAPH_VARIABLE_LABELS:
            continue
        if str(node.get("parentMethod") or "") != str(candidate.caller_method_name or ""):
            continue
        if int(node.get("lineNumber") or -1) != int(candidate.line_number):
            continue
        if _normalize_identifier(str(node.get("name") or "")) not in wanted:
            continue
        anchor_ids.append(node_id)
    if not anchor_ids:
        return []

    hop_limit = None if max_hops is None else max(0, int(max_hops))
    collected: List[List[int]] = []

    def dfs(node_id: int, depth: int, path: List[int], seen: set[int], saw_downstream_match: bool) -> None:
        node = node_map.get(int(node_id))
        next_path = list(path)
        next_saw_downstream_match = bool(saw_downstream_match or direction == "backward")
        if node is not None:
            line_no = int(node.get("lineNumber") or -1)
            if line_no > 0:
                next_path.append(line_no)
            node_name = _normalize_identifier(str(node.get("name") or ""))
            if depth > 0 and node_name in wanted:
                next_saw_downstream_match = True
        if hop_limit is not None and depth >= hop_limit:
            base_sequence = [int(candidate.line_number)] + next_path
            compact = _compact_line_sequence(base_sequence if next_saw_downstream_match else [int(candidate.line_number)])
            if compact:
                collected.append(compact)
            return
        adjacency = out_edges if direction == "forward" else in_edges
        next_nodes = [int(nxt) for nxt in adjacency.get(int(node_id), []) if int(nxt) not in seen]
        if not next_nodes:
            base_sequence = [int(candidate.line_number)] + next_path
            if direction == "backward":
                base_sequence = list(reversed(base_sequence))
            compact = _compact_line_sequence(base_sequence if next_saw_downstream_match else [int(candidate.line_number)])
            if compact:
                collected.append(compact)
            return
        advanced = False
        for nxt in next_nodes:
            advanced = True
            dfs(int(nxt), depth + 1, next_path, seen | {int(nxt)}, next_saw_downstream_match)
        if not advanced:
            base_sequence = [int(candidate.line_number)] + next_path
            if direction == "backward":
                base_sequence = list(reversed(base_sequence))
            compact = _compact_line_sequence(base_sequence if next_saw_downstream_match else [int(candidate.line_number)])
            if compact:
                collected.append(compact)

    for anchor_id in sorted(set(anchor_ids)):
        dfs(int(anchor_id), 0, [], {int(anchor_id)}, False)

    unique_sequences: List[List[int]] = []
    seen_sequences = set()
    for sequence in collected:
        key = tuple(int(line_no) for line_no in sequence if int(line_no) > 0)
        if not key or key in seen_sequences:
            continue
        seen_sequences.add(key)
        unique_sequences.append(list(key))
    if not unique_sequences:
        return []
    return [
        {
            "arg_code": str(arg_text or ""),
            "arg_index": int(arg_index),
            "caller_method": str(candidate.caller_method_name or ""),
            "call_line": int(candidate.line_number),
            "call_name": str(candidate.raw_name or ""),
            "line_sequences": unique_sequences,
            "direction": str(direction),
            "pathCount": int(len(unique_sequences)),
        }
    ]


def extract_code_gadget_payload(
    *,
    code_text: str,
    project_name: Optional[str] = None,
    commit_id: Optional[str] = None,
    project_source_root: Optional[Path] = None,
    project_cache_dir: Path = DEFAULT_PROJECT_CACHE_DIR,
    project_repo_map_path: Optional[Path] = DEFAULT_PROJECT_REPO_MAP_PATH,
    strict_project_context: bool = True,
    prompt_text: str = "",
    joern_cli_dir: Path = DEFAULT_JOERN_CLI_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    joern_cache_dir: Path = DEFAULT_JOERN_CACHE_DIR,
    slice_depth: int = 20,
    parallelism: int = 1,
    timeout_sec: int = 180,
    max_hops: Optional[int] = None,
) -> Dict[str, Any]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        _cache_key(
            code_text=code_text,
            project_name=project_name,
            commit_id=commit_id,
            strict_project_context=bool(strict_project_context),
            max_hops=max_hops,
            slice_depth=slice_depth,
            parallelism=parallelism,
            timeout_sec=timeout_sec,
        )
        + ".json"
    )
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    project_context: Optional[ProjectContext] = None
    if strict_project_context:
        if not project_name or not commit_id:
            raise RuntimeError("Strict project context requires both project_name and commit_id.")
        project_context = _resolve_project_context(
            code_text=code_text,
            project_name=project_name,
            commit_id=commit_id,
            project_source_root=project_source_root,
            project_cache_dir=Path(project_cache_dir),
            project_repo_map_path=project_repo_map_path,
            timeout_sec=max(1, int(timeout_sec)),
        )
        analysis_code = project_context.source_text
        analysis_source_path = project_context.source_path
        target_start_line = int(project_context.snippet_start_line)
        target_end_line = int(project_context.snippet_end_line)
    else:
        analysis_code = str(code_text or "")
        analysis_source_path = None
        target_start_line = 1
        target_end_line = max(1, len(str(code_text or "").splitlines()))

    source_map = _source_lines(analysis_code)
    statement_lookup = _statement_lookup_by_line(analysis_code)

    source_errors: Dict[str, str] = {}
    raw_call_rows = extract_joern_call_rows(
        code_text=analysis_code,
        source_path=analysis_source_path,
        prompt_text=str(prompt_text or ""),
        joern_cli_dir=Path(joern_cli_dir),
        timeout_sec=max(1, int(timeout_sec)),
    )
    raw_method_rows = extract_joern_method_rows(
        code_text=analysis_code,
        source_path=analysis_source_path,
        prompt_text=str(prompt_text or ""),
        joern_cli_dir=Path(joern_cli_dir),
        timeout_sec=max(1, int(timeout_sec)),
    )
    raw_symbol_rows = extract_joern_user_symbol_rows(
        code_text=analysis_code,
        source_path=analysis_source_path,
        prompt_text=str(prompt_text or ""),
        joern_cli_dir=Path(joern_cli_dir),
        timeout_sec=max(1, int(timeout_sec)),
    )
    all_calls = [
        candidate
        for candidate in (
            _call_candidate_from_joern_row(row, statement_lookup=statement_lookup)
            for row in raw_call_rows
            if isinstance(row, dict)
        )
        if candidate is not None
    ]
    function_scopes = _function_scopes_from_joern_rows(raw_method_rows)
    declaration_only_targets = _declaration_only_method_names(
        raw_method_rows,
        source_map=source_map,
    )
    call_candidates = _select_api_call_candidates(
        all_calls,
        declaration_only_targets=declaration_only_targets,
    )
    line_to_function = _line_to_function_map(function_scopes)
    user_defined_functions = [scope.name for scope in function_scopes]
    user_symbols = _user_symbols_from_joern_rows(raw_symbol_rows)
    user_defined_variable_names = sorted({symbol.name for symbol in user_symbols})
    function_call_graph = _build_function_call_graph(
        all_calls,
        user_defined_functions=user_defined_functions,
    )
    directions_needed = sorted({_call_direction(candidate) for candidate in call_candidates} or {"backward"})

    flow_rows_by_direction: Dict[str, List[Dict[str, Any]]] = {}
    flow_graph_payload: Optional[Dict[str, Any]] = None
    for active_direction in directions_needed:
        try:
            flow_rows_by_direction[active_direction] = extract_joern_external_call_argument_flows(
                code_text=analysis_code,
                source_path=analysis_source_path,
                prompt_text=str(prompt_text or ""),
                direction=active_direction,
                joern_cli_dir=Path(joern_cli_dir),
                cache_dir=Path(joern_cache_dir),
                timeout_sec=max(1, int(timeout_sec)),
            )
        except Exception as exc:
            source_errors[active_direction] = str(exc)
    if "forward" in directions_needed:
        try:
            flow_graph_payload = generate_joern_slice_graph(
                code_text=analysis_code,
                source_path=analysis_source_path,
                prompt_text=str(prompt_text or ""),
                joern_cli_dir=Path(joern_cli_dir),
                cache_dir=Path(joern_cache_dir),
                slice_depth=max(1, int(slice_depth)),
                parallelism=max(1, int(parallelism)),
                timeout_sec=max(1, int(timeout_sec)),
                allow_empty=True,
            )
        except Exception as exc:
            source_errors["forward_argument_flows"] = str(exc)

    code_gadgets: List[Dict[str, Any]] = []
    for gadget_index, candidate in enumerate(call_candidates):
        active_direction = _call_direction(candidate)
        flow_rows = flow_rows_by_direction.get(active_direction, [])
        active_forward_args = set(_forward_source_arg_indices(candidate)) if active_direction == "forward" else set()
        return_source_identifiers = (
            _extract_assignment_lhs_identifiers(candidate.statement_text, candidate.raw_name)
            if active_direction == "forward" and _forward_has_return_source(candidate)
            else []
        )

        matched_rows: List[Dict[str, Any]] = []
        matched_global_keys = set()
        argument_slices: List[Dict[str, Any]] = []
        for arg_index, arg_idents in enumerate(candidate.arg_identifiers):
            source_arg_enabled = (arg_index + 1) in active_forward_args if active_direction == "forward" else True
            matches = _match_argument_flow_rows(
                flow_rows,
                candidate=candidate,
                arg_index=arg_index + 1,
            ) if source_arg_enabled else []
            if (
                active_direction == "forward"
                and source_arg_enabled
                and flow_graph_payload is not None
                and not _has_nonempty_line_sequences(matches)
            ):
                matches = _extract_graph_argument_flow_groups(
                    flow_graph_payload,
                    candidate=candidate,
                    arg_index=arg_index + 1,
                    arg_text=candidate.arg_texts[arg_index] if arg_index < len(candidate.arg_texts) else "",
                    arg_identifiers=arg_idents,
                    max_hops=max_hops,
                    direction="forward",
                )
            if matches:
                arg_line_numbers = _line_numbers_from_flow_rows(
                    call_line=candidate.line_number,
                    flow_rows=matches,
                )
                arg_line_sequence, _ = _assemble_code_gadget(
                    line_numbers=arg_line_numbers,
                    source_map=source_map,
                    line_to_function=line_to_function,
                    function_call_graph=function_call_graph,
                    seed=int(hashlib.sha256(f"{candidate.raw_name}:{candidate.line_number}:{','.join(arg_idents)}".encode("utf-8")).hexdigest()[:8], 16),
                )
            else:
                arg_line_sequence = [int(candidate.line_number)]
            argument_slices.append(
                {
                    "arg_text": candidate.arg_texts[arg_index] if arg_index < len(candidate.arg_texts) else "",
                    "arg_identifiers": list(arg_idents),
                    "line_sequence": arg_line_sequence,
                    "flow_count": int(sum(int(entry.get("pathCount", 0)) for entry in matches)),
                    "flow_groups": [_arg_flow_group_summary(entry) for entry in matches],
                }
            )
            for entry in matches:
                key = (
                    _flow_row_call_key(entry),
                    _entry_arg_index(entry),
                    tuple(
                        tuple(int(line_no) for line_no in seq if int(line_no) > 0)
                        for seq in (entry.get("lineSequences") or entry.get("line_sequences") or [])
                        if isinstance(seq, list)
                    ),
                )
                if key in matched_global_keys:
                    continue
                matched_global_keys.add(key)
                matched_rows.append(entry)

        if active_direction == "forward" and return_source_identifiers and flow_graph_payload is not None:
            return_matches = _extract_graph_argument_flow_groups(
                flow_graph_payload,
                candidate=candidate,
                arg_index=0,
                arg_text="<return>",
                arg_identifiers=return_source_identifiers,
                max_hops=max_hops,
                direction="forward",
            )
            argument_slices.append(
                {
                    "arg_text": "<return>",
                    "arg_identifiers": list(return_source_identifiers),
                    "line_sequence": _assemble_code_gadget(
                        line_numbers=_line_numbers_from_flow_rows(
                            call_line=candidate.line_number,
                            flow_rows=return_matches,
                        ),
                        source_map=source_map,
                        line_to_function=line_to_function,
                        function_call_graph=function_call_graph,
                        seed=int(hashlib.sha256(f"{candidate.raw_name}:{candidate.line_number}:return".encode("utf-8")).hexdigest()[:8], 16),
                    )[0]
                    if return_matches
                    else [int(candidate.line_number)],
                    "flow_count": int(sum(int(entry.get("pathCount", 0)) for entry in return_matches)),
                    "flow_groups": [_arg_flow_group_summary(entry) for entry in return_matches],
                }
            )
            for entry in return_matches:
                key = (
                    _flow_row_call_key(entry),
                    _entry_arg_index(entry),
                    tuple(
                        tuple(int(line_no) for line_no in seq if int(line_no) > 0)
                        for seq in (entry.get("lineSequences") or entry.get("line_sequences") or [])
                        if isinstance(seq, list)
                    ),
                )
                if key in matched_global_keys:
                    continue
                matched_global_keys.add(key)
                matched_rows.append(entry)

        if not matched_rows:
            continue

        line_sequence_numbers = _line_numbers_from_flow_rows(
            call_line=candidate.line_number,
            flow_rows=matched_rows,
        )
        line_sequence, function_pieces = _assemble_code_gadget(
            line_numbers=line_sequence_numbers,
            source_map=source_map,
            line_to_function=line_to_function,
            function_call_graph=function_call_graph,
            seed=int(hashlib.sha256(f"{candidate.raw_name}:{candidate.line_number}".encode("utf-8")).hexdigest()[:8], 16),
        )
        code_gadget_text = "\n".join(source_map.get(line_no, "") for line_no in line_sequence if line_no in source_map)
        symbolic_code_gadget, symbolic_mappings = _symbolic_code_gadget(
            assembled_lines=line_sequence,
            source_map=source_map,
            user_defined_functions=user_defined_functions,
            api_call_names=[entry.normalized_name for entry in call_candidates],
            user_defined_variable_names=user_defined_variable_names,
        )
        if int(candidate.line_number) < target_start_line or int(candidate.line_number) > target_end_line:
            continue

        symbolic_tokens = _lexical_tokens(symbolic_code_gadget)
        code_gadgets.append(
            {
                "gadget_index": int(gadget_index),
                "api_call_name": candidate.normalized_name,
                "raw_call_name": candidate.raw_name,
                "direction": active_direction,
                "call_line": int(candidate.line_number),
                "statement_text": candidate.statement_text,
                "arg_texts": list(candidate.arg_texts),
                "arg_identifiers": [list(names) for names in candidate.arg_identifiers],
                "argument_slices": argument_slices,
                "line_sequence": line_sequence,
                "snippet_span": {
                    "start_line": int(target_start_line),
                    "end_line": int(target_end_line),
                },
                "function_pieces": function_pieces,
                "source_path": str(analysis_source_path) if analysis_source_path is not None else None,
                "code_gadget": code_gadget_text,
                "symbolic_code_gadget": symbolic_code_gadget,
                "symbolic_tokens": symbolic_tokens,
                "vectorization_policy": {
                    "padding_side": "left" if active_direction == "backward" else "right",
                    "truncation_side": "left" if active_direction == "backward" else "right",
                },
                "symbolic_mappings": symbolic_mappings,
            }
        )

    payload = {
        "code_gadgets": code_gadgets,
        "meta": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "backend": "code_gadget",
            "source_backend": "joern",
            "construction": "vuldeepecker",
            "strict_project_context": bool(strict_project_context),
            "symbolic_transform": True,
            "call_classification": "vuldeepecker_api_keypoints",
            "graph_stats": {
                "direction_summaries": {
                    active_direction: _summarize_arg_flow_rows(rows, direction=active_direction)
                    for active_direction, rows in flow_rows_by_direction.items()
                },
            },
            "gadget_count": int(len(code_gadgets)),
            "api_call_count": int(len(call_candidates)),
            "all_call_count": int(len(all_calls)),
            "function_scope_count": int(len(function_scopes)),
            "user_symbol_count": int(len(user_symbols)),
            "project_name": str(project_name or ""),
            "commit_id": str(commit_id or ""),
            "analysis_source_path": str(analysis_source_path) if analysis_source_path is not None else None,
            "target_start_line": int(target_start_line),
            "target_end_line": int(target_end_line),
            "source_errors": dict(source_errors),
            "slice_empty": len(code_gadgets) == 0,
            "empty_reason": None if code_gadgets else "no-api-call-gadgets",
        },
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
