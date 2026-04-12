from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


DEFAULT_JOERN_CLI_DIR = Path("/people/cs/x/xxr230000/bin/joern/joern-cli")
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "joern_slice"
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

DEFAULT_SINK_FILTER = (
    r"(?i)(strcpy|strncpy|strcat|strncat|memcpy|memmove|gets|scanf|sscanf|fscanf|"
    r"sprintf|snprintf|vsprintf|vsnprintf|read|recv|recvfrom|stpcpy|wcscpy|wcsncpy|"
    r"strlcpy|bcopy)"
)

EMPTY_SLICE_MARKERS = (
    "Empty slice, no file generated.",
    "empty slice, no file generated.",
)


class JoernSliceError(RuntimeError):
    pass


@dataclass(frozen=True)
class SliceNode:
    id: int
    label: str
    name: str
    code: str
    parent_method: str
    parent_file: str
    line_number: Optional[int]
    column_number: Optional[int]


@dataclass(frozen=True)
class SliceEdge:
    src: int
    dst: int
    label: str


@dataclass(frozen=True)
class VariableSlice:
    variable_key: str
    variable_name: str
    parent_method: str
    direction: str
    anchor_node_ids: list[int]
    anchor_lines: list[int]
    slice_node_ids: list[int]
    slice_lines: list[int]
    line_scores: dict[int, float]
    sink_related: bool


LANGUAGE_ALIASES: Dict[str, str] = {
    "c": "c",
    "newc": "c",
    "cpp": "c",
    "c++": "c",
    "cc": "c",
    "cxx": "c",
    "hpp": "c",
    "hxx": "c",
    "h": "c",
    "java": "javasrc",
    "javasrc": "javasrc",
    "python": "python",
    "py": "python",
    "pythonsrc": "python",
    "javascript": "javascript",
    "js": "javascript",
    "jssrc": "javascript",
}

FRONTEND_COMMANDS: Dict[str, str] = {
    "c": "c2cpg.sh",
    "javasrc": "javasrc2cpg",
    "python": "pysrc2cpg",
    "javascript": "jssrc2cpg.sh",
}

DEFAULT_EXTENSIONS: Dict[str, str] = {
    "c": ".c",
    "javasrc": ".java",
    "python": ".py",
    "javascript": ".js",
}

SOURCE_DIR_LANGUAGES = {"python"}
VARIABLE_NODE_LABELS = {"IDENTIFIER", "METHOD_PARAMETER_IN", "LOCAL", "MEMBER"}
FLOW_LABEL = "REACHING_DEF"
UNDIRECTED_CONTEXT_LABELS = {"ARGUMENT", "REF"}
CONTROL_CONTEXT_LABELS = {"CFG", "DOMINATE"}


def _parse_prompt_language(prompt_text: str) -> Optional[str]:
    match = re.search(r"```([A-Za-z0-9_+.-]+)\n", str(prompt_text or ""))
    if not match:
        return None
    return match.group(1).strip().lower()


def infer_joern_language(
    *,
    language_hint: Optional[str] = None,
    source_path: Optional[Path] = None,
    prompt_text: str = "",
) -> str:
    candidates: list[str] = []
    if language_hint:
        candidates.append(str(language_hint).strip().lower())
    prompt_lang = _parse_prompt_language(prompt_text)
    if prompt_lang:
        candidates.append(prompt_lang)
    if source_path is not None and source_path.suffix:
        suffix = source_path.suffix.lower().lstrip(".")
        candidates.append(suffix)
        if suffix in {"cxx", "cc", "hpp", "hxx"}:
            candidates.append("cpp")
    for candidate in candidates:
        if candidate in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[candidate]
    raise JoernSliceError(
        "Unable to infer a Joern language. "
        f"language_hint={language_hint!r}, source_path={str(source_path) if source_path else None!r}"
    )


def infer_source_extension(*, joern_language: str, source_path: Optional[Path] = None) -> str:
    if source_path is not None and source_path.suffix:
        return source_path.suffix
    return DEFAULT_EXTENSIONS.get(joern_language, ".txt")


def _slice_runner_command(joern_cli_dir: Path) -> list[str]:
    conf = joern_cli_dir / "conf" / "log4j2.xml"
    runner = joern_cli_dir / "bin" / "joern-slice"
    return [
        str(runner),
        "-J-XX:+UseG1GC",
        "-J-XX:CompressedClassSpaceSize=128m",
        f"-Dlog4j.configurationFile={conf}",
    ]


def _run_checked(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    timeout_sec: int = 180,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise JoernSliceError(
            "Command timed out.\n"
            f"cmd={' '.join(map(str, cmd))}\n"
            f"cwd={str(cwd) if cwd is not None else os.getcwd()}\n"
            f"timeout_sec={int(timeout_sec)}\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
        ) from exc
    if proc.returncode != 0:
        raise JoernSliceError(
            "Command failed.\n"
            f"cmd={' '.join(map(str, cmd))}\n"
            f"cwd={str(cwd) if cwd is not None else os.getcwd()}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _cache_key(
    *,
    code_text: str,
    joern_language: str,
    slice_depth: int,
    sink_filter: Optional[str],
    joern_cli_dir: Path,
    retry_without_sink_filter: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(code_text.encode("utf-8"))
    digest.update(joern_language.encode("utf-8"))
    digest.update(str(int(slice_depth)).encode("utf-8"))
    digest.update(str(sink_filter or "").encode("utf-8"))
    digest.update(str(Path(joern_cli_dir)).encode("utf-8"))
    digest.update(str(bool(retry_without_sink_filter)).encode("utf-8"))
    return digest.hexdigest()


def _build_empty_graph_payload(
    *,
    joern_language: str,
    slice_depth: int,
    sink_filter: Optional[str],
    source_name: str,
    source_path: Optional[Path],
    empty_reason: str,
    stdout: str,
    stderr: str,
) -> Dict[str, Any]:
    return {
        "meta": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "joern_language": joern_language,
            "slice_depth": int(slice_depth),
            "sink_filter": str(sink_filter or ""),
            "source_name": source_name,
            "source_path": str(source_path) if source_path is not None else None,
            "slice_empty": True,
            "empty_reason": str(empty_reason),
            "slice_stdout_tail": "\n".join(str(stdout or "").splitlines()[-20:]),
            "slice_stderr_tail": "\n".join(str(stderr or "").splitlines()[-20:]),
        },
        "graph": {"nodes": [], "edges": []},
    }


def _is_empty_slice_result(proc: subprocess.CompletedProcess[str]) -> bool:
    text = f"{proc.stdout}\n{proc.stderr}"
    return any(marker in text for marker in EMPTY_SLICE_MARKERS)


def generate_joern_slice_graph(
    *,
    code_text: str,
    language_hint: Optional[str] = None,
    source_path: Optional[Path] = None,
    prompt_text: str = "",
    joern_cli_dir: Path = DEFAULT_JOERN_CLI_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    slice_depth: int = 20,
    sink_filter: Optional[str] = None,
    parallelism: int = 1,
    timeout_sec: int = 180,
    allow_empty: bool = True,
    retry_without_sink_filter: bool = False,
) -> Dict[str, Any]:
    joern_language = infer_joern_language(
        language_hint=language_hint,
        source_path=source_path,
        prompt_text=prompt_text,
    )
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(
        code_text=code_text,
        joern_language=joern_language,
        slice_depth=slice_depth,
        sink_filter=sink_filter,
        joern_cli_dir=Path(joern_cli_dir),
        retry_without_sink_filter=retry_without_sink_filter,
    )
    cache_path = cache_dir / f"{key}.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    frontend_name = FRONTEND_COMMANDS.get(joern_language)
    if not frontend_name:
        raise JoernSliceError(f"No Joern frontend configured for language {joern_language!r}.")
    frontend = Path(joern_cli_dir) / frontend_name
    if not frontend.is_file():
        raise JoernSliceError(
            f"Required Joern frontend is missing for language {joern_language!r}: {frontend}"
        )

    with tempfile.TemporaryDirectory(prefix="joern-slice-") as tmp:
        tmpdir = Path(tmp)
        ext = infer_source_extension(joern_language=joern_language, source_path=source_path)
        file_name = source_path.name if source_path is not None else f"snippet{ext}"
        src_path = tmpdir / file_name
        src_path.write_text(code_text, encoding="utf-8")
        cpg_path = tmpdir / "cpg.bin"

        frontend_input = tmpdir if joern_language in SOURCE_DIR_LANGUAGES else src_path
        _run_checked(
            [str(frontend), str(frontend_input), "--output", str(cpg_path)],
            cwd=tmpdir,
            timeout_sec=timeout_sec,
        )

        def run_slice(active_sink_filter: Optional[str]) -> subprocess.CompletedProcess[str]:
            slice_cmd = _slice_runner_command(Path(joern_cli_dir)) + [
                "data-flow",
                str(cpg_path),
                "-o",
                "slices",
                "--slice-depth",
                str(max(1, int(slice_depth))),
            ]
            if active_sink_filter:
                slice_cmd += ["--sink-filter", str(active_sink_filter)]
            if parallelism and int(parallelism) > 1:
                slice_cmd += ["--parallelism", str(int(parallelism))]
            return _run_checked(slice_cmd, cwd=tmpdir, timeout_sec=timeout_sec)

        proc = run_slice(sink_filter)
        slice_json_path = tmpdir / "slices.json"

        if not slice_json_path.is_file() and retry_without_sink_filter and sink_filter:
            if _is_empty_slice_result(proc):
                proc = run_slice(None)

        if slice_json_path.is_file():
            payload = json.loads(slice_json_path.read_text(encoding="utf-8"))
            cache_payload = {
                "meta": {
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "joern_language": joern_language,
                    "slice_depth": int(slice_depth),
                    "sink_filter": str(sink_filter or ""),
                    "source_name": file_name,
                    "source_path": str(source_path) if source_path is not None else None,
                    "slice_empty": False,
                    "retry_without_sink_filter": bool(retry_without_sink_filter),
                    "slice_stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
                    "slice_stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
                },
                "graph": payload,
            }
            cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
            return cache_payload

        if allow_empty and _is_empty_slice_result(proc):
            cache_payload = _build_empty_graph_payload(
                joern_language=joern_language,
                slice_depth=slice_depth,
                sink_filter=sink_filter,
                source_name=file_name,
                source_path=source_path,
                empty_reason="empty-slice",
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
            cache_payload["meta"]["retry_without_sink_filter"] = bool(retry_without_sink_filter)
            cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
            return cache_payload

        raise JoernSliceError(f"Joern slice output missing: {slice_json_path}")


def _load_graph_objects(graph_payload: Dict[str, Any]) -> tuple[dict[int, SliceNode], list[SliceEdge]]:
    graph = graph_payload.get("graph") if "graph" in graph_payload else graph_payload
    nodes_raw = list(graph.get("nodes") or [])
    edges_raw = list(graph.get("edges") or [])

    nodes: dict[int, SliceNode] = {}
    for raw in nodes_raw:
        node = SliceNode(
            id=int(raw["id"]),
            label=str(raw.get("label", "")),
            name=str(raw.get("name", "")),
            code=str(raw.get("code", "")),
            parent_method=str(raw.get("parentMethod", "")),
            parent_file=str(raw.get("parentFile", "")),
            line_number=int(raw["lineNumber"]) if raw.get("lineNumber") is not None else None,
            column_number=int(raw["columnNumber"]) if raw.get("columnNumber") is not None else None,
        )
        nodes[node.id] = node

    edges = [
        SliceEdge(src=int(raw["src"]), dst=int(raw["dst"]), label=str(raw.get("label", "")))
        for raw in edges_raw
    ]
    return nodes, edges


def _incident_edge_labels(
    node_id: int,
    out_edges: Dict[int, list[SliceEdge]],
    in_edges: Dict[int, list[SliceEdge]],
) -> set[str]:
    labels = {edge.label for edge in out_edges.get(node_id, [])}
    labels.update(edge.label for edge in in_edges.get(node_id, []))
    return labels


def _is_variable_candidate(
    node: SliceNode,
    *,
    out_edges: Dict[int, list[SliceEdge]],
    in_edges: Dict[int, list[SliceEdge]],
) -> bool:
    if node.label not in VARIABLE_NODE_LABELS:
        return False
    if node.parent_method.endswith(":<module>") or node.parent_method == ":<module>":
        return False
    if not IDENT_RE.fullmatch(node.name):
        return False
    labels = _incident_edge_labels(node.id, out_edges, in_edges)
    return FLOW_LABEL in labels or "REF" in labels


def _group_variable_nodes(
    nodes: Dict[int, SliceNode],
    out_edges: Dict[int, list[SliceEdge]],
    in_edges: Dict[int, list[SliceEdge]],
) -> Dict[str, list[int]]:
    groups: Dict[str, list[int]] = defaultdict(list)
    for node in nodes.values():
        if not _is_variable_candidate(node, out_edges=out_edges, in_edges=in_edges):
            continue
        key = f"{node.parent_method}::{node.name}"
        groups[key].append(node.id)
    return {key: sorted(set(ids)) for key, ids in groups.items()}


def _select_anchor_nodes(
    node_ids: Sequence[int],
    *,
    direction: str,
    nodes: Dict[int, SliceNode],
    out_edges: Dict[int, list[SliceEdge]],
    in_edges: Dict[int, list[SliceEdge]],
) -> list[int]:
    if direction == "forward":
        anchors = [
            node_id
            for node_id in node_ids
            if nodes[node_id].label == "METHOD_PARAMETER_IN"
            or any(edge.label == FLOW_LABEL for edge in out_edges.get(node_id, []))
        ]
        if not anchors:
            anchors = [
                node_id
                for node_id in node_ids
                if any(edge.label == "REF" for edge in out_edges.get(node_id, []))
            ]
    else:
        anchors = [
            node_id
            for node_id in node_ids
            if any(edge.label == FLOW_LABEL for edge in in_edges.get(node_id, []))
            and not any(edge.label == FLOW_LABEL for edge in out_edges.get(node_id, []))
        ]
        if not anchors:
            anchors = [
                node_id
                for node_id in node_ids
                if any(edge.label == FLOW_LABEL for edge in in_edges.get(node_id, []))
            ]
    if not anchors:
        anchors = list(node_ids)
    return sorted(set(anchors))


def _core_flow_distances(
    anchor_ids: Sequence[int],
    *,
    direction: str,
    out_edges: Dict[int, list[SliceEdge]],
    in_edges: Dict[int, list[SliceEdge]],
    max_hops: Optional[int],
) -> Dict[int, int]:
    if direction not in {"forward", "backward"}:
        raise JoernSliceError(f"Unsupported direction {direction!r}; expected 'forward' or 'backward'.")
    edges_by_dir = out_edges if direction == "forward" else in_edges
    distances: Dict[int, int] = {}
    queue: deque[tuple[int, int]] = deque()
    for anchor_id in anchor_ids:
        distances[anchor_id] = 0
        queue.append((anchor_id, 0))
    hop_limit = None if max_hops is None else max(0, int(max_hops))
    while queue:
        node_id, depth = queue.popleft()
        if hop_limit is not None and depth >= hop_limit:
            continue
        for edge in edges_by_dir.get(node_id, []):
            if edge.label != FLOW_LABEL:
                continue
            nxt = edge.dst if direction == "forward" else edge.src
            next_depth = depth + 1
            prev = distances.get(nxt)
            if prev is not None and prev <= next_depth:
                continue
            distances[nxt] = next_depth
            queue.append((nxt, next_depth))
    return distances


def _expand_context(
    distances: Dict[int, int],
    *,
    out_edges: Dict[int, list[SliceEdge]],
    in_edges: Dict[int, list[SliceEdge]],
    include_control: bool,
    include_post_dominance: bool,
) -> Dict[int, int]:
    expanded = dict(distances)
    labels = set(UNDIRECTED_CONTEXT_LABELS)
    if include_control:
        labels.update(CONTROL_CONTEXT_LABELS)
    if include_post_dominance:
        labels.add("POST_DOMINATE")

    for node_id, depth in list(distances.items()):
        for edge in out_edges.get(node_id, []):
            if edge.label in labels:
                expanded.setdefault(edge.dst, depth + 1)
        for edge in in_edges.get(node_id, []):
            if edge.label in labels:
                expanded.setdefault(edge.src, depth + 1)
    return expanded


def _node_line(node: SliceNode) -> Optional[int]:
    if node.line_number is None:
        return None
    if int(node.line_number) <= 0:
        return None
    return int(node.line_number)


def _line_scores_from_distances(
    node_distances: Dict[int, int],
    *,
    anchor_ids: Sequence[int],
    sink_node_ids: set[int],
    nodes: Dict[int, SliceNode],
) -> dict[int, float]:
    scores: Dict[int, float] = defaultdict(float)
    anchor_set = set(anchor_ids)
    for node_id, depth in node_distances.items():
        node = nodes.get(node_id)
        if node is None:
            continue
        line_no = _node_line(node)
        if line_no is None:
            continue
        weight = 1.0 / (1.0 + float(depth))
        if node_id in anchor_set:
            weight *= 1.5
        if node.label == "CALL":
            weight *= 1.2
        if node_id in sink_node_ids:
            weight *= 1.3
        scores[line_no] += weight
    return {int(line_no): float(score) for line_no, score in sorted(scores.items())}


def build_variable_slices(
    graph_payload: Dict[str, Any],
    *,
    direction: str = "backward",
    include_control: bool = True,
    include_post_dominance: bool = False,
    max_hops: Optional[int] = None,
    sink_filter: Optional[str] = None,
) -> Dict[str, Any]:
    nodes, edges = _load_graph_objects(graph_payload)
    out_edges: Dict[int, list[SliceEdge]] = defaultdict(list)
    in_edges: Dict[int, list[SliceEdge]] = defaultdict(list)
    for edge in edges:
        out_edges[edge.src].append(edge)
        in_edges[edge.dst].append(edge)

    sink_re = re.compile(str(sink_filter), flags=0) if sink_filter else None
    sink_node_ids: set[int] = set()
    if sink_re is not None:
        for node in nodes.values():
            if node.label != "CALL":
                continue
            if sink_re.search(f"{node.name} {node.code}"):
                sink_node_ids.add(node.id)

    variable_groups = _group_variable_nodes(nodes, out_edges, in_edges)
    variable_slices: list[VariableSlice] = []
    for key, group_node_ids in sorted(variable_groups.items()):
        anchor_ids = _select_anchor_nodes(
            group_node_ids,
            direction=direction,
            nodes=nodes,
            out_edges=out_edges,
            in_edges=in_edges,
        )
        core_distances = _core_flow_distances(
            anchor_ids,
            direction=direction,
            out_edges=out_edges,
            in_edges=in_edges,
            max_hops=max_hops,
        )
        all_distances = _expand_context(
            core_distances,
            out_edges=out_edges,
            in_edges=in_edges,
            include_control=include_control,
            include_post_dominance=include_post_dominance,
        )
        if not all_distances:
            continue
        line_scores = _line_scores_from_distances(
            all_distances,
            anchor_ids=anchor_ids,
            sink_node_ids=sink_node_ids,
            nodes=nodes,
        )
        if not line_scores:
            continue
        variable_name = nodes[group_node_ids[0]].name
        parent_method = nodes[group_node_ids[0]].parent_method
        anchor_lines = sorted(
            {
                line_no
                for node_id in anchor_ids
                for line_no in [_node_line(nodes[node_id])]
                if line_no is not None
            }
        )
        variable_slices.append(
            VariableSlice(
                variable_key=key,
                variable_name=variable_name,
                parent_method=parent_method,
                direction=direction,
                anchor_node_ids=sorted(anchor_ids),
                anchor_lines=anchor_lines,
                slice_node_ids=sorted(all_distances.keys()),
                slice_lines=sorted(line_scores.keys()),
                line_scores=line_scores,
                sink_related=bool(set(all_distances.keys()) & sink_node_ids),
            )
        )

    selected = [entry for entry in variable_slices if entry.sink_related]
    if not selected:
        selected = variable_slices

    aggregate_scores: Dict[int, float] = defaultdict(float)
    for entry in selected:
        for line_no, score in entry.line_scores.items():
            aggregate_scores[int(line_no)] += float(score)

    return {
        "direction": str(direction),
        "num_graph_nodes": int(len(nodes)),
        "num_graph_edges": int(len(edges)),
        "num_variable_slices": int(len(variable_slices)),
        "num_selected_variable_slices": int(len(selected)),
        "sink_filter": str(sink_filter or ""),
        "sink_node_ids": sorted(int(x) for x in sink_node_ids),
        "aggregate_line_scores": {
            int(line_no): float(score)
            for line_no, score in sorted(aggregate_scores.items())
        },
        "variable_slices": [asdict(entry) for entry in variable_slices],
    }


def extract_joern_variable_slices(
    *,
    code_text: str,
    language_hint: Optional[str] = None,
    source_path: Optional[Path] = None,
    prompt_text: str = "",
    direction: str = "backward",
    joern_cli_dir: Path = DEFAULT_JOERN_CLI_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    slice_depth: int = 20,
    sink_filter: Optional[str] = None,
    parallelism: int = 1,
    timeout_sec: int = 180,
    include_control: bool = True,
    include_post_dominance: bool = False,
    max_hops: Optional[int] = None,
    allow_empty: bool = True,
    retry_without_sink_filter: bool = False,
) -> Dict[str, Any]:
    graph_payload = generate_joern_slice_graph(
        code_text=code_text,
        language_hint=language_hint,
        source_path=source_path,
        prompt_text=prompt_text,
        joern_cli_dir=joern_cli_dir,
        cache_dir=cache_dir,
        slice_depth=slice_depth,
        sink_filter=sink_filter,
        parallelism=parallelism,
        timeout_sec=timeout_sec,
        allow_empty=allow_empty,
        retry_without_sink_filter=retry_without_sink_filter,
    )
    payload = build_variable_slices(
        graph_payload,
        direction=direction,
        include_control=include_control,
        include_post_dominance=include_post_dominance,
        max_hops=max_hops,
        sink_filter=sink_filter,
    )
    payload["meta"] = dict(graph_payload.get("meta") or {})
    payload["meta"]["direction"] = str(direction)
    payload["meta"]["joern_cli_dir"] = str(Path(joern_cli_dir))
    payload["meta"]["include_control"] = bool(include_control)
    payload["meta"]["include_post_dominance"] = bool(include_post_dominance)
    payload["meta"]["max_hops"] = None if max_hops is None else int(max_hops)
    payload["meta"]["allow_empty"] = bool(allow_empty)
    return payload


def aggregate_line_scores_to_weights(payload: Dict[str, Any]) -> Dict[int, float]:
    raw = payload.get("aggregate_line_scores") or {}
    return {
        int(line_no): float(score)
        for line_no, score in raw.items()
        if float(score) > 0.0 and int(line_no) > 0
    }
