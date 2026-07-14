from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass(frozen=True)
class SourceSpec:
    name: str; hf_dataset: str; hf_revision: str; split: str
    task_ids: tuple[str, ...]; content_sha: str

@dataclass(frozen=True)
class Manifest:
    version: int; sources: tuple[SourceSpec, ...]

def content_sha(records: list[dict]) -> str:
    blob = json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

def save(m: Manifest, path) -> None:
    data = {"version": m.version,
            "sources": [asdict(s) | {"task_ids": list(s.task_ids)} for s in m.sources]}
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

def load(path) -> Manifest:
    data = json.loads(Path(path).read_text())
    sources = tuple(
        SourceSpec(s["name"], s["hf_dataset"], s["hf_revision"], s["split"],
                   tuple(s["task_ids"]), s["content_sha"])
        for s in data["sources"])
    return Manifest(version=data["version"], sources=sources)
