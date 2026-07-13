"""
Index SoccerNet-Tracking sequences.

After download, SoccerNet-Tracking uses a MOTChallenge-style layout:
    <root>/tracking/<split>/SNMOT-XXX/
        img1/000001.jpg ...
        gt/gt.txt
        seqinfo.ini        # name, seqLength, imWidth, imHeight, frameRate
        gameinfo.ini       # per-clip metadata (teams, actions, source game)

We locate sequences by finding every ``seqinfo.ini`` (robust to extra nesting)
and best-effort extract a *source-match* id so clips from the same game never
straddle a train/val/test boundary.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

SPLIT_NAMES = ("train", "valid", "test", "challenge")

# candidate keys in gameinfo.ini that could identify the source match
_MATCH_KEYS = ("gameID", "game", "match", "matchID", "url", "source",
               "clip_source", "video", "name_game", "game_url")


@dataclass
class SNSequence:
    name: str
    split: str
    path: Path
    length: int
    im_width: int
    im_height: int
    match_id: str
    has_gt: bool


def _read_ini(path: Path) -> dict:
    """Flatten an .ini into a case-preserving {key: value} dict (all sections)."""
    cp = configparser.ConfigParser()
    cp.optionxform = str  # keep original case (seqLength, imWidth, ...)
    out: dict = {}
    try:
        cp.read(path)
    except Exception:
        return out
    for section in cp.sections():
        for k, v in cp.items(section):
            out[k] = v
    return out


def _infer_split(path: Path) -> str:
    for part in path.parts:
        if part in SPLIT_NAMES:
            return part
    return "unknown"


def _match_id(seq_dir: Path, seqinfo: dict) -> str:
    """Best-effort source-match id for leak-free grouping.

    Looks for a game/match identifier in gameinfo.ini. If none is found, falls
    back to the sequence name — each clip becomes its own group, which is still
    leak-safe (whole clips never split), just a coarser grouping. The
    make-splits script reports which case occurred.
    """
    gi = _read_ini(seq_dir / "gameinfo.ini")
    for key in _MATCH_KEYS:
        if gi.get(key):
            return str(gi[key])
    return seqinfo.get("name") or seq_dir.name


def index_sequences(root: Union[str, Path], split: Optional[str] = None) -> list[SNSequence]:
    """Return all SoccerNet-Tracking sequences under ``root`` (optionally one split)."""
    root = Path(root)
    seqs: list[SNSequence] = []
    for seqinfo_path in sorted(root.glob("**/seqinfo.ini")):
        d = seqinfo_path.parent
        info = _read_ini(seqinfo_path)
        sp = _infer_split(d)
        if split and sp != split:
            continue
        seqs.append(
            SNSequence(
                name=info.get("name", d.name),
                split=sp,
                path=d,
                length=int(info.get("seqLength", 0) or 0),
                im_width=int(info.get("imWidth", 0) or 0),
                im_height=int(info.get("imHeight", 0) or 0),
                match_id=_match_id(d, info),
                has_gt=(d / "gt" / "gt.txt").exists(),
            )
        )
    return seqs
