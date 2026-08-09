"""G1 — the durable checkpoint tier must not grow without bound.

The node-local tier has always pruned to 2; S3 pruned NOTHING, so a 24h run at a
120s interval accumulated ~690 objects x ~1.5 GB. The storage bill is the small
half of the problem: ``aws.max_checkpoint_step`` runs a full paginated LIST over
this prefix on every supervisor tick and ``s3_store._s3_latest`` does the same on
every trainer resume, so an unpruned prefix makes the entire run monotonically
slower.

Tested against a local directory -- the same pattern the other s3_store tests
use, and legitimate here because ``prune_checkpoints`` puts both backends behind
one interface exactly like ``latest()``.
"""

from __future__ import annotations

from pathlib import Path

from spot_train import s3_store


def _ckpt(d: Path, step: int, suffix: str = "") -> Path:
    """Mirror checkpoint._ckpt_name: 12-digit zero pad, so lexicographic order
    IS numeric order -- the assumption prune and latest() both rely on."""
    p = d / f"ckpt-{step:012d}.pt{suffix}"
    p.write_bytes(b"x")
    return p


def _steps(d: Path) -> list[int]:
    return sorted(
        int(f.name[len("ckpt-") : -len(".pt")]) for f in d.iterdir() if f.name.endswith(".pt")
    )


def test_keeps_the_n_newest(tmp_path: Path):
    for s in range(1, 16):
        _ckpt(tmp_path, s)
    removed = s3_store.prune_checkpoints(str(tmp_path), keep=10)
    assert removed == 5
    assert _steps(tmp_path) == list(range(6, 16))


def test_ordering_is_numeric_not_lexicographic_by_accident(tmp_path: Path):
    """Zero-padding is load-bearing. Unpadded, '9' would sort after '10' and the
    prune would delete the NEWEST checkpoint -- the worst possible failure."""
    for s in (9, 10, 100, 1000, 12000):
        _ckpt(tmp_path, s)
    s3_store.prune_checkpoints(str(tmp_path), keep=2)
    assert _steps(tmp_path) == [1000, 12000]


def test_in_flight_tmp_file_is_never_a_candidate(tmp_path: Path):
    """An atomic save uploads to <name>.tmp then copies. Deleting one races the
    copy in _s3_save and corrupts the write in progress."""
    for s in range(1, 6):
        _ckpt(tmp_path, s)
    tmp = _ckpt(tmp_path, 6, suffix=".tmp")
    removed = s3_store.prune_checkpoints(str(tmp_path), keep=2)
    assert tmp.exists(), "in-flight .tmp checkpoint was deleted"
    assert removed == 3
    assert _steps(tmp_path) == [4, 5]


def test_keep_zero_disables_pruning(tmp_path: Path):
    for s in range(1, 6):
        _ckpt(tmp_path, s)
    assert s3_store.prune_checkpoints(str(tmp_path), keep=0) == 0
    assert _steps(tmp_path) == [1, 2, 3, 4, 5]


def test_fewer_than_keep_is_a_noop(tmp_path: Path):
    for s in range(1, 4):
        _ckpt(tmp_path, s)
    assert s3_store.prune_checkpoints(str(tmp_path), keep=10) == 0
    assert _steps(tmp_path) == [1, 2, 3]


def test_empty_and_absent_dirs_do_not_raise(tmp_path: Path):
    assert s3_store.prune_checkpoints(str(tmp_path), keep=5) == 0
    assert s3_store.prune_checkpoints(str(tmp_path / "nope"), keep=5) == 0


def test_non_checkpoint_files_are_untouched(tmp_path: Path):
    for s in range(1, 6):
        _ckpt(tmp_path, s)
    other = tmp_path / "metrics.json"
    other.write_text("{}")
    s3_store.prune_checkpoints(str(tmp_path), keep=1)
    assert other.exists()


def test_latest_still_resolves_after_a_prune(tmp_path: Path):
    """The invariant that actually matters: whatever we keep, resume must still
    find the newest one."""
    for s in range(1, 21):
        _ckpt(tmp_path, s)
    s3_store.prune_checkpoints(str(tmp_path), keep=3)
    assert s3_store.latest(str(tmp_path)) == str(tmp_path / f"ckpt-{20:012d}.pt")


def test_prune_failure_never_kills_the_run(tmp_path: Path, monkeypatch):
    """A prune failure must degrade to today's behaviour, not end training."""
    from spot_train import checkpoint

    def boom(*_a, **_k):
        raise RuntimeError("s3 exploded")

    monkeypatch.setattr(s3_store, "prune_checkpoints", boom)
    logged: list[str] = []
    checkpoint._prune_durable(str(tmp_path), 10, logged.append)
    assert logged and "prune failed" in logged[0]
