"""Full-OpenWebText prep tests — hermetic (no network, no HuggingFace).

The prep job runs for ~1-2 hours and writes a 17 GB bin, so the parts worth
pinning are the ones that decide whether an interrupted run corrupts or resumes:
the chunked memmap writer (exact bytes, uint16, offsets), the atomic
partial->final publish, the progress sidecar's resume arithmetic, and the split
math we print before committing to the job. The tokenizing pass itself is
HuggingFace's and is deliberately not exercised here.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PREPARE_PY = os.path.join(_ROOT, "data", "openwebtext", "prepare.py")


def _load_prepare():
    """Import the standalone prep script (it is not part of a package)."""
    spec = importlib.util.spec_from_file_location("owt_prepare", _PREPARE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load_prepare()


def _chunks(seq, n):
    return [np.asarray(seq[i : i + n], dtype=np.uint16) for i in range(0, len(seq), n)]


# --------------------------------------------------------------------------- #
# Chunked writer
# --------------------------------------------------------------------------- #
def test_chunked_write_roundtrips_exact_token_sequence(tmp_path):
    """The whole point: stream chunks into a preallocated memmap and get the
    concatenation back, byte for byte, as uint16."""
    tokens = list(range(1000, 1000 + 512))
    parts = _chunks(tokens, 64)
    final = str(tmp_path / "train.bin")

    writer = prepare.ChunkedBinWriter(final, len(tokens), total_chunks=len(parts))
    for part in parts:
        writer.write(part)
    assert writer.finalize() == len(tokens)

    data = np.memmap(final, dtype=np.uint16, mode="r")
    assert data.dtype == np.uint16
    assert list(data) == tokens
    assert prepare.token_count(final) == len(tokens)
    assert os.path.getsize(final) == len(tokens) * prepare.BYTES_PER_TOKEN


def test_ragged_chunks_land_at_the_running_offset(tmp_path):
    """Real chunks are ragged (documents don't divide evenly), so the offset
    arithmetic — not a fixed stride — is what places the bytes."""
    parts = [np.arange(n, dtype=np.uint16) for n in (7, 1, 40, 0, 13)]
    expected = list(np.concatenate(parts))
    final = str(tmp_path / "train.bin")

    writer = prepare.ChunkedBinWriter(final, len(expected), total_chunks=len(parts))
    for part in parts:
        writer.write(part)
    writer.finalize()

    assert list(np.memmap(final, dtype=np.uint16, mode="r")) == expected


def test_bin_is_published_only_when_complete(tmp_path):
    """Existence of the final bin must imply completeness — that is the invariant
    bins_ready() and the S3 staging both lean on."""
    final = str(tmp_path / "train.bin")
    writer = prepare.ChunkedBinWriter(final, 100, total_chunks=2)
    writer.write(np.arange(50, dtype=np.uint16))

    assert not os.path.exists(final)
    assert os.path.exists(prepare.partial_path(final))
    assert not prepare.bins_ready(str(tmp_path))

    writer.write(np.arange(50, dtype=np.uint16))
    writer.finalize()
    assert os.path.exists(final)
    # Scratch artifacts are cleaned up on publish.
    assert prepare.partial_leftovers(str(tmp_path)) == []


def test_finalize_refuses_a_short_bin(tmp_path):
    final = str(tmp_path / "train.bin")
    writer = prepare.ChunkedBinWriter(final, 100, total_chunks=2)
    writer.write(np.arange(50, dtype=np.uint16))
    with pytest.raises(ValueError, match="refusing to publish"):
        writer.finalize()
    assert not os.path.exists(final)


def test_chunk_overrunning_the_allocation_is_rejected(tmp_path):
    final = str(tmp_path / "train.bin")
    writer = prepare.ChunkedBinWriter(final, 10, total_chunks=1)
    with pytest.raises(ValueError, match="overruns"):
        writer.write(np.arange(11, dtype=np.uint16))


def test_out_of_range_token_id_is_rejected(tmp_path):
    """uint16 is only safe because GPT-2 BPE tops out at 50256; a wider id must
    raise rather than wrap silently."""
    final = str(tmp_path / "train.bin")
    writer = prepare.ChunkedBinWriter(final, 4, total_chunks=1)
    with pytest.raises(ValueError, match="exceeds uint16"):
        writer.write(np.array([1, 2, 3, 70000], dtype=np.int32))


# --------------------------------------------------------------------------- #
# Resume after an interruption
# --------------------------------------------------------------------------- #
def test_interrupted_run_resumes_at_the_next_chunk(tmp_path):
    tokens = list(range(200))
    parts = _chunks(tokens, 50)
    final = str(tmp_path / "train.bin")

    first = prepare.ChunkedBinWriter(final, len(tokens), total_chunks=len(parts))
    for part in parts[:2]:
        first.write(part)
    del first  # simulate the process dying mid-job

    second = prepare.ChunkedBinWriter(final, len(tokens), total_chunks=len(parts))
    assert second.chunks_written == 2
    assert second.tokens_written == 100
    for part in parts[2:]:
        second.write(part)
    second.finalize()

    assert list(np.memmap(final, dtype=np.uint16, mode="r")) == tokens


def test_progress_sidecar_records_durable_chunks_only(tmp_path):
    final = str(tmp_path / "train.bin")
    writer = prepare.ChunkedBinWriter(final, 30, total_chunks=3)
    writer.write(np.arange(10, dtype=np.uint16))
    with open(prepare.progress_path(final)) as f:
        doc = json.load(f)
    assert doc == {
        "total_tokens": 30,
        "total_chunks": 3,
        "tokens_written": 10,
        "chunks_written": 1,
    }


def test_partial_from_a_different_dataset_restarts_instead_of_stitching(tmp_path):
    """If the token total or chunk count changed, the leftover bytes belong to a
    different corpus — resuming would splice two datasets together."""
    final = str(tmp_path / "train.bin")
    first = prepare.ChunkedBinWriter(final, 100, total_chunks=2)
    first.write(np.full(50, 7, dtype=np.uint16))
    del first

    resized = prepare.ChunkedBinWriter(final, 120, total_chunks=2)
    assert resized.chunks_written == 0 and resized.tokens_written == 0
    assert os.path.getsize(prepare.partial_path(final)) == 120 * prepare.BYTES_PER_TOKEN

    del resized
    rechunked = prepare.ChunkedBinWriter(final, 120, total_chunks=4)
    assert rechunked.chunks_written == 0


def test_truncated_partial_without_progress_restarts(tmp_path):
    """A partial left by a kill *between* the mmap creation and the first flush
    has no sidecar: start over rather than trust unaccounted bytes."""
    final = str(tmp_path / "train.bin")
    with open(prepare.partial_path(final), "wb") as f:
        f.write(b"\x01\x02\x03\x04")

    writer = prepare.ChunkedBinWriter(final, 100, total_chunks=1)
    assert writer.tokens_written == 0
    assert os.path.getsize(prepare.partial_path(final)) == 100 * prepare.BYTES_PER_TOKEN


def test_empty_allocation_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="refusing to preallocate"):
        prepare.ChunkedBinWriter(str(tmp_path / "train.bin"), 0)


# --------------------------------------------------------------------------- #
# Existence / partial detection
# --------------------------------------------------------------------------- #
def test_bins_ready_requires_both_splits_nonempty(tmp_path):
    assert not prepare.bins_ready(str(tmp_path))

    (tmp_path / "train.bin").write_bytes(np.arange(8, dtype=np.uint16).tobytes())
    assert not prepare.bins_ready(str(tmp_path))  # val.bin still missing

    (tmp_path / "val.bin").write_bytes(b"")
    assert not prepare.bins_ready(str(tmp_path))  # empty bin is not a bin

    (tmp_path / "val.bin").write_bytes(np.arange(4, dtype=np.uint16).tobytes())
    assert prepare.bins_ready(str(tmp_path))


def test_odd_sized_bin_is_flagged_not_silently_truncated(tmp_path):
    path = tmp_path / "train.bin"
    path.write_bytes(b"\x00\x01\x02")  # 1.5 tokens
    with pytest.raises(ValueError, match="whole number of uint16 tokens"):
        prepare.token_count(str(path))


def test_partial_leftovers_lists_in_progress_artifacts(tmp_path):
    assert prepare.partial_leftovers(str(tmp_path)) == []
    (tmp_path / "train.bin.partial").write_bytes(b"")
    (tmp_path / "train.bin.progress").write_text("{}")
    (tmp_path / "val.bin").write_bytes(b"")
    assert prepare.partial_leftovers(str(tmp_path)) == [
        "train.bin.partial",
        "train.bin.progress",
    ]


# --------------------------------------------------------------------------- #
# The split-writing loop (HuggingFace Dataset faked to the 3 calls it makes)
# --------------------------------------------------------------------------- #
class _FakeSplit:
    """Stand-in for a tokenized HF Dataset: len(), with_format(), shard()."""

    def __init__(self, docs):
        self.docs = list(docs)

    def __len__(self):
        return len(self.docs)

    def with_format(self, fmt):
        assert fmt == "numpy"
        return self

    def __getitem__(self, column):
        if column == "len":
            return np.array([len(d) for d in self.docs], dtype=np.int64)
        if column == "ids":
            return [np.asarray(d, dtype=np.uint16) for d in self.docs]
        raise KeyError(column)

    def shard(self, num_shards, index, contiguous=True):
        assert contiguous, "the writer relies on contiguous shards to stay in order"
        n = len(self.docs)
        start, end = index * n // num_shards, (index + 1) * n // num_shards
        return _FakeSplit(self.docs[start:end])


class _FakeBar:
    """tqdm's surface as _write_split uses it, with an optional kill switch."""

    def __init__(self, *_, fail_after=None, **__):
        self.updates = 0
        self.fail_after = fail_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, _n):
        self.updates += 1
        if self.fail_after is not None and self.updates > self.fail_after:
            raise KeyboardInterrupt("simulated Ctrl-C mid-write")


def _fake_docs():
    return [list(range(i, i + 3 + (i % 5))) for i in range(37)]


def test_write_split_streams_every_document_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "WRITE_CHUNKS", 4)
    docs = _fake_docs()
    final = str(tmp_path / "train.bin")

    total = prepare._write_split(_FakeSplit(docs), final, _FakeBar)

    expected = [t for doc in docs for t in doc]
    assert total == len(expected)
    assert list(np.memmap(final, dtype=np.uint16, mode="r")) == expected


def test_write_split_resumes_a_killed_run_without_gaps_or_repeats(tmp_path, monkeypatch):
    """The failure this whole design exists for: the job dies partway through
    the 17 GB write, and the re-run must produce the same bin as an uninterrupted
    one — not a short bin, and not one with a chunk written twice."""
    monkeypatch.setattr(prepare, "WRITE_CHUNKS", 8)
    docs = _fake_docs()
    final = str(tmp_path / "train.bin")

    killed = functools.partial(_FakeBar, fail_after=3)
    with pytest.raises(KeyboardInterrupt):
        prepare._write_split(_FakeSplit(docs), final, killed)
    assert not os.path.exists(final)  # nothing published

    prepare._write_split(_FakeSplit(docs), final, _FakeBar)
    assert list(np.memmap(final, dtype=np.uint16, mode="r")) == [t for doc in docs for t in doc]
    assert prepare.partial_leftovers(str(tmp_path)) == []


# --------------------------------------------------------------------------- #
# Split + size arithmetic
# --------------------------------------------------------------------------- #
def test_split_doc_counts_matches_nanogpt_reference():
    """nanoGPT's OWT prep reports 8,009,762 train / 4,007 val documents from
    8,013,769 — our pre-flight estimate must agree or our split diverged."""
    assert prepare.split_doc_counts(8_013_769) == (8_009_762, 4_007)


def test_split_doc_counts_always_holds_out_at_least_one_doc():
    assert prepare.split_doc_counts(100) == (99, 1)
    assert prepare.split_doc_counts(1) == (0, 1)


def test_bin_size_arithmetic_is_two_bytes_per_token(tmp_path):
    """Sanity-checks the numbers quoted in the docstring: ~9.04B tokens ~ 17 GB."""
    assert prepare.BYTES_PER_TOKEN == np.dtype(prepare.DTYPE).itemsize == 2
    owt_train_bytes = 9_035_582_198 * prepare.BYTES_PER_TOKEN
    assert prepare.human_bytes(owt_train_bytes) == "16.8 GB"
    assert prepare.human_bytes(4_434_897 * prepare.BYTES_PER_TOKEN) == "8.5 MB"

    path = tmp_path / "train.bin"
    path.write_bytes(np.arange(1234, dtype=np.uint16).tobytes())
    assert prepare.token_count(str(path)) == 1234


def test_default_num_proc_honours_the_env_override(monkeypatch):
    monkeypatch.setenv("OWT_NUM_PROC", "3")
    assert prepare.default_num_proc() == 3
    monkeypatch.delenv("OWT_NUM_PROC")
    assert prepare.default_num_proc() >= 1


def test_prepare_is_a_no_op_when_bins_exist(tmp_path, monkeypatch, capsys):
    """Idempotence, same contract as the 300M script: re-running must not touch
    HuggingFace at all (importing `datasets` here would blow up the test)."""
    monkeypatch.setattr(prepare, "_HERE", str(tmp_path))
    for split in ("train", "val"):
        (tmp_path / f"{split}.bin").write_bytes(np.arange(4, dtype=np.uint16).tobytes())

    def boom():
        raise AssertionError("main() must not build an encoder when the bins exist")

    monkeypatch.setattr(prepare, "_encoder", boom)
    prepare.main()
    assert "nothing to do" in capsys.readouterr().out
