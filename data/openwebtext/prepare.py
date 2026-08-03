"""Prepare the FULL OpenWebText corpus (~9B GPT-2 BPE tokens) as nanoGPT bins.

The corpus behind Karpathy's GPT-2-124M reproduction: ~8M documents, ~9.04B
train tokens (``train.bin``, ~17 GB) + ~4.4M val tokens (``val.bin``, ~8.5 MB).
Same recipe as ``third_party/nanoGPT/data/openwebtext/prepare.py`` (GPT-2 BPE via
tiktoken, uint16, one EOT per document, the 0.05%/seed-2357 split) so our loss
curve is comparable to the reference — with this repo's conventions on top:
no ``meta.pkl`` (BPE vocab is fixed; the trainer falls back to 50304), and the
bins are published atomically so an interrupted job can never be mistaken for a
finished one.

**Why this is not just ``openwebtext_300m`` with a bigger cap.** The capped
script accumulates every token in a Python list; at 9B tokens that list alone is
~250 GB of RAM (28 bytes per int + pointer). Here the per-document lengths from
the ``datasets.map`` pass give us the exact total up front, so we preallocate a
``np.memmap`` of the final size and stream ~9M-token chunks into it. Peak RAM for
the write phase is ONE chunk (~18 MB), independent of corpus size.

Cost of running this (one time, locally — no GPU):

- **Disk:** ~54 GB HuggingFace cache for the raw corpus, ~35 GB more for the
  tokenized arrow cache, ~17 GB for the bins. **Keep ~110 GB free.** Only the
  17 GB of bins is permanent — the caches under ``$HF_HOME`` can be deleted
  afterwards (set ``HF_HOME=/some/big/disk`` if ``~`` is small).
- **RAM:** ~8 GB is comfortable. The tokenizer workers dominate (each holds a
  batch of documents); the writer's own peak is one chunk.
- **Runtime:** ~1-2 h wall clock — download ~54 GB (network-bound), tokenize
  (~30-60 min on 8 workers), write the bins (~5-10 min, disk-bound).

Run it:

    pip install datasets tiktoken tqdm numpy
    python data/openwebtext/prepare.py          # OWT_NUM_PROC=<n> to tune workers

Then ``DATASET=openwebtext spot-orchestrate stage-data`` uploads the bins to
``s3://<bucket>/data/openwebtext/`` (multipart; ~17 GB, so budget on your uplink).

**Interruptible.** A run this long gets Ctrl-C'd, OOM-killed, or loses its ssh
session. Tokens land in ``<split>.bin.partial`` and are renamed to
``<split>.bin`` only when the last token is written, so the *existence* of a bin
means it is complete; a ``<split>.bin.progress`` sidecar records how many chunks
are already durable, so re-running resumes at that chunk instead of starting
over (the HF caches make the tokenize pass a cache hit on the second run).

**Boot-time cost of a 17 GB train.bin** (why the streaming data plane in
docs/gpt2-reproduction.md matters): every training box downloads the whole bin
before step 1, and again on every preemption replacement. On a g5.xlarge in the
same region as the bucket, boto3's managed download runs ~10 parallel ranged
GETs and is bottlenecked by the *destination disk*, not the network: a default
gp3 root volume caps at 125 MB/s => **~2.5 min per box per boot**; on the
instance-store NVMe (or a gp3 provisioned past the ~437 MB/s g5.xlarge EBS
ceiling) it is network-bound at **~40-60 s**. Same-region S3 transfer is free
and the ~2.1k GETs cost <$0.01, so the price is wall clock, not dollars.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time

import numpy as np

# GPT-2 BPE tops out at token id 50256, so uint16 (2 bytes/token) is lossless.
DTYPE = np.uint16
BYTES_PER_TOKEN = 2
# nanoGPT's split, reproduced exactly: 0.05% of documents held out, seed 2357.
VAL_FRACTION = 0.0005
SPLIT_SEED = 2357
# Chunks per bin. 9B tokens / 1024 ~= 9M tokens ~= 18 MB per write: big enough
# that the memmap write is sequential and cheap, small enough to stay in RAM and
# to bound the work lost when a run is interrupted mid-chunk.
WRITE_CHUNKS = 1024

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARTIAL_SUFFIX = ".partial"
_PROGRESS_SUFFIX = ".progress"
_SPLITS = ("train", "val")

_ENC = None  # tiktoken encoder, created lazily per process (see _encoder)


# --------------------------------------------------------------------------- #
# Pure helpers (imported by tests — no network, no HuggingFace)
# --------------------------------------------------------------------------- #
def bin_path(dirpath: str, split: str) -> str:
    return os.path.join(dirpath, f"{split}.bin")


def partial_path(final_path: str) -> str:
    """Where bytes accumulate until the bin is complete."""
    return final_path + _PARTIAL_SUFFIX


def progress_path(final_path: str) -> str:
    """Sidecar recording how many chunks of ``final_path`` are durable."""
    return final_path + _PROGRESS_SUFFIX


def token_count(path: str) -> int:
    """Tokens in a uint16 bin. Raises on a size that isn't a whole number of
    tokens — a half-token file is a truncated write, never something to train on."""
    size = os.path.getsize(path)
    if size % BYTES_PER_TOKEN:
        raise ValueError(f"{path}: {size} bytes is not a whole number of uint16 tokens")
    return size // BYTES_PER_TOKEN


def bins_ready(dirpath: str) -> bool:
    """True when both bins are present and non-empty.

    Existence is sufficient *because* of the partial->final rename: a bin only
    appears under its final name once every token is written. A leftover
    ``.partial`` is therefore never mistaken for a usable dataset."""
    for split in _SPLITS:
        path = bin_path(dirpath, split)
        if not os.path.exists(path) or token_count(path) == 0:
            return False
    return True


def partial_leftovers(dirpath: str) -> list[str]:
    """Names of in-progress artifacts from an interrupted run, sorted."""
    if not os.path.isdir(dirpath):
        return []
    return sorted(n for n in os.listdir(dirpath) if n.endswith((_PARTIAL_SUFFIX, _PROGRESS_SUFFIX)))


def split_doc_counts(n_docs: int, val_fraction: float = VAL_FRACTION) -> tuple[int, int]:
    """(train_docs, val_docs) for ``train_test_split(test_size=val_fraction)``.

    Mirrors the sklearn/datasets rule (ceil the test side) so we can print the
    expected shape before a 1-hour job commits to it. For OpenWebText's
    8,013,769 documents this is (8,009,762, 4,007) — nanoGPT's exact numbers."""
    n_val = math.ceil(n_docs * val_fraction)
    return n_docs - n_val, n_val


def human_bytes(n: int) -> str:
    step = 1024.0
    val = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < step or unit == "TB":
            return f"{val:,.1f} {unit}" if unit != "B" else f"{int(val)} B"
        val /= step
    return f"{val:,.1f} TB"  # pragma: no cover — loop always returns


def default_num_proc() -> int:
    """Workers for load_dataset/map. Karpathy's rule of thumb is ~half the cores:
    tokenizing is CPU-bound but each worker also holds a slice of the arrow table."""
    override = os.environ.get("OWT_NUM_PROC")
    if override:
        return max(1, int(override))
    return max(1, (os.cpu_count() or 8) // 2)


# --------------------------------------------------------------------------- #
# Chunked memmap writer
# --------------------------------------------------------------------------- #
class ChunkedBinWriter:
    """Fills a preallocated uint16 memmap in chunks, publishing it atomically.

    The whole reason this class exists: the corpus is ~17 GB of tokens and must
    never be held in RAM. ``total_tokens`` (known from the map pass's per-document
    lengths) sizes the file up front; each :meth:`write` copies one chunk into
    the mapping at the running offset, so peak RAM is one chunk.

    Crash safety follows the repo's atomic-write principle: bytes go to
    ``<name>.partial`` and are renamed onto ``<name>`` only in :meth:`finalize`,
    so a killed job leaves no short bin that a later run could silently train on.
    A ``<name>.progress`` sidecar (written after each chunk is flushed) lets a
    re-run skip the chunks already durable rather than redo an hour of work; the
    total token count and chunk count are recorded with it, and any mismatch
    means the upstream dataset changed, so we start the file over rather than
    stitch two incompatible halves together."""

    def __init__(self, final_path: str, total_tokens: int, total_chunks: int = 0):
        if total_tokens <= 0:
            raise ValueError(f"{final_path}: refusing to preallocate {total_tokens} tokens")
        self.final_path = final_path
        self.partial_path = partial_path(final_path)
        self.progress_path = progress_path(final_path)
        self.total_tokens = int(total_tokens)
        self.total_chunks = int(total_chunks)
        self.tokens_written = 0
        self.chunks_written = 0
        self._arr: np.memmap | None = None
        self._open()

    # -- lifecycle ---------------------------------------------------------- #
    def _open(self) -> None:
        want_bytes = self.total_tokens * BYTES_PER_TOKEN
        prog = self._read_progress()
        resumable = (
            prog is not None
            and prog.get("total_tokens") == self.total_tokens
            and prog.get("total_chunks") == self.total_chunks
            and os.path.exists(self.partial_path)
            and os.path.getsize(self.partial_path) == want_bytes
        )
        if resumable:
            assert prog is not None  # narrowed by `resumable`
            self.tokens_written = int(prog["tokens_written"])
            self.chunks_written = int(prog["chunks_written"])
            mode = "r+"
        else:
            # Stale, absent, or mis-sized partial: we cannot account for the bytes
            # in it, so start clean. Re-tokenizing is cheap (HF cache hit) next to
            # shipping a bin whose middle is someone else's tokens.
            for path in (self.partial_path, self.progress_path):
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(path)
            mode = "w+"
        self._arr = np.memmap(self.partial_path, dtype=DTYPE, mode=mode, shape=(self.total_tokens,))

    def write(self, tokens) -> int:
        """Append one chunk. Returns the number of tokens written so far."""
        if self._arr is None:
            raise RuntimeError(f"{self.final_path}: writer already finalized")
        src = np.asarray(tokens)
        if src.dtype != DTYPE and src.size and int(src.max()) > np.iinfo(DTYPE).max:
            # Guards the uint16 assumption: a tokenizer change (or a stray value)
            # would otherwise wrap silently and poison the corpus.
            raise ValueError(f"{self.final_path}: token id {int(src.max())} exceeds uint16")
        end = self.tokens_written + src.size
        if end > self.total_tokens:
            raise ValueError(
                f"{self.final_path}: chunk of {src.size} tokens overruns the "
                f"preallocated {self.total_tokens} (already wrote {self.tokens_written})"
            )
        if src.size:
            self._arr[self.tokens_written : end] = src.astype(DTYPE, copy=False)
        self.tokens_written = end
        self.chunks_written += 1
        self._flush_progress()
        return self.tokens_written

    def finalize(self) -> int:
        """Publish the bin. Refuses to rename a short file."""
        if self._arr is None:
            raise RuntimeError(f"{self.final_path}: writer already finalized")
        if self.tokens_written != self.total_tokens:
            raise ValueError(
                f"{self.final_path}: refusing to publish {self.tokens_written:,} of "
                f"{self.total_tokens:,} tokens — the write was interrupted"
            )
        self._arr.flush()
        self._arr = None  # drop the mapping before renaming the file under it
        os.replace(self.partial_path, self.final_path)  # atomic on POSIX
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.progress_path)
        return self.total_tokens

    # -- progress sidecar --------------------------------------------------- #
    def _read_progress(self) -> dict | None:
        try:
            with open(self.progress_path) as f:
                return json.load(f)
        except (FileNotFoundError, ValueError):
            return None

    def _flush_progress(self) -> None:
        # Data before bookkeeping: msync the mapping first, so the sidecar never
        # claims tokens that are not on disk yet.
        assert self._arr is not None
        self._arr.flush()
        doc = {
            "total_tokens": self.total_tokens,
            "total_chunks": self.total_chunks,
            "tokens_written": self.tokens_written,
            "chunks_written": self.chunks_written,
        }
        tmp = self.progress_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f)
        os.replace(tmp, self.progress_path)


# --------------------------------------------------------------------------- #
# Tokenization (runs in datasets.map worker processes)
# --------------------------------------------------------------------------- #
def _encoder():
    """One tiktoken encoder per process, built on first use (lazy import keeps
    this module importable — and testable — without tiktoken installed)."""
    global _ENC
    if _ENC is None:
        import tiktoken

        _ENC = tiktoken.get_encoding("gpt2")
    return _ENC


def _tokenize(example: dict) -> dict:
    """One document -> its BPE ids plus a trailing EOT, and the length.

    The length column is what makes the memmap preallocation possible: summing it
    gives the exact token total before we write a single byte."""
    enc = _encoder()
    ids = enc.encode_ordinary(example["text"])  # encode_ordinary ignores specials
    ids.append(enc.eot_token)  # document separator (50256 for the GPT-2 BPE)
    return {"ids": ids, "len": len(ids)}


def _write_split(dset, final_path: str, tqdm) -> int:
    """Stream one tokenized split into its bin. Returns the token count."""
    # numpy format so the length column is a 64 MB int array, not an 8M-element
    # Python list (which would be ~250 MB of boxed ints just to take a sum).
    lengths = dset.with_format("numpy")["len"]
    total = int(np.sum(lengths, dtype=np.uint64))
    chunks = max(1, min(WRITE_CHUNKS, len(dset)))
    writer = ChunkedBinWriter(final_path, total, total_chunks=chunks)
    if writer.chunks_written:
        print(
            f"[prepare] resuming {os.path.basename(final_path)} at chunk "
            f"{writer.chunks_written}/{chunks} ({writer.tokens_written:,} tokens durable)"
        )
    bar = tqdm(
        total=total,
        initial=writer.tokens_written,
        unit="tok",
        unit_scale=True,
        desc=f"writing {os.path.basename(final_path)}",
    )
    with bar:
        for index in range(chunks):
            if index < writer.chunks_written:
                continue  # already durable from an earlier, interrupted run
            batch = dset.shard(num_shards=chunks, index=index, contiguous=True)
            batch = batch.with_format("numpy")
            # np.concatenate rejects an empty list, and an empty chunk still has
            # to be counted or the resume index would drift.
            ids = np.concatenate(batch["ids"]) if len(batch) else np.empty(0, dtype=DTYPE)
            writer.write(ids)
            bar.update(len(ids))
    return writer.finalize()


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def main() -> None:
    started = time.time()
    if bins_ready(_HERE):
        print(f"[prepare] bins already present in {_HERE} — nothing to do")
        _summarize(started)
        return

    leftovers = partial_leftovers(_HERE)
    if leftovers:
        print(f"[prepare] found in-progress artifacts from an earlier run: {leftovers}")

    # Imported here (not at module scope) so the pure helpers above stay testable
    # without the heavyweight data stack installed.
    from datasets import load_dataset
    from tqdm import tqdm

    _encoder()  # fail fast on a missing/broken tiktoken, before the 54 GB download

    num_proc = default_num_proc()
    print(f"[prepare] loading OpenWebText (~54 GB into the HF cache), num_proc={num_proc}")
    # "Skylion007/openwebtext" is the live copy; the bare "openwebtext" alias that
    # nanoGPT's script uses was retired from the Hub.
    dataset = load_dataset("Skylion007/openwebtext", num_proc=num_proc)

    n_docs = len(dataset["train"])
    expect_train, expect_val = split_doc_counts(n_docs)
    print(f"[prepare] {n_docs:,} documents -> {expect_train:,} train / {expect_val:,} val")
    split = dataset["train"].train_test_split(test_size=VAL_FRACTION, seed=SPLIT_SEED, shuffle=True)
    split["val"] = split.pop("test")  # nanoGPT names the held-out split "val"

    tokenized = split.map(
        _tokenize,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for name, dset in tokenized.items():
        final = bin_path(_HERE, name)
        if os.path.exists(final):
            print(f"[prepare] {name}.bin already written — skipping")
            continue
        written = _write_split(dset, final, tqdm)
        print(f"[prepare] wrote {written:,} {name} tokens -> {final}")

    _summarize(started)


def _summarize(started: float) -> None:
    parts = []
    for split in _SPLITS:
        path = bin_path(_HERE, split)
        if os.path.exists(path):
            parts.append(
                f"{split}.bin {token_count(path):,} tokens "
                f"({human_bytes(os.path.getsize(path))})"
            )
    print(f"[prepare] {' | '.join(parts)} | {time.time() - started:.0f}s elapsed")
    print("[prepare] next: DATASET=openwebtext spot-orchestrate stage-data")


if __name__ == "__main__":
    main()
