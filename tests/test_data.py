"""Dataset-provisioning tests — hermetic (no S3; transfers are monkeypatched).

Pins the multi-rank download contract on a box: only LOCAL_RANK 0 pulls from
S3 and lands each file atomically (temp + os.replace), while other local ranks
wait for the files to appear instead of racing to overwrite them. Also pins the
properties that only matter at OpenWebText scale (a ~17 GB train.bin): the
download's temp file stays on the destination filesystem, and ``stage-data``
skips per file so a resumed staging never re-uploads a bin that already landed.
"""

from __future__ import annotations

import os

import pytest

from orchestrator import dataset as dataset_mod
from spot_train import data as data_mod


def _make_loader(tmp_path, monkeypatch, local_rank: str | None):
    """Build a PositionedLoader far enough to run _ensure_data, no more."""
    if local_rank is None:
        monkeypatch.delenv("LOCAL_RANK", raising=False)
    else:
        monkeypatch.setenv("LOCAL_RANK", local_rank)
    loader = data_mod.PositionedLoader.__new__(data_mod.PositionedLoader)
    loader.data_local_dir = str(tmp_path / "ds")
    loader.data_uri = "s3://bucket/data/ds"
    return loader


def test_rank0_downloads_atomically(tmp_path, monkeypatch):
    staged = tmp_path / "staged"
    staged.mkdir()

    def fake_download(ref, dest_dir=""):
        name = ref.rsplit("/", 1)[-1]
        p = staged / name
        p.write_bytes(b"payload:" + name.encode())
        return str(p)

    monkeypatch.setattr(data_mod.s3_store, "download", fake_download)
    monkeypatch.setattr(data_mod.s3_store, "exists", lambda ref: True)
    loader = _make_loader(tmp_path, monkeypatch, "0")
    loader._ensure_data()
    for name in data_mod._FILES:
        dest = os.path.join(loader.data_local_dir, name)
        assert os.path.exists(dest)
        with open(dest, "rb") as f:
            assert f.read() == b"payload:" + name.encode()
    # No temp debris left behind (everything was os.replace'd into place).
    assert all(".tmp-" not in n for n in os.listdir(loader.data_local_dir))


def test_nonzero_local_rank_waits_instead_of_downloading(tmp_path, monkeypatch):
    def must_not_download(ref):
        raise AssertionError("non-zero local rank must never download")

    monkeypatch.setattr(data_mod.s3_store, "download", must_not_download)
    loader = _make_loader(tmp_path, monkeypatch, "1")
    os.makedirs(loader.data_local_dir)

    # Files already present (rank 0 finished first): returns without downloading.
    for name in data_mod._FILES:
        with open(os.path.join(loader.data_local_dir, name), "wb") as f:
            f.write(b"x")
    loader._ensure_data()

    # Files absent and rank 0 never delivers: bounded wait, then a clear error.
    for name in data_mod._FILES:
        os.unlink(os.path.join(loader.data_local_dir, name))
    with pytest.raises(TimeoutError, match="rank 0"):
        loader._wait_for_files(list(data_mod._FILES), timeout=0.1)


def test_single_process_still_downloads_without_local_rank(tmp_path, monkeypatch):
    calls = []

    def fake_download(ref, dest_dir=""):
        name = ref.rsplit("/", 1)[-1]
        calls.append(name)
        p = tmp_path / name
        p.write_bytes(b"x")
        return str(p)

    monkeypatch.setattr(data_mod.s3_store, "download", fake_download)
    monkeypatch.setattr(data_mod.s3_store, "exists", lambda ref: True)
    loader = _make_loader(tmp_path, monkeypatch, None)  # LOCAL_RANK unset
    loader._ensure_data()
    assert sorted(calls) == sorted(data_mod._FILES)


def test_missing_optional_meta_is_skipped_not_downloaded(tmp_path, monkeypatch):
    """BPE datasets (OpenWebText) ship no meta.pkl. The box must fetch the
    required bins and skip the un-staged meta.pkl cleanly — not 404 on it."""
    calls = []

    def fake_download(ref, dest_dir=""):
        name = ref.rsplit("/", 1)[-1]
        calls.append(name)
        p = tmp_path / name
        p.write_bytes(b"x")
        return str(p)

    # meta.pkl is not in S3; train/val are.
    def fake_exists(ref):
        return not ref.endswith("meta.pkl")

    monkeypatch.setattr(data_mod.s3_store, "download", fake_download)
    monkeypatch.setattr(data_mod.s3_store, "exists", fake_exists)
    loader = _make_loader(tmp_path, monkeypatch, "0")
    loader._ensure_data()  # must not raise

    assert sorted(calls) == sorted(data_mod._REQUIRED)  # meta.pkl never fetched
    for name in data_mod._REQUIRED:
        assert os.path.exists(os.path.join(loader.data_local_dir, name))
    assert not os.path.exists(os.path.join(loader.data_local_dir, "meta.pkl"))


def test_download_temp_lands_in_the_destination_dir(tmp_path, monkeypatch):
    """At OpenWebText scale the bin is ~17 GB: if the temp lands in $TMPDIR on
    another filesystem, the move into place becomes a second full copy (minutes
    of boot time, 2x the free space). Rank 0 must ask for a same-dir temp."""
    seen = {}

    def fake_download(ref, dest_dir=""):
        seen[ref.rsplit("/", 1)[-1]] = dest_dir
        # Honour dest_dir the way s3_store does, so the move stays a rename.
        p = os.path.join(dest_dir, "tmp-" + ref.rsplit("/", 1)[-1])
        with open(p, "wb") as f:
            f.write(b"x")
        return p

    monkeypatch.setattr(data_mod.s3_store, "download", fake_download)
    monkeypatch.setattr(data_mod.s3_store, "exists", lambda ref: True)
    loader = _make_loader(tmp_path, monkeypatch, "0")
    loader._ensure_data()

    assert set(seen) == set(data_mod._FILES)
    assert set(seen.values()) == {loader.data_local_dir}


def test_wait_timeout_is_sized_for_a_multi_gb_corpus():
    """A 17 GB pull can outlast the old 600 s ceiling; non-rank-0 ranks must not
    fail the box while rank 0 is still legitimately downloading."""
    assert data_mod._WAIT_TIMEOUT_SECONDS >= 1800


# --------------------------------------------------------------------------- #
# stage-data (upload side)
# --------------------------------------------------------------------------- #
class _FakeAws:
    """Records uploads and answers HeadObject-style size queries from a dict."""

    def __init__(self, sizes: dict[str, int]):
        self.sizes = sizes
        self.uploaded: list[str] = []

    def object_exists(self, bucket, key):
        return key in self.sizes

    def object_size(self, bucket, key):
        return self.sizes.get(key)

    def upload_file(self, path, bucket, key):
        self.uploaded.append(key)
        self.sizes[key] = os.path.getsize(path)


def _staged_cfg(tmp_path, monkeypatch, sizes):
    from orchestrator.config import OrchestratorConfig

    data_dir = tmp_path / "data" / "ds"
    data_dir.mkdir(parents=True)
    (data_dir / "prepare.py").write_text("")
    (data_dir / "train.bin").write_bytes(b"t" * 4096)
    (data_dir / "val.bin").write_bytes(b"v" * 64)

    fake = _FakeAws(sizes)
    monkeypatch.setattr(dataset_mod, "aws", fake)
    monkeypatch.setattr(dataset_mod, "_local_dir", lambda cfg: str(data_dir))
    cfg = OrchestratorConfig(bucket="bkt", dataset="ds")
    return cfg, fake


def test_stage_data_skips_the_file_already_in_s3(tmp_path, monkeypatch):
    """The interrupted-staging case: train.bin (17 GB in real life) landed, then
    the upload died before val.bin. Re-running must send only val.bin."""
    cfg, fake = _staged_cfg(tmp_path, monkeypatch, {"data/ds/train.bin": 4096})
    dataset_mod.stage_data(cfg)
    assert fake.uploaded == ["data/ds/val.bin"]


def test_stage_data_reuploads_when_the_size_disagrees(tmp_path, monkeypatch):
    """A remote object of a different size is a differently-prepared corpus, not
    a finished upload — training on the mix would be silent garbage."""
    cfg, fake = _staged_cfg(tmp_path, monkeypatch, {"data/ds/train.bin": 999})
    dataset_mod.stage_data(cfg)
    assert fake.uploaded == ["data/ds/train.bin", "data/ds/val.bin"]


def test_stage_data_is_a_no_op_when_both_bins_are_present(tmp_path, monkeypatch):
    cfg, fake = _staged_cfg(
        tmp_path, monkeypatch, {"data/ds/train.bin": 4096, "data/ds/val.bin": 64}
    )
    dataset_mod.stage_data(cfg)
    assert fake.uploaded == []
