"""Inference config isolation — the two projects must never read the same var.

Training (us-east-1) and inference (us-east-2) share `OrchestratorConfig` *in
the same process*, so precedence on the generic vars is unsafe: putting
`AWS_REGION=us-east-2` in a shared .env to move inference would silently move
training too. `for_inference()` therefore reads `INFERENCE_*` vars only, and
training's plain constructor is untouched.
"""

from __future__ import annotations

import pytest

from orchestrator.config import OrchestratorConfig

TRAINING_VARS = {
    "AWS_REGION": "us-east-1",
    "SPOT_TRAIN_BUCKET": "training-bucket",
    "IAM_ROLE": "spot-train-role",
    "IAM_PROFILE": "spot-train-profile",
    "SECURITY_GROUP": "spot-train-sg",
}
INFERENCE_VARS = {
    "INFERENCE_REGION": "us-east-2",
    "INFERENCE_BUCKET": "optitrain-inference-us-east-2",
    "INFERENCE_IAM_ROLE": "inference-role",
    "INFERENCE_IAM_PROFILE": "inference-profile",
    "INFERENCE_SECURITY_GROUP": "inference-sg",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(TRAINING_VARS) + list(INFERENCE_VARS) + ["INFERENCE_ALLOW_US_EAST_1"]:
        monkeypatch.delenv(k, raising=False)


def _set(monkeypatch, mapping):
    for k, v in mapping.items():
        monkeypatch.setenv(k, v)


# --- isolation ------------------------------------------------------------


def test_one_env_holding_both_projects_keeps_them_separate(monkeypatch):
    """The case that matters: a single .env carrying both projects' variables."""
    _set(monkeypatch, TRAINING_VARS)
    _set(monkeypatch, INFERENCE_VARS)

    train = OrchestratorConfig()
    infer = OrchestratorConfig.for_inference()

    assert (train.region, infer.region) == ("us-east-1", "us-east-2")
    assert (train.bucket, infer.bucket) == ("training-bucket", "optitrain-inference-us-east-2")
    assert (train.role_name, infer.role_name) == ("spot-train-role", "inference-role")
    assert (train.instance_profile, infer.instance_profile) == (
        "spot-train-profile",
        "inference-profile",
    )
    assert (train.security_group, infer.security_group) == ("spot-train-sg", "inference-sg")


def test_inference_ignores_training_vars_entirely(monkeypatch):
    """Training's vars set, inference's unset => inference falls back to its OWN
    defaults, never to training's values."""
    _set(monkeypatch, TRAINING_VARS)
    cfg = OrchestratorConfig.for_inference()
    assert cfg.region == "us-east-2"  # NOT us-east-1
    assert cfg.bucket == ""  # NOT training-bucket
    assert cfg.role_name == "inference-role"
    assert cfg.instance_profile == "inference-profile"
    assert cfg.security_group == "inference-sg"


def test_training_ignores_inference_vars_entirely(monkeypatch):
    """The reverse, and the more dangerous direction: nothing we add can move
    training out of its region or off its bucket."""
    _set(monkeypatch, INFERENCE_VARS)
    cfg = OrchestratorConfig()
    assert cfg.region == "us-east-1"
    assert cfg.bucket == ""
    assert cfg.role_name == "spot-train-role"
    assert cfg.security_group == "spot-train-sg"


def test_inference_defaults_need_no_env_at_all(monkeypatch):
    cfg = OrchestratorConfig.for_inference()
    assert cfg.region == "us-east-2"
    assert (cfg.role_name, cfg.instance_profile, cfg.security_group) == (
        "inference-role",
        "inference-profile",
        "inference-sg",
    )


# --- the us-east-1 guard --------------------------------------------------


def test_inference_refuses_us_east_1(monkeypatch):
    """A typo must not land the fleet in training's region and race it for GPUs."""
    monkeypatch.setenv("INFERENCE_REGION", "us-east-1")
    with pytest.raises(SystemExit, match="us-east-1"):
        OrchestratorConfig.for_inference()


def test_us_east_1_allowed_with_explicit_override(monkeypatch):
    monkeypatch.setenv("INFERENCE_REGION", "us-east-1")
    monkeypatch.setenv("INFERENCE_ALLOW_US_EAST_1", "yes")
    assert OrchestratorConfig.for_inference().region == "us-east-1"


def test_blank_inference_region_falls_back_to_us_east_2(monkeypatch):
    """An exported-but-empty var must not resolve to the training default."""
    monkeypatch.setenv("INFERENCE_REGION", "")
    assert OrchestratorConfig.for_inference().region == "us-east-2"


# --- bucket requirement ---------------------------------------------------


def test_missing_bucket_names_both_vars(monkeypatch):
    cfg = OrchestratorConfig.for_inference()
    with pytest.raises(SystemExit, match="INFERENCE_BUCKET"):
        cfg.require_bucket()
