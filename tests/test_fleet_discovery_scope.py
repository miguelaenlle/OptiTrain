"""`fleet down` must not reach past its own project or its own fleet.

Discovery used to match on `fleet_role` alone, which meant every fleet in the
region — so tearing down one experiment would terminate another that happened to
be running. These pin both guards.
"""

from __future__ import annotations

import pytest

from orchestrator import fleet
from orchestrator.config import OrchestratorConfig


def _inst(iid, role, project, fleet_id):
    return {
        "id": iid,
        "state": "running",
        "type": "g5.xlarge",
        "public_ip": "",
        "private_ip": "10.0.0.1",
        "tags": {"project": project, fleet.ROLE_TAG: role, fleet.FLEET_TAG: fleet_id},
    }


@pytest.fixture
def fake_aws(monkeypatch):
    """Two inference fleets plus a stray box mislabelled as training.

    Patches the FUNCTION, not the module. `_discover` does `from . import aws`
    at call time, which resolves through the package attribute once anything
    else has imported it — so substituting sys.modules["orchestrator.aws"]
    works in isolation and silently stops working as soon as another test file
    imports aws first. Patching the attribute is order-independent.
    """
    everything = [
        _inst("i-mine-w", "worker", "inference", "fleet-A"),
        _inst("i-mine-r", "router", "inference", "fleet-A"),
        _inst("i-other-w", "worker", "inference", "fleet-B"),
        _inst("i-train-w", "worker", "spot-train", "fleet-T"),
    ]

    def fake_instances_by_tag(key, value):
        return [i for i in everything if i["tags"].get(key) == value]

    monkeypatch.setattr("orchestrator.aws.instances_by_tag", fake_instances_by_tag)
    return everything


def test_project_scope_excludes_training_instances(fake_aws):
    """Even a box wrongly carrying fleet_role must be ignored if project differs."""
    cfg = OrchestratorConfig.for_inference()
    workers, routers = fleet._discover(cfg)
    ids = {w["id"] for w in workers} | {r["id"] for r in routers}
    assert "i-train-w" not in ids
    assert ids == {"i-mine-w", "i-mine-r", "i-other-w"}


def test_fleet_id_scope_isolates_one_fleet(fake_aws):
    """The regression that matters: tearing down fleet-A must leave fleet-B up."""
    cfg = OrchestratorConfig.for_inference()
    workers, routers = fleet._discover(cfg, fleet_id="fleet-A")
    assert {w["id"] for w in workers} == {"i-mine-w"}
    assert {r["id"] for r in routers} == {"i-mine-r"}


def test_training_config_never_sees_inference_instances(fake_aws):
    """The reverse direction: a training teardown must not reap inference boxes."""
    cfg = OrchestratorConfig()  # project_tag == "spot-train"
    workers, _ = fleet._discover(cfg)
    assert {w["id"] for w in workers} == {"i-train-w"}


def test_inference_config_stamps_the_inference_project_tag():
    assert OrchestratorConfig.for_inference().project_tag == "inference"
    assert OrchestratorConfig().project_tag == "spot-train"
