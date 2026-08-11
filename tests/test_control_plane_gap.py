"""A control-plane outage must be stated, not left blank.

When the supervisor stops writing status ticks, world size and the Gantt stop
CHANGING -- which is pixel-identical to "the fleet was stable", the more
dangerous of the two readings. So the gap is detected and rendered as an explicit
band plus a `supervisor_up` series.

The hard part is not detecting silence, it is knowing which silence counts.
Measured on the 8-node rehearsal (`multinode-preempt-1786239456`, 483 real
ticks): median 4.1s, p90 4.4s -- but three gaps of 21s, 21s and 38s, every one a
KILL, because the supervisor blocks inside aws.terminate + wait_quota_released
while it happens. The mass-loss event terminates six boxes at once and blocks
longest of all, so a threshold tuned to the median would shade "control plane
down" straight across the headline result.

Hence the 90s floor, and hence these tests: the false-positive case matters more
than the true-positive one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXPORT = Path(__file__).resolve().parents[1] / "deploy" / "grafana" / "export.py"


def _mod():
    spec = importlib.util.spec_from_file_location("gexport", EXPORT)
    m = importlib.util.module_from_spec(spec)
    sys.argv = ["export.py", "run", "--live", "--nodes=8"]
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def gx():
    return _mod()


def _hist(times):
    return [(t, {}) for t in times]


def _ticks(start, count, step=4.1):
    return [start + i * step for i in range(count)]


# --------------------------------------------------------------------------- #
# False positives — the expensive direction
# --------------------------------------------------------------------------- #
def test_a_steady_run_has_no_gaps(gx):
    assert gx.control_plane_gaps(_hist(_ticks(1000.0, 200))) == []


def test_a_38s_blocking_kill_is_NOT_an_outage(gx):
    """The exact pause observed during the rehearsal's simultaneous 2-node kill.
    Flagging this would put a control-plane band over a fault-tolerance event."""
    times = _ticks(1000.0, 50) + _ticks(1000.0 + 50 * 4.1 + 38.2, 50)
    assert gx.control_plane_gaps(_hist(times)) == []


def test_a_mass_loss_style_60s_block_is_NOT_an_outage(gx):
    """Six boxes terminated at once blocks longest of all. Still not an outage --
    the supervisor is alive and resumes ticking on its own."""
    times = _ticks(1000.0, 50) + _ticks(1000.0 + 50 * 4.1 + 60.0, 50)
    assert gx.control_plane_gaps(_hist(times)) == []


# --------------------------------------------------------------------------- #
# True positives
# --------------------------------------------------------------------------- #
def test_a_multi_minute_box_failure_IS_an_outage(gx):
    """A dead BOX is gone until someone relaunches it -- minutes, not seconds.
    This is the Step 3 scenario."""
    times = _ticks(1000.0, 50) + _ticks(1000.0 + 50 * 4.1 + 300.0, 50)
    gaps = gx.control_plane_gaps(_hist(times))
    assert len(gaps) == 1
    assert gaps[0][1] - gaps[0][0] == pytest.approx(300.0, abs=5.0)


def test_supervisor_up_is_zero_only_inside_the_gap(gx):
    gaps = [(2000.0, 2300.0)]
    assert gx._in_gap(1999.0, gaps) is False
    assert gx._in_gap(2150.0, gaps) is True
    assert gx._in_gap(2301.0, gaps) is False


def test_regions_carry_the_duration_and_say_training_continued(gx, tmp_path):
    out = tmp_path / "cp.json"
    n = gx.write_control_plane_down([(1000.0, 1300.0)], out)
    import json

    doc = json.loads(out.read_text())
    assert n == 1
    assert doc[0]["time"] == 1_000_000 and doc[0]["timeEnd"] == 1_300_000
    assert "300s" in doc[0]["text"] and "training continued" in doc[0]["text"]
    # A DISTINCT tag from the degraded-world band: "short a node" and "nobody is
    # steering" are different failures and must not share a colour.
    assert doc[0]["tags"] == ["control-plane"]


# --------------------------------------------------------------------------- #
# Against the real rehearsal data
# --------------------------------------------------------------------------- #
REHEARSAL = (
    Path(__file__).resolve().parents[1]
    / "deploy"
    / "grafana"
    / ".live"
    / "multinode-preempt-1786239456"
    / "status"
)


@pytest.mark.skipif(not REHEARSAL.is_dir(), reason="rehearsal status objects not present")
def test_the_real_rehearsal_reports_no_outage(gx):
    """The supervisor was SIGKILLed in that run, but systemd restarted it inside
    the floor, and every other pause was a kill. So: no band. If this ever starts
    failing, the detector has drifted back toward flagging blocking calls."""
    times = sorted(int(p.name[:-5]) / 1000 for p in REHEARSAL.glob("*.json"))
    assert len(times) > 100, "expected the real tick history"
    assert gx.control_plane_gaps(_hist(times)) == []


@pytest.mark.skipif(not REHEARSAL.is_dir(), reason="rehearsal status objects not present")
def test_deleting_a_window_from_real_ticks_produces_exactly_that_band(gx):
    """The $0 verification promised in the plan: excise a contiguous window from
    the run's OWN tick history and confirm the detector recovers precisely it."""
    times = sorted(int(p.name[:-5]) / 1000 for p in REHEARSAL.glob("*.json"))
    lo, hi = len(times) // 3, len(times) // 3 + 60  # ~4 minutes of ticks
    cut_from, cut_to = times[lo - 1], times[hi]
    kept = times[:lo] + times[hi:]
    gaps = gx.control_plane_gaps(_hist(kept))
    assert len(gaps) == 1
    assert gaps[0] == pytest.approx((cut_from, cut_to), abs=0.01)
