"""A supervisor restart must not replay the chaos schedule.

``_fired_kills`` and ``_train_start`` were process memory, and ``_train_start``
re-armed to the restart moment -- so after ANY supervisor restart the whole
PREEMPT_SCHEDULE fired again from zero, at its original offsets. On the 24h run
a restart at hour 8 would re-fire the mass loss of 6 of 8, and there is no live
knob to stop it because the schedule is baked into user-data.

The subtle half is the CLOCK. ``_train_start`` is a ``time.monotonic()`` reading
and monotonic resets with the process, which is precisely the bug -- so the WALL
start is what gets persisted, and the monotonic field is reconstructed by
offsetting it. These tests pin that arithmetic, because getting it wrong fails in
the most expensive direction: every entry looks due at once.
"""

from __future__ import annotations

import json

import pytest

from orchestrator import supervisor as sup


class _FakeAws:
    """Just enough of the aws module for save/restore, with no network."""

    def __init__(self, doc=None):
        self.doc = doc
        self.puts: list[str] = []

    def get_json(self, _bucket, _key):
        return self.doc

    def put_text(self, _bucket, _key, body):
        self.puts.append(body)
        self.doc = json.loads(body)


class _Sup:
    """A Supervisor with only the fields the schedule code touches."""

    _save_schedule = sup.Supervisor._save_schedule
    _restore_schedule = sup.Supervisor._restore_schedule

    def __init__(self, schedule, *, fired=None, train_start=None, wall=0.0):
        self.cfg = type("Cfg", (), {"bucket": "b", "run_schedule_key": lambda self, r: "sched"})()
        self.run_id = "run-1"
        self.kill_schedule = schedule
        self._fired_kills = set(fired or ())
        self._train_start = train_start
        self._train_start_wall = wall
        self.events: list[str] = []

    def _event(self, msg):
        self.events.append(msg)


SCHEDULE = [(480.0, 3), (960.0, 1), (1440.0, 4), (2400.0, 0)]


@pytest.fixture
def fake(monkeypatch):
    f = _FakeAws()
    monkeypatch.setattr(sup, "aws", f)
    return f


def test_save_records_fired_entries_and_the_wall_start(fake):
    s = _Sup(SCHEDULE, fired={0, 1}, train_start=100.0, wall=1_786_000_000.0)
    s._save_schedule(1_786_000_500.0)
    assert fake.doc["fired"] == [0, 1]
    assert fake.doc["train_start_wall"] == 1_786_000_000.0


def test_restore_does_not_replay_spent_entries(fake):
    """The whole point: entries already fired stay fired."""
    fake.doc = {"train_start_wall": 1_786_000_000.0, "fired": [0, 1]}
    s = _Sup(SCHEDULE)
    # Restart 1500s after training began.
    s._restore_schedule(now=5_000.0, wall=1_786_001_500.0)
    assert s._fired_kills == {0, 1}


def test_restore_rebuilds_the_monotonic_clock_from_the_wall_clock(fake):
    """elapsed must measure time since TRAINING began, not since this process
    started. Get this wrong and every remaining entry fires on the first tick."""
    fake.doc = {"train_start_wall": 1_786_000_000.0, "fired": [0]}
    s = _Sup(SCHEDULE)
    s._restore_schedule(now=5_000.0, wall=1_786_001_500.0)
    # 1500s of real training happened before the restart.
    elapsed = 5_000.0 - s._train_start
    assert elapsed == pytest.approx(1500.0)
    # So entry 1 (960s) is due, entry 2 (1440s) is due, entry 3 (2400s) is NOT.
    due = [i for i, (secs, _v) in enumerate(SCHEDULE) if elapsed >= secs]
    assert due == [0, 1, 2]
    pending = [i for i in due if i not in s._fired_kills]
    assert pending == [1, 2], "entries whose time passed during the outage still fire"


def test_a_cold_start_is_untouched(fake):
    """No document => nothing has fired and training has not started."""
    fake.doc = None
    s = _Sup(SCHEDULE)
    s._restore_schedule(now=10.0, wall=1_786_000_000.0)
    assert s._fired_kills == set()
    assert s._train_start is None


def test_a_malformed_document_does_not_crash_the_restart(fake):
    """A truncated write must degrade to a cold start, not take down the control
    plane on boot."""
    fake.doc = {"train_start_wall": "nope", "fired": None}
    s = _Sup(SCHEDULE)
    s._restore_schedule(now=10.0, wall=1_786_000_000.0)
    assert s._fired_kills == set()
    assert s._train_start is None


def test_save_before_training_starts_is_a_noop(fake):
    """No train start => nothing meaningful to record; do not write a document
    that would later restore a bogus clock."""
    s = _Sup(SCHEDULE, train_start=None)
    s._save_schedule(1_786_000_000.0)
    assert fake.puts == []


def test_save_failure_never_propagates(fake, monkeypatch):
    """Losing this costs a replayed schedule; raising costs the run."""

    def boom(*_a, **_k):
        raise RuntimeError("s3 down")

    monkeypatch.setattr(fake, "put_text", boom)
    s = _Sup(SCHEDULE, train_start=1.0, wall=1_786_000_000.0)
    s._save_schedule(1_786_000_100.0)  # must not raise


def test_restore_reports_what_it_re_armed(fake):
    fake.doc = {"train_start_wall": 1_786_000_000.0, "fired": [0, 1]}
    s = _Sup(SCHEDULE)
    s._restore_schedule(now=5_000.0, wall=1_786_001_500.0)
    assert s.events and "re-armed kill schedule" in s.events[0]
    assert "2 of 4 already fired" in s.events[0]
