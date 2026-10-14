"""Record a run's snapshot, diff it into events, keep history bounded, read events.

:func:`record_snapshot` runs once per index run, after every store is final.
It snapshots the components, and when the previous run carries the same stamp
it stores the typed occurrences between the two and hands them to the
configured sinks. A stamp change (a new analyzer, parser, git-history or
contract-extraction version) starts a fresh baseline instead: figures computed
under different rules are not comparable, so their difference is not an event.

Settings, under ``events:`` in ``.repowise/config.yaml``::

    events:
      enabled: true            # snapshots and events at all
      retention_days: 365      # runs and events older than this are dropped
      compact_after_days: 30   # older runs keep one per ISO week
      kinds: {new_hotspot: true}
      sinks: [jsonl, stdout]   # names in the event-sink seam
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select

from .diff import diff_snapshots
from .model import EVENT_SCHEMA_VERSION, KINDS, EngineEvent, event_id, severity_for

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EventSettings:
    enabled: bool = True
    retention_days: int = 365
    compact_after_days: int = 30
    kinds: frozenset[str] = frozenset(k.name for k in KINDS.values() if k.default_on)
    sinks: tuple[str, ...] = ()


def load_event_settings(repo_root: Path | str) -> EventSettings:
    """The repo's ``events:`` settings; a missing or invalid value keeps its default."""
    from repowise.core.repo_config import load_repo_config

    try:
        raw = load_repo_config(repo_root).get("events")
    except Exception:
        raw = None
    if not isinstance(raw, dict):
        return EventSettings()
    base = EventSettings()
    kinds = set(base.kinds)
    for name, on in (raw.get("kinds") or {}).items() if isinstance(raw.get("kinds"), dict) else ():
        if name in KINDS and isinstance(on, bool):
            (kinds.add if on else kinds.discard)(name)
    sinks = raw.get("sinks")
    return EventSettings(
        enabled=raw.get("enabled") is not False,
        retention_days=_positive(raw.get("retention_days"), base.retention_days),
        compact_after_days=_positive(raw.get("compact_after_days"), base.compact_after_days),
        kinds=frozenset(kinds),
        sinks=tuple(str(s) for s in sinks) if isinstance(sinks, list) else (),
    )


def _positive(value: Any, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else default


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


async def record_snapshot(
    session: Any, repository_id: str, *, now: datetime | None = None
) -> list[EngineEvent]:
    """Snapshot the repository's components and return the events stored for it.

    A run at the commit and stamp of the newest run is skipped (an update that
    found nothing to do). Sinks are called before the caller commits; their
    events carry stable ids, so a retry after a failed commit re-delivers the
    same ids rather than new events.
    """
    from repowise.core.attribution import attribution_enabled
    from repowise.core.persistence.models import ComponentSnapshot, Repository, SnapshotRun

    from .snapshot import collect_facts, current_stamp

    repo = await session.get(Repository, repository_id)
    if repo is None or not repo.local_path:
        return []
    root = Path(repo.local_path)
    settings = load_event_settings(root)
    if not settings.enabled:
        return []
    stamp = current_stamp()
    head = repo.head_commit
    previous = await session.scalar(
        select(SnapshotRun)
        .where(SnapshotRun.repository_id == repository_id)
        .order_by(SnapshotRun.taken_at.desc(), SnapshotRun.id.desc())
        .limit(1)
    )
    if previous is not None and head and previous.commit_sha == head and previous.stamp == stamp:
        return []

    facts = await collect_facts(
        session, repository_id, root, head_commit=head, attribution=attribution_enabled(root)
    )
    taken_at = now or datetime.now(UTC)
    run = SnapshotRun(repository_id=repository_id, taken_at=taken_at, commit_sha=head, stamp=stamp)
    session.add(run)
    await session.flush()
    session.add_all(
        ComponentSnapshot(run_id=run.id, component=name, facts_json=json.dumps(f, sort_keys=True))
        for name, f in facts.items()
    )

    events: list[EngineEvent] = []
    if previous is not None and previous.stamp == stamp:
        before = await _run_facts(session, previous.id)
        events = await _store_events(
            session, repo, run, previous, before, facts, settings, taken_at
        )
    await compact(session, repository_id, settings, now=taken_at)
    await session.flush()
    if events:
        await _deliver(events, settings, root)
    return events


async def _run_facts(session: Any, run_id: str) -> dict[str, dict[str, Any]]:
    from repowise.core.persistence.models import ComponentSnapshot

    rows = await session.execute(
        select(ComponentSnapshot.component, ComponentSnapshot.facts_json).where(
            ComponentSnapshot.run_id == run_id
        )
    )
    return {name: json.loads(raw) for name, raw in rows}


async def _store_events(
    session: Any,
    repo: Any,
    run: Any,
    previous: Any,
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    settings: EventSettings,
    taken_at: datetime,
) -> list[EngineEvent]:
    from repowise.core.persistence.models import Change, EngineEventRecord

    occurrences = [o for o in diff_snapshots(before, after) if o[0] in settings.kinds]
    if not occurrences:
        return []
    vcs: dict[str, Any] = {}
    if repo.url:
        vcs["vcs.repository.url.full"] = repo.url
    if run.commit_sha:
        change_id = await session.scalar(
            select(Change.change_id).where(
                Change.repository_id == repo.id, Change.head_sha == run.commit_sha
            )
        )
        if change_id:
            vcs["vcs.change.id"] = change_id
    events = [
        EngineEvent(
            id=event_id(repo.id, previous.commit_sha, run.commit_sha, kind, component, subject),
            kind=kind,
            repo_id=repo.id,
            component=component,
            severity=severity_for(after[component].get("criticality")),
            occurred_at=taken_at,
            evidence=evidence,
            stamp=run.stamp,
            commit_sha=run.commit_sha,
            base_commit_sha=previous.commit_sha,
            source=repo.url or "",
            vcs=vcs,
        )
        for kind, component, subject, evidence in occurrences
    ]
    seen = set(
        await session.scalars(
            select(EngineEventRecord.id).where(EngineEventRecord.id.in_([e.id for e in events]))
        )
    )
    events = [e for e in events if e.id not in seen]
    session.add_all(
        EngineEventRecord(
            id=e.id,
            repository_id=e.repo_id,
            kind=e.kind,
            component=e.component,
            severity=e.severity,
            occurred_at=e.occurred_at,
            commit_sha=e.commit_sha,
            base_commit_sha=e.base_commit_sha,
            stamp=e.stamp,
            payload_json=json.dumps(
                {"evidence": dict(e.evidence), "vcs": dict(e.vcs), "source": e.source},
                sort_keys=True,
            ),
            schema_version=e.schema,
        )
        for e in events
    )
    return events


async def _deliver(events: Sequence[EngineEvent], settings: EventSettings, root: Path) -> None:
    from .sinks import event_sinks

    for name in settings.sinks or (event_sinks.default or "jsonl",):
        try:
            await event_sinks.get(name).deliver(events, repo_root=root)
        except Exception as exc:  # the store keeps them; `repowise events` replays
            logger.warning("event sink %s failed: %s", name, exc)


async def compact(
    session: Any, repository_id: str, settings: EventSettings, *, now: datetime
) -> None:
    """Drop runs and events past retention; keep one run per ISO week past
    ``compact_after_days``. The newest run is always kept: the next diff reads it."""
    from repowise.core.persistence.models import ComponentSnapshot, EngineEventRecord, SnapshotRun

    now = _aware(now)
    drop_before = now - timedelta(days=settings.retention_days)
    weekly_before = now - timedelta(days=settings.compact_after_days)
    runs = (
        await session.execute(
            select(SnapshotRun.id, SnapshotRun.taken_at)
            .where(SnapshotRun.repository_id == repository_id)
            .order_by(SnapshotRun.taken_at.desc(), SnapshotRun.id.desc())
        )
    ).all()
    doomed: list[str] = []
    kept_weeks: dict[tuple[int, int], str] = {}
    for i, (run_id, taken_at) in enumerate(runs):
        taken_at = _aware(taken_at)
        if i == 0 or taken_at >= weekly_before:
            continue
        if taken_at < drop_before:
            doomed.append(run_id)
            continue
        week = taken_at.isocalendar()[:2]
        if week in kept_weeks:
            doomed.append(run_id)
        else:
            kept_weeks[week] = run_id  # newest first, so the week's last run
    if kept_weeks:
        await _mark_compacted(session, kept_weeks.values())
    for i in range(0, len(doomed), 500):
        chunk = doomed[i : i + 500]
        await session.execute(delete(ComponentSnapshot).where(ComponentSnapshot.run_id.in_(chunk)))
        await session.execute(delete(SnapshotRun).where(SnapshotRun.id.in_(chunk)))
    await session.execute(
        delete(EngineEventRecord).where(
            EngineEventRecord.repository_id == repository_id,
            EngineEventRecord.occurred_at < drop_before,
        )
    )


async def _mark_compacted(session: Any, run_ids: Iterable[str]) -> None:
    from sqlalchemy import update

    from repowise.core.persistence.models import SnapshotRun

    await session.execute(
        update(SnapshotRun).where(SnapshotRun.id.in_(list(run_ids))).values(compacted=True)
    )


def event_from_row(row: Any) -> EngineEvent:
    payload = json.loads(row.payload_json or "{}")
    return EngineEvent(
        id=row.id,
        kind=row.kind,
        repo_id=row.repository_id,
        component=row.component,
        severity=row.severity,
        occurred_at=_aware(row.occurred_at),
        evidence=payload.get("evidence") or {},
        stamp=row.stamp,
        commit_sha=row.commit_sha,
        base_commit_sha=row.base_commit_sha,
        source=payload.get("source") or "",
        vcs=payload.get("vcs") or {},
        schema=row.schema_version or EVENT_SCHEMA_VERSION,
    )


async def list_events(
    session: Any,
    repository_id: str,
    *,
    since: datetime | None = None,
    kinds: Sequence[str] = (),
    limit: int = 200,
) -> list[EngineEvent]:
    """Stored events, oldest first, from *since* on."""
    from repowise.core.persistence.models import EngineEventRecord

    query = select(EngineEventRecord).where(EngineEventRecord.repository_id == repository_id)
    if since is not None:
        query = query.where(EngineEventRecord.occurred_at >= since)
    if kinds:
        query = query.where(EngineEventRecord.kind.in_(list(kinds)))
    query = query.order_by(EngineEventRecord.occurred_at, EngineEventRecord.id).limit(limit)
    return [event_from_row(r) for r in await session.scalars(query)]
