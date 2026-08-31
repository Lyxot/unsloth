# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A/B the #9934 write amplification against the real durable-chat append path.

Studio persists a generation's output in batches so the run survives a disconnect. Every
storage accessor opens and closes its own connection, so before the fix each batch's writer
was routinely the last WAL participant and sqlite checkpointed the whole WAL back into
studio.db on that close.

Run with --keeper for the shipped behaviour and without it for the negative control, which
is exactly the pre-fix code path: the keeper is the only difference between the two.

Reports, per batch: whether studio.db changed, and whether the -wal survived.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[3] / "studio" / "backend"
sys.path.insert(0, str(BACKEND))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keeper", action = "store_true")
    parser.add_argument("--batches", type = int, default = 64)
    parser.add_argument("--events-per-batch", type = int, default = 16)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    home = Path(tempfile.mkdtemp(prefix = "wal-ab-"))
    os.environ["UNSLOTH_STUDIO_HOME"] = str(home)

    import storage.chat_generation_runs_db as runs_db
    import storage.studio_db as studio_db
    from utils.paths import studio_db_path

    db_path = Path(str(studio_db_path()))
    wal_path = Path(f"{db_path}-wal")
    shm_path = Path(f"{db_path}-shm")

    studio_db.get_connection().close()
    journal = _journal_mode(studio_db, db_path)

    studio_db.upsert_chat_thread(
        {
            "id": "ab-thread",
            "title": "Chat",
            "modelType": "base",
            "modelId": "local",
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "ab-user",
            "threadId": "ab-thread",
            "role": "user",
            "content": [{"type": "text", "text": "hi"}],
            "createdAt": 2,
        }
    )
    runs_db.create_run(
        run_id = "ab-run",
        owner_subject = "ab",
        thread_id = "ab-thread",
        user_message_id = "ab-user",
        assistant_message_id = "ab-assistant",
        request_payload = {"model": "ab", "messages": [{"role": "user", "content": "hi"}]},
    )
    worker_token = runs_db.get_worker_token("ab-run")
    runs_db.mark_running("ab-run", worker_token)

    engaged = studio_db.open_wal_keeper() if args.keeper else False

    rewrites = 0
    batches_without_wal = 0
    digest = _digest(db_path)
    for index in range(args.batches):
        runs_db.append_events(
            "ab-run",
            worker_token,
            [("chunk", {"text": f"{index}-{n}"}) for n in range(args.events_per_batch)],
        )
        current = _digest(db_path)
        if current != digest:
            rewrites += 1
            digest = current
        if not wal_path.exists():
            batches_without_wal += 1

    persisted = len(runs_db.list_events("ab-run", limit = 100000))
    if args.keeper:
        studio_db.close_wal_keeper()

    integrity = _integrity(studio_db)
    result = {
        "mode": "fixed" if args.keeper else "baseline (negative control)",
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "journal_mode": journal,
        "keeper_engaged": engaged,
        "batches": args.batches,
        "events_per_batch": args.events_per_batch,
        "main_database_rewrites": rewrites,
        "batches_with_no_surviving_wal": batches_without_wal,
        "persisted_events": persisted,
        "integrity_check": integrity,
        "wal_after_shutdown": wal_path.exists(),
        "shm_after_shutdown": shm_path.exists(),
    }
    print(json.dumps(result, indent = 2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent = 2))

    expected_rewrites = 0 if args.keeper else args.batches
    expected_missing_wal = 0 if args.keeper else args.batches
    ok = (
        rewrites == expected_rewrites
        and batches_without_wal == expected_missing_wal
        # >=, since the run's own lifecycle events share the log with the chunks.
        and persisted >= args.batches * args.events_per_batch
        and integrity == "ok"
    )
    print("PASS" if ok else "FAIL", file = sys.stderr)
    return 0 if ok else 1


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _journal_mode(studio_db, path: Path) -> str:
    conn = studio_db.get_connection()
    try:
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        conn.close()


def _integrity(studio_db) -> str:
    conn = studio_db.get_connection()
    try:
        return str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
