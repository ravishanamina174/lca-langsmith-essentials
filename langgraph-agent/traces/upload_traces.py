"""Load downloaded traces, spread timestamps over a recent window, regenerate IDs, and upload.

Run from the traces folder with:

    uv run python3 upload_traces.py
    uv run python3 upload_traces.py --project my-project --input traces.json
    uv run python3 upload_traces.py --days 0.5 --seed 42

Threads are the unit here, not traces. Each customer turn in a LangSlice
conversation is its own trace, and LangSmith groups those turns into a thread by
the ``thread_id`` in run metadata. So this script:

  * mints one fresh thread id per source thread and rewrites it into every run's
    metadata, so a second upload creates new conversations instead of appending
    turns to the ones already in the project;
  * scatters whole *threads* across the window and keeps each thread's turns in
    their original order, compressing the idle gaps between turns (``--max-gap``)
    so a conversation someone left open overnight replays as one sitting.
"""

import argparse
import asyncio
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

# override=True so .env always wins over pre-existing OS environment
# variables, which keeps a student's shell from silently shadowing their .env.
load_dotenv(override=True)

from langsmith import Client, uuid7
from langsmith.utils import LangSmithNotFoundError

# Project name comes from .env (LANGSMITH_PROJECT); override with --project.
DEFAULT_PROJECT = os.getenv("LANGSMITH_PROJECT")

# Ingest rejects any run whose start_time is more than 24 hours from now, on both
# the multipart and batch endpoints, with a 422. That caps how far back traces can
# be dated: ask for more and the runs are silently dropped. Stay under the limit
# so the last trace in the spread still clears it once the upload has run a while.
MAX_BACKDATE_DAYS = 0.95

# Longest pause to keep between two consecutive turns of the same conversation.
# Real capture sessions have minutes or hours of dead air in them (the author went
# to lunch mid-order); replayed as-is those gaps stretch a thread past the window
# and read as an abandoned conversation. 90s looks like a customer thinking.
DEFAULT_MAX_GAP_SECONDS = 90

# Metadata keys LangSmith reads to group runs into a thread, highest priority
# first. Whichever ones a run carries are all remapped, so the grouping survives
# the id regeneration.
THREAD_METADATA_KEYS = ("thread_id", "session_id", "conversation_id")

# Ingested runs take a moment to become queryable, so the landing check retries
# rather than failing on the first empty read.
WAIT_ATTEMPTS = 10
WAIT_SECONDS = 3

# Runs per request in the landing check. 1000 is the server maximum, so a normal
# upload verifies in a single round trip instead of paging at the 100-run default.
QUERY_PAGE_SIZE = 1000


def parse_dt(s):
    """Parse an ISO timestamp string into a naive (tz-stripped) datetime."""
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def thread_id_of(extra):
    """The thread this run belongs to, or None if it is not part of one."""
    metadata = (extra or {}).get("metadata") or {}
    for key in THREAD_METADATA_KEYS:
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def remap_extra(extra, thread_map, id_map):
    """Point a run's metadata at the regenerated thread and run ids.

    Left alone, every uploaded run would still name the thread it was captured
    in, so a second upload would file its turns under the conversations already
    in the project -- and the two runs of turn 3 would sit side by side in one
    thread. Rewriting the thread key is what makes a re-upload a fresh set of
    conversations.
    """
    if not extra:
        return {}
    remapped = dict(extra)
    metadata = extra.get("metadata")
    if not metadata:
        return remapped

    cleaned = dict(metadata)
    for key in THREAD_METADATA_KEYS:
        old = cleaned.get(key)
        if old and str(old) in thread_map:
            cleaned[key] = thread_map[str(old)]
    # LangGraph also stamps the root run's own id into metadata.run_id. It is
    # only cosmetic, but a stale id there points at a run that is not in this
    # project, so carry it through the same remap.
    old_run_id = cleaned.get("run_id")
    if old_run_id and str(old_run_id) in id_map:
        cleaned["run_id"] = id_map[str(old_run_id)]

    remapped["metadata"] = cleaned
    return remapped


def lay_out_thread(spans, max_gap):
    """Place a thread's traces on a fresh timeline, compressing the idle gaps.

    `spans` maps trace id -> (start, end) in capture time. Returns the offset of
    each trace from the start of the thread, plus the thread's total duration.
    Turn order is taken from capture time and preserved exactly; only the dead
    air between turns is shortened.
    """
    offsets = {}
    cursor = timedelta(0)
    previous_end = None
    for trace_id, (start, end) in sorted(spans.items(), key=lambda kv: kv[1][0]):
        if previous_end is not None:
            # max(..., 0) covers turns that overlap in the capture (a retry, or
            # two windows on one thread): they get serialized rather than
            # reordered.
            gap = max(start - previous_end, timedelta(0))
            cursor += min(gap, max_gap)
        offsets[trace_id] = cursor
        cursor += max(end - start, timedelta(0))
        previous_end = end
    return offsets, cursor


def bootstrap_project(client, name):
    """Materialize the project up front so downstream reads don't race propagation.

    On a fresh project, read_project 404s until the first run has landed and been
    indexed -- so the verification step at the end of the upload would crash even
    though the ingest itself succeeded. Creating (or confirming) the project here
    means everything that follows can trust that read_project resolves, and a real
    auth/tenant problem surfaces before we upload hundreds of runs.
    """
    try:
        return client.read_project(project_name=name).id
    except LangSmithNotFoundError:
        return client.create_project(project_name=name).id


async def landed_run_ids(client, project_id, expected_ids, min_start_time, max_start_time):
    """Poll the project for the uploaded run ids and return the subset now queryable.

    ``runs.query`` is async-only -- its sync predecessor ``client.list_runs`` is
    deprecated and stops working after 31 Jan 2027 -- so the whole retry loop lives
    in here and one event loop covers every attempt. Only ``ID`` is selected: this
    is a counting check, and ID is all ``selects`` returns by default anyway.
    """
    landed = set()
    for _ in range(WAIT_ATTEMPTS):
        landed = {
            str(run.id)
            async for run in client.runs.query(
                # project_ids takes UUIDs, not names: the name -> id index lags
                # create_project by several seconds, so resolving the name here
                # would 404 on a project that already exists. bootstrap_project
                # has the id already.
                project_ids=[str(project_id)],
                ids=list(expected_ids),
                selects=["ID"],
                min_start_time=min_start_time,
                max_start_time=max_start_time,
                page_size=QUERY_PAGE_SIZE,
            )
        }
        if len(landed) == len(expected_ids):
            break
        await asyncio.sleep(WAIT_SECONDS)
    return landed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Target project name")
    parser.add_argument("--input", default="traces.json", help="Input file path")
    parser.add_argument(
        "--days",
        type=float,
        default=MAX_BACKDATE_DAYS,
        help=f"Spread threads randomly over this many days ending now "
        f"(default: {MAX_BACKDATE_DAYS}; the ingest API rejects anything older than 24h)",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=DEFAULT_MAX_GAP_SECONDS,
        help=f"Longest pause kept between consecutive turns of one thread, in "
        f"seconds (default: {DEFAULT_MAX_GAP_SECONDS})",
    )
    parser.add_argument("--seed", type=int, help="Seed for a reproducible upload")
    args = parser.parse_args()

    if args.days > MAX_BACKDATE_DAYS:
        parser.error(
            f"--days {args.days} exceeds the ingest API's 24-hour backdating limit; "
            f"runs older than that are rejected with a 422 and never land. "
            f"Use --days {MAX_BACKDATE_DAYS} or less."
        )

    if args.seed is not None:
        random.seed(args.seed)

    if not args.project:
        parser.error("No project name found. Set LANGSMITH_PROJECT in .env or pass --project.")

    with open(args.input) as f:
        runs = json.load(f)

    print(f"Loaded {len(runs)} runs from {args.input}")
    if not runs:
        print("Nothing to upload.")
        return

    if not any(r.get("start_time") for r in runs):
        raise ValueError("No runs have a start_time; cannot compute time shift.")

    # Build a map from old IDs to fresh uuid7s (uuid7 is time-ordered).
    # For root runs, trace_id must equal id, so map both to the same new uuid7.
    id_map = {}
    for run in runs:
        if run.get("parent_run_id") is None:
            root_new_id = str(uuid7())
            id_map[run["id"]] = root_new_id
            id_map[run["trace_id"]] = root_new_id
    for run in runs:
        for field in ("id", "parent_run_id"):
            old_id = run.get(field)
            if old_id and old_id not in id_map:
                id_map[old_id] = str(uuid7())

    # One fresh thread id per captured thread. Sorted so a --seed run is
    # reproducible end to end rather than depending on dict iteration order.
    thread_map = {
        old: str(uuid7())
        for old in sorted({t for t in (thread_id_of(r.get("extra")) for r in runs) if t})
    }

    # Group runs by (new) trace id, keeping original times for now.
    traces = defaultdict(list)
    for run in runs:
        trace_id = id_map[run["trace_id"]]
        traces[trace_id].append(
            {
                "id": id_map[run["id"]],
                "trace_id": trace_id,
                "dotted_order": None,  # populated below
                "parent_run_id": id_map.get(run.get("parent_run_id")),
                "name": run["name"],
                "run_type": run["run_type"],
                "inputs": run.get("inputs") or {},
                "outputs": run.get("outputs"),
                "error": run.get("error"),
                "extra": remap_extra(run.get("extra"), thread_map, id_map),
                "tags": run.get("tags"),
                "start_time": parse_dt(run["start_time"]),
                "end_time": parse_dt(run["end_time"]) if run.get("end_time") else None,
                # Carried through the id remap and replayed once the runs exist.
                # Absent from traces captured before feedback was collected.
                "feedback": run.get("feedback") or [],
                # Which conversation this turn belongs to, pre-remap. Used to
                # group the traces below; not sent to the API.
                "_thread": thread_id_of(run.get("extra")),
            }
        )

    # Collect each thread's traces. A trace whose runs carry no thread id is its
    # own one-turn thread, so it still gets scattered like everything else.
    threads = defaultdict(dict)
    for trace_id, trace_runs in traces.items():
        thread = next((r["_thread"] for r in trace_runs if r["_thread"]), None)
        starts = [r["start_time"] for r in trace_runs if r["start_time"]]
        ends = [r["end_time"] for r in trace_runs if r["end_time"]] or starts
        threads[thread or f"trace:{trace_id}"][trace_id] = (min(starts), max(ends))

    # Scatter each thread to a random point in the last `--days` days. Every run
    # of a given turn shifts by the same delta, so nesting is intact; every turn
    # of a given thread shifts onto one compressed timeline, so the conversation
    # replays in order with believable pauses between turns. Threads land
    # independently of one another, so the batch looks like organic day-to-day
    # traffic rather than one replayed burst.
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    window = timedelta(days=args.days)
    max_gap = timedelta(seconds=args.max_gap)
    overlong = []
    for thread, spans in sorted(threads.items(), key=lambda kv: min(s for s, _ in kv[1].values())):
        offsets, duration = lay_out_thread(spans, max_gap)
        if duration > window:
            # A single conversation longer than the whole window: drop the pauses
            # entirely rather than let its tail land in the future.
            offsets, duration = lay_out_thread(spans, timedelta(0))
            if duration > window:
                overlong.append(thread)
        latest_start = max(window - duration, timedelta(0))
        base = now - window + timedelta(seconds=random.uniform(0, latest_start.total_seconds()))
        for trace_id, offset in offsets.items():
            delta = (base + offset) - spans[trace_id][0]
            for run in traces[trace_id]:
                run["start_time"] += delta
                if run["end_time"]:
                    run["end_time"] += delta

    for run_list in traces.values():
        for run in run_list:
            del run["_thread"]

    all_starts = [r["start_time"] for trace_runs in traces.values() for r in trace_runs]
    print(
        f"Spread {len(traces)} traces across {len(threads)} threads over {args.days} days: "
        f"{min(all_starts):%Y-%m-%d %H:%M} to {max(all_starts):%Y-%m-%d %H:%M}"
    )
    if overlong:
        print(
            f"Warning: {len(overlong)} thread(s) are longer than the {args.days}-day "
            f"window even with pauses removed; their last turns may be dated in the future.",
            file=sys.stderr,
        )

    client = Client()
    project_id = bootstrap_project(client, args.project)
    print(f"Uploading {len(traces)} traces to project '{args.project}'...")

    for i, (trace_id, trace_runs) in enumerate(traces.items()):
        # Sort: root first, then children by start_time.
        trace_runs.sort(key=lambda r: (r["parent_run_id"] is not None, r["start_time"]))

        # Build dotted_order by walking the parent chain, so nesting is correct
        # regardless of run order or start_time skew.
        runs_by_id = {run["id"]: run for run in trace_runs}
        dotted_orders = {}

        def build_dotted_order(run):
            rid = run["id"]
            if rid in dotted_orders:
                return dotted_orders[rid]
            ts = run["start_time"].strftime("%Y%m%dT%H%M%S%f") + "Z"
            segment = f"{ts}{rid}"
            parent = runs_by_id.get(run["parent_run_id"])
            order = segment if parent is None else f"{build_dotted_order(parent)}.{segment}"
            dotted_orders[rid] = order
            run["dotted_order"] = order
            return order

        for run in trace_runs:
            build_dotted_order(run)

        for run in trace_runs:
            client.create_run(
                id=run["id"],
                trace_id=run["trace_id"],
                dotted_order=run["dotted_order"],
                parent_run_id=run["parent_run_id"],
                name=run["name"],
                run_type=run["run_type"],
                inputs=run["inputs"],
                outputs=run.get("outputs"),
                error=run.get("error"),
                extra=run.get("extra"),
                tags=run.get("tags"),
                start_time=run["start_time"],
                end_time=run["end_time"],
                project_name=args.project,
            )

        if (i + 1) % 10 == 0:
            print(f"  Uploaded {i + 1}/{len(traces)} traces")

    # Wait for all background operations to complete.
    print("Flushing...")
    client.flush()

    # create_run() only enqueues; the HTTP POST happens on a background thread and
    # a rejected batch is logged there, not raised here. Count what actually landed
    # before claiming success, so a server-side rejection cannot pass for an upload.
    # Filter by the ids just uploaded instead of listing the project and intersecting
    # locally: the project holds every previous upload too, and paging all of it back
    # costs ~15x this query and grows every run. The ids go in the POST body, so the
    # whole batch fits one call, and the filter makes the intersection implicit.
    expected_ids = {run["id"] for trace_runs in traces.values() for run in trace_runs}
    # runs.query bounds every query by start_time and its defaults are wrong here:
    # min_start_time would be 1 day ago and max_start_time now, while this script
    # dates runs right up to the 24-hour backdating limit and lets an overlong
    # thread spill slightly past now. Derive the window from the spread computed
    # above, with an hour of slack so no boundary run reads as missing.
    query_margin = timedelta(hours=1)
    landed_ids = asyncio.run(
        landed_run_ids(
            client,
            project_id,
            expected_ids,
            min(all_starts).replace(tzinfo=timezone.utc) - query_margin,
            max(all_starts).replace(tzinfo=timezone.utc) + query_margin,
        )
    )

    missing = len(expected_ids) - len(landed_ids)
    if missing:
        print(
            f"ERROR: only {len(landed_ids)}/{len(expected_ids)} runs landed in "
            f"'{args.project}' ({missing} missing). Check the ingest warnings above.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"Verified {len(landed_ids)}/{len(expected_ids)} runs landed.")

    # Feedback is keyed by run id, so it can only be attached once the runs it
    # points at have been created -- hence after the verification above, and against
    # the regenerated ids rather than the ones in the input file. Whatever the
    # capture carried (evaluator scores, human review) is replayed as-is; nothing
    # is invented here.
    #
    # Passing trace_id puts each record on the batched tracing queue instead of a
    # blocking POST per record; the flush below waits for them. session_id is the
    # project the run lives in -- omitting it is deprecated and will stop working.
    n_feedback = 0
    for trace_runs in traces.values():
        for run in trace_runs:
            for feedback in run["feedback"]:
                client.create_feedback(
                    run_id=run["id"],
                    trace_id=run["trace_id"],
                    session_id=project_id,
                    key=feedback["key"],
                    score=feedback.get("score"),
                    value=feedback.get("value"),
                    comment=feedback.get("comment"),
                )
                n_feedback += 1
    if n_feedback:
        client.flush()
        print(f"Replayed {n_feedback} feedback records.")

    print(
        f"Done! Uploaded {len(traces)} traces across {len(threads)} threads "
        f"to '{args.project}'."
    )


if __name__ == "__main__":
    main()
