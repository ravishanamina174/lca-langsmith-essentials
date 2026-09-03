"""Download all runs from a LangSmith project, with their feedback, and save
them to a JSON file.

Run from the traces folder with:

    uv run python3 download_traces.py
    uv run python3 download_traces.py --project my-project --output traces.json

The LangSlice agent is a chat agent: every customer turn is its own trace, and
LangSmith stitches the turns of one conversation into a thread by the
``thread_id`` in each run's metadata. That key is what makes the Threads view in
the outline work, so it is preserved verbatim here and re-mapped (not dropped)
on upload.
"""

import argparse
import asyncio
import json
import os
import re
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

# override=True so .env always wins over pre-existing OS environment
# variables, which keeps a student's shell from silently shadowing their .env.
load_dotenv(override=True)

from langsmith import Client

# Project name comes from .env (LANGSMITH_PROJECT); override with --project.
DEFAULT_PROJECT = os.getenv("LANGSMITH_PROJECT")

# LangSmith groups runs into a thread by the first of these it finds in a run's
# metadata. LangGraph writes thread_id; the other two are here because a trace
# captured from a plain SDK app may use them instead.
THREAD_METADATA_KEYS = ("thread_id", "session_id", "conversation_id")

# Runtime tracebacks (captured in run.error and sometimes inputs/outputs) bake in
# the absolute path of the local install, e.g.
#   /Users/<you>/.../langsmith-essentials-repo/.venv/lib/python3.13/site-packages/...
# That leaks the author's home directory into the committed traces. Rewrite any
# such local project path to a neutral "/app" so the file is portable and clean.
# Derived from this file's location (repo root is the parent of traces/) so it
# keeps working if the checkout is renamed. Two alternatives, longest first:
# the actual repo root path, then any POSIX home-style prefix ending in the
# project dir name (covers traces captured from a different checkout location).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_PATH_RE = re.compile(
    re.escape(str(_PROJECT_ROOT))
    + r"|/(?:Users|home)/[^\s\"'\\]*?/"
    + re.escape(_PROJECT_ROOT.name)
)


def scrub(text):
    """Replace local absolute project paths with a neutral '/app' prefix."""
    return _LOCAL_PATH_RE.sub("/app", text)


# extra.runtime records the box the trace was captured on. The library and SDK
# versions in there are worth keeping -- they explain the shape of the runs --
# but the OS build string and Python patch level identify one laptop and say
# nothing about the agent, so drop just those two.
_RUNTIME_FINGERPRINT_KEYS = ("platform", "runtime_version")

# The langgraph dev server stamps every run with who called it and which local
# server took the request. None of it survives a replay meaningfully, and the
# auth/user fields carry a real identity once the agent runs on a deployment,
# so drop them. thread_id, graph_id, assistant_id and the version fields stay:
# the first drives the Threads view, the rest explain the trace.
_CAPTURE_METADATA_KEYS = (
    "created_by",
    "user_id",
    "langgraph_auth_user_id",
    "langgraph_request_id",
    "langgraph_api_url",
)


# The tracing SDK copies the LANGSMITH_* environment into every run's metadata,
# which pins the capture to one workspace/project/endpoint. Those values are
# meaningless (and wrong) once the traces are replayed somewhere else, so drop
# the whole family rather than carrying a stale workspace id around.
def scrub_metadata(extra):
    """Drop capture-environment identifiers from a run's extra.metadata/runtime."""
    if not extra:
        return extra
    scrubbed = dict(extra)

    metadata = extra.get("metadata")
    if metadata:
        cleaned = {
            k: v
            for k, v in metadata.items()
            if not k.startswith("LANGSMITH_") and k not in _CAPTURE_METADATA_KEYS
        }
        # revision_id is the capture machine's git describe, so it picks up a
        # "-dirty" suffix whenever the checkout had uncommitted edits. Keep the
        # commit, drop the local working-tree state.
        revision_id = cleaned.get("revision_id")
        if isinstance(revision_id, str) and revision_id.endswith("-dirty"):
            cleaned["revision_id"] = revision_id[: -len("-dirty")]
        scrubbed["metadata"] = cleaned

    runtime = extra.get("runtime")
    if runtime:
        scrubbed["runtime"] = {
            k: v for k, v in runtime.items() if k not in _RUNTIME_FINGERPRINT_KEYS
        }

    return scrubbed


def thread_id_of(extra):
    """The thread this run belongs to, or None if it is not part of one."""
    metadata = (extra or {}).get("metadata") or {}
    for key in THREAD_METADATA_KEYS:
        value = metadata.get(key)
        if value:
            return str(value)
    return None


# Feedback is a separate resource from runs -- runs.query never returns it -- so it
# has to be fetched by run id and stitched back on. The ids travel as repeated
# query params, so ask for a batch at a time rather than the whole project at once.
FEEDBACK_BATCH = 50

# Feedback LangSmith wrote about the source project rather than feedback about the
# agent: issue ids from its triage, and the categories its Insights job assigns.
# Both belong to that project, so replaying them would staple someone else's
# analysis onto a fresh upload. Human and evaluator scores (positive_sentiment,
# sentiment, ...) are the point of the exercise and are kept.
SKIP_FEEDBACK_KEYS = {"langsmith_issue_id"}
SKIP_FEEDBACK_PREFIXES = ("langsmith:",)


def skip_feedback(key):
    return key in SKIP_FEEDBACK_KEYS or key.startswith(SKIP_FEEDBACK_PREFIXES)


def fetch_feedback(client, run_ids):
    """Map run id -> its feedback records, fetched in batches."""
    by_run = defaultdict(list)
    for i in range(0, len(run_ids), FEEDBACK_BATCH):
        for feedback in client.list_feedback(run_ids=run_ids[i:i + FEEDBACK_BATCH]):
            if skip_feedback(feedback.key):
                continue
            by_run[str(feedback.run_id)].append(
                {
                    "key": feedback.key,
                    "score": feedback.score,
                    "value": feedback.value,
                    "comment": feedback.comment,
                }
            )
    return by_run


# runs.query always bounds results by start_time and the server rejects a window
# wider than 401 days, so there is no "whole history" query any more -- list_runs,
# its deprecated predecessor, had no bound at all. 399 days is the widest window
# that clears the cap with room for clock skew. A project with runs older than
# that has to be pulled in chunks.
MAX_QUERY_DAYS = 399

# Runs per request. 1000 is the server maximum; the default is 100.
QUERY_PAGE_SIZE = 1000

# runs.query populates only the fields named here and leaves every other attribute
# None, so this list has to cover every key written into the JSON below. Two names
# changed with the API: select -> selects with SCREAMING_SNAKE_CASE values, and
# parent_run_id -> PARENT_RUN_IDS, which is the whole ancestor chain root-first
# rather than a single id.
RUN_SELECTS = [
    "ID",
    "TRACE_ID",
    "PARENT_RUN_IDS",
    "NAME",
    "RUN_TYPE",
    "INPUTS",
    "OUTPUTS",
    "ERROR",
    "EXTRA",
    "TAGS",
    "START_TIME",
    "END_TIME",
]


async def fetch_runs(client, project_name):
    """Every run in the project, as the plain dicts written to the JSON file.

    Async because ``runs.query`` is async-only; ``client.list_runs``, the sync call
    this replaces, is deprecated and stops working after 31 Jan 2027.
    """
    # runs.query takes project UUIDs, not names.
    project = await client.aread_project(project_name=project_name)
    now = datetime.now(timezone.utc)
    runs = []
    async for run in client.runs.query(
        project_ids=[str(project.id)],
        selects=RUN_SELECTS,
        min_start_time=now - timedelta(days=MAX_QUERY_DAYS),
        # Runs replayed by upload_traces.py can land a little ahead of now, and the
        # default upper bound is now, which would silently drop them.
        max_start_time=now + timedelta(days=1),
        page_size=QUERY_PAGE_SIZE,
    ):
        # Root first, immediate parent last; empty for a root run. extra still
        # nests metadata/runtime the way list_runs returned it, so the scrubbers
        # below are unchanged.
        parents = run.parent_run_ids or []
        runs.append(
            {
                "id": str(run.id),
                "trace_id": str(run.trace_id),
                "parent_run_id": str(parents[-1]) if parents else None,
                "name": run.name,
                "run_type": run.run_type,
                "inputs": run.inputs,
                "outputs": run.outputs,
                "error": run.error,
                "extra": scrub_metadata(run.extra),
                "tags": run.tags,
                "start_time": run.start_time.isoformat() if run.start_time else None,
                "end_time": run.end_time.isoformat() if run.end_time else None,
            }
        )
    return runs


def serialize(obj):
    """JSON serializer for objects not serializable by default (datetime, UUID)."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Source project name")
    parser.add_argument("--output", default="traces.json", help="Output file path")
    args = parser.parse_args()

    if not args.project:
        parser.error("No project name found. Set LANGSMITH_PROJECT in .env or pass --project.")

    client = Client()
    print(f"Fetching runs from project '{args.project}'...")

    runs = asyncio.run(fetch_runs(client, args.project))

    feedback_by_run = fetch_feedback(client, [r["id"] for r in runs])
    for run in runs:
        run["feedback"] = feedback_by_run.get(run["id"], [])

    # Sort for stable, human-readable output: whole conversations stay together
    # (thread, oldest thread first), then trace, then root-first within a trace.
    # Runs with no thread sort last under a "~" key rather than crashing the sort.
    thread_first_start = {}
    for run in runs:
        thread = thread_id_of(run["extra"]) or "~"
        start = run["start_time"] or ""
        if thread not in thread_first_start or start < thread_first_start[thread]:
            thread_first_start[thread] = start

    def sort_key(run):
        thread = thread_id_of(run["extra"]) or "~"
        return (
            thread_first_start[thread],
            thread,
            run["start_time"] or "" if run["parent_run_id"] is None else "",
            run["trace_id"],
            run["parent_run_id"] is not None,
            run["start_time"] or "",
        )

    runs.sort(key=sort_key)

    # Serialize first, then scrub any local absolute paths out of the whole
    # payload (error tracebacks, inputs, outputs, extra) in one pass.
    payload = scrub(json.dumps(runs, indent=2, default=serialize))
    with open(args.output, "w") as f:
        f.write(payload)

    n_traces = len({r["trace_id"] for r in runs})
    n_feedback = sum(len(r["feedback"]) for r in runs)
    threads = {thread_id_of(r["extra"]) for r in runs}
    orphans = sum(1 for r in runs if thread_id_of(r["extra"]) is None)
    threads.discard(None)
    print(
        f"Saved {len(runs)} runs across {n_traces} traces in {len(threads)} threads "
        f"({n_feedback} feedback records) to {args.output}"
    )
    if orphans:
        print(
            f"Note: {orphans} runs carry no thread_id; upload_traces.py will place "
            f"each of their traces in a thread of its own."
        )


if __name__ == "__main__":
    main()
