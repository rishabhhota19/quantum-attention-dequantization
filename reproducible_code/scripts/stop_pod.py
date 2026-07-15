"""Turn the RunPod pod OFF when the run finishes (so it stops billing).

Uses the RunPod **REST** API (Bearer auth). Needs RUNPOD_API_KEY in the env (or
--api-key). The pod id is resolved ROBUSTLY:

  1. --pod-id, if given; else
  2. RUNPOD_POD_ID env (set inside most pods), if present AND still RUNNING; else
  3. whatever pod is currently RUNNING, resolved live from GET /v1/pods.

Resolving the running pod from the REST list is deliberate: RUNPOD_POD_ID is NOT
always set in the pod env, and after a pod MIGRATION the old id goes stale (the
"stopped the wrong pod, new pod billed idle" bug). Targeting the live RUNNING pod
avoids both. Default action = stop (pause; keeps the /workspace volume so results
persist); --terminate deletes the pod.

    python scripts/stop_pod.py                 # stop the live RUNNING pod
    python scripts/stop_pod.py --pod-id <id>   # stop a specific pod
    python scripts/stop_pod.py --terminate
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

REST = "https://rest.runpod.io/v1/pods"


def _req(url, key, method="GET"):
    return urllib.request.Request(
        url, method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})


def list_pods(key):
    """GET /v1/pods -> list of pod dicts (handles list or {pods|data:[...]} shapes)."""
    with urllib.request.urlopen(_req(REST, key), timeout=30) as r:
        data = json.loads(r.read().decode())
    return data if isinstance(data, list) else data.get("pods", data.get("data", []))


def running_pod_ids(key):
    out = []
    for p in list_pods(key):
        if str(p.get("desiredStatus", "")).upper() == "RUNNING":
            out.append(p.get("id"))
    return [i for i in out if i]


def stop_pod(pod_id: str, key: str, terminate: bool = False):
    # stop = POST /pods/{id}/stop ; terminate = DELETE /pods/{id}
    if terminate:
        url, method = f"{REST}/{pod_id}", "DELETE"
    else:
        url, method = f"{REST}/{pod_id}/stop", "POST"
    with urllib.request.urlopen(_req(url, key, method), timeout=30) as r:
        return f"HTTP {r.status} {r.read().decode()[:200]}"


def resolve_targets(a, key):
    """Pick the pod(s) to stop, preferring an explicit/known-RUNNING id, else the
    live RUNNING set. Avoids stopping a stale (migrated) id."""
    if a.pod_id:
        return [a.pod_id]
    try:
        running = running_pod_ids(key)
    except Exception as e:
        print(f"[stop_pod] could not list pods ({e});"
              f" falling back to RUNPOD_POD_ID env if set.")
        env_id = os.environ.get("RUNPOD_POD_ID")
        return [env_id] if env_id else []
    env_id = os.environ.get("RUNPOD_POD_ID")
    if env_id and env_id in running:            # prefer THIS pod when it's the running one
        return [env_id]
    return running                              # else stop whatever is actually RUNNING


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pod-id", default=None,
                   help="explicit pod id; default = resolve the live RUNNING pod")
    p.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    p.add_argument("--terminate", action="store_true")
    a = p.parse_args()

    key = a.api_key
    if not key:
        print("[stop_pod] missing RUNPOD_API_KEY (env or --api-key) -> NOT stopping pod "
              "(set it, or stop the pod manually in the console).")
        return

    targets = resolve_targets(a, key)
    if not targets:
        print("[stop_pod] no RUNNING pod found to stop "
              "(already stopped?). Nothing to do.")
        return

    action = "terminate" if a.terminate else "stop"
    for pod_id in targets:
        try:
            print(f"[stop_pod] {action} pod {pod_id}: {stop_pod(pod_id, key, a.terminate)}")
        except Exception as e:
            print(f"[stop_pod] FAILED to {action} pod {pod_id}: {e}  -> stop it manually!")


if __name__ == "__main__":
    main()
