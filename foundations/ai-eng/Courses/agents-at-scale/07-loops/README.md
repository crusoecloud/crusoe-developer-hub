# Chapter 7: Loop engineering

The harness lives inside one run. The loop lives across runs. A trigger fires, the loop loads its state, builds a compact hand-off, runs the chapter 5 team inside the chapter 6 harness in a fresh context, verifies with the chapter 4 graders first and the verifier agent second, saves state, and decides: done, retry, or escalate.

## Files

| file | what it shows |
|---|---|
| `loop.py` | `iterate`, the one iteration; `Budget` with max attempts, dollars and minutes; three stop rules (verified pass, budget exhausted, the same problems twice); `publish` and `escalate` behind a dry-run flag; the `STOP` file kill switch; state saved after every topic. |
| `topics.txt` | The human's file. Add a line, the loop picks it up on the next trigger. |
| `STATE.example.json` | The shape of the loop's file: per topic, attempts, status and the problems from the last attempt, plus the running spend. The real file lives outside git. |
| `../.github/workflows/brief.yml` | The trigger: a weekday cron at six, a manual dispatch, and an issues event for `brief-request`, with the state directory cached between runs. |

## Run

From the repository root, in dry-run mode (the default), which runs everything except the two side effects:

```bash
uv run --env-file .env python 07-loops/loop.py
```

Live, which commits passing briefs to a `results` branch and opens a GitHub issue for escalations:

```bash
BRIEF_DRY_RUN=0 BRIEF_REPO=your-org/your-repo uv run --env-file .env python 07-loops/loop.py
```

To stop the next trigger from doing anything:

```bash
touch STOP
```

State lives at `state/STATE.json` by default. Edit it to reset a topic. Each attempt's run is in `runs/loop/<slug>/<attempt>/` with the team's per-role usage in `team.json`.

## Environment variables

| variable | required | purpose |
|---|---|---|
| `CRUSOE_API_KEY` | yes | the team and the verifier |
| `TAVILY_API_KEY` | yes | search for the researchers and the verifier |
| `BRIEF_DRY_RUN` | no | `1` (default) logs instead of publishing or escalating; `0` goes live |
| `BRIEF_STATE` | no | path of the state file; defaults to `state/STATE.json` |
| `BRIEF_TOPICS` | no | path of the topics file; defaults to `07-loops/topics.txt` |
| `BRIEF_REPO` | no | `owner/name` for `gh issue create`; defaults to the current repository |
| `GITHUB_TOKEN` | live mode | used by `gh` in the workflow |

## Unattended safety

Spend caps and quotas at the provider, on a project and key used only by the loop. The chapter 6 sandbox for anything that executes code. An allowlist of tools per role. Dry-run mode for testing changes to the loop. The kill switch. Then read every escalation for the first week; each one becomes a task in the eval set.

## Notes

- `verify` compares problems by a normalised key (source number plus kind) rather than by exact string, so the verifier phrasing the same problem differently does not burn another attempt.
- Dollars per run come from `03-agent/prices.toml`; until it is filled in, `spent_usd` stays at zero and only the attempt and minute budgets bind.

## Verified

`loop.py` ran one iteration in dry-run mode against the endpoint on 5 November 2026 on a single topic: the team produced a brief, the deterministic graders passed, the verifier agent returned five specific problems, and `state/STATE.json` recorded them with the topic still pending for the next trigger. Publishing and escalation stayed in dry-run. A second iteration, the budget stop and the kill switch were not exercised live; `load_state`, `save_state` and `problem_key` were exercised offline.
