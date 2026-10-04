# Fix Loop Declaration

Status: **comparison tooling shipped; empirical fix not yet shipped**.

The current worst measured benchmark issue is not selectable from the checked-in data because
there is no laser ground-truth manifest. The existing drift reports are diagnostic ablations,
not gate results. Before making a prediction, add a manifest with raw captures, repeat capture,
laser/tape measurements, and the consumer-app export.

## Required before/after record

| Field | Before | After |
|---|---|---|
| Failing gate and number | TODO | TODO |
| Root-cause hypothesis | TODO | TODO |
| Evidence | `bench/results.json` | `bench/results_after.json` |
| Shipped change | TODO | TODO |
| Predicted gate value | TODO | TODO |
| Observed gate value | TODO | TODO |

Both runs must be regenerable from raw inputs. The proposed command shape is:

```powershell
.venv\Scripts\python.exe -m roomscan bench bench\manifest_before.json --out bench\results_before
.venv\Scripts\python.exe -m roomscan bench bench\manifest_after.json --out bench\results_after
.venv\Scripts\python.exe -m roomscan fixloop bench\manifest_before.json bench\manifest_after.json --out fixloop\results
```