---
title: XZZ2l2nu Data Deduplication
---

# XZZ2l2nu data deduplication

**Summary**

- Data comes from two primary datasets per era: SingleMuon + EGamma (SingleElectron in 2016/2017).
- One collision can be in both, so an event can appear twice.
- Stage1 output (`f1_0/`) **still has duplicates**.
- Duplicates are removed **only at compaction**, and only if enabled.

## Where duplicates come from

- **Dataset overlap:** e.g. an event firing `IsoMu24` and `Ele32_WPTight_Gsf` is in both SingleMuon and EGamma.
- **Duplicate files:** 2018A SingleMuon `UL2018_NanoAODv15-v1` (INVALID on DAS) has files in blocks `100000` and `2810000` with the same events.
- Event identity = exact `(run, luminosityBlock, event)`.

## Pipeline

### Step 1 — Stage1 stream veto (partial)

[copperhead_processor.py](../src/copperhead_processor.py), HLT step, XZZ Run 2 data only:

```python
pass_hlt = OR(all configured HLT paths)
muon_trig = OR(paths matching r"Mu(?=\d|_)")   # IsoMu*, Mu50, Mu55, Mu17_Mu8, Mu37_TkMu27, Mu*_Ele* ...
if stream == "SingleMuon":            keep = pass_hlt
elif stream in ("EGamma", "SingleElectron"): keep = pass_hlt and not muon_trig
else:                                  raise error
```

- Removes most of the overlap, not all of it.

### Step 2 — Compaction: exact duplicate removal

[compact_parquet_data.py](../scripts/compact_parquet_data.py) `--deduplicate-data-inventory <prestage JSON>` → [data_stream_deduplication.py](../src/lib/data_stream_deduplication.py):

```python
for era in data_A, data_B, ...:                 # each era holds both streams
    shards = stage1 parquet files of era
    # part_<raw stem>_*.parquet -> raw file -> stream (via prestage JSON)
    order shards by (SingleMuon first, then EGamma/SingleElectron, then path)
    keys = (run, luminosityBlock, event) as exact uint64   # never float
    keep first occurrence of each key, drop the rest
    assert compacted rows == unique keys
    write compacted/<era>/0/data_deduplication.json  # counts + 10 examples
```

- Data only; MC never goes through this.

## How to enable

```bash
# common workflow (common_workflow.sh, build_compact_cmd)
export COMPACT_DEDUP_INVENTORY=prestage_output/processor_samples_2018_NanoAODv15_XZZ2l2nu.json
bash ./run_full_xzz2l2nu_pipeline.sh compact

# or directly
python scripts/compact_parquet_data.py -y 2018 --analysis XZZ2l2nu --input_path <label dir> \
    --deduplicate-data-inventory prestage_output/processor_samples_2018_NanoAODv15_XZZ2l2nu.json
```

Caveats:

- **No variable / option → duplicates stay, with no warning.**
- With the option, an existing compacted data dir without `data_deduplication.json` is refused → move it aside, recompact.
- Who sees duplicates:
  - `run_plotter.py`: reads `compacted/` → clean.
  - `get_yields.py`: prefers `compacted/`, falls back to `f1_0/` → duplicates if not compacted.
  - Cutflow tools: read Stage1 output → duplicates.

## 2018 result

`Run2_nanoAODv15_XZZ2l2nu_Full2018_Oct23_officialsel`:

| Era | Rows in | Removed |
|---|---:|---:|
| A | 15,873,653 | 2,021 |
| B | 8,052,763 | 1,264 |
| C | 7,851,659 | 1,196 |
| D | 36,196,769 | 5,284 |
| **Total** | | **9,765** |

Examples:

- A `316995:439:663183886` — kept `SingleMuon/...-v1/100000/18645d2d...`, removed `SingleMuon/...-v1/2810000/a438b53d...` (duplicate file)
- B `317182:55:37011298` — kept `SingleMuon/...-v1/100000/9a6147d8...`, removed `EGamma/...-v2/2520000/012a564d...` (dataset overlap)

## Comparison with the reference

Reference: [`removeDuplicates.C`](https://github.com/ram1123/nanoAOD_skim/blob/HZZ_Analysis_2l2nu/scripts/analysis/removeDuplicates.C)

| | Reference | Ours |
|---|---|---|
| Key | `(run, lumi, event)` | Same |
| Kept copy | First in input-list order | First, SingleMuon before EGamma |
| Applied to | All skimmed events | Stage1-selected events, at compaction |
| Trigger per stream | Full OR everywhere | EGamma vetoes muon-leg paths |

- Duplicate removal: **equivalent** (both copies are identical, so they pass/fail selection together).
- Stream veto: **ours only** → see below.

## Open issue: Stage1 stream veto

- The veto also removes μμ / eμ cross triggers, which belong to DoubleMuon / MuonEG.
- We don't read those datasets → an EGamma event with a cross trigger but no single-muon trigger is lost.
- Estimate, one 2018D EGamma file (1.6M events, approx. selection, pT > 25):

| Channel | Pass OR | Vetoed | Possibly lost* |
|---|---:|---:|---:|
| ee | 34,956 | 67 | 63 (0.2%) |
| μμ | 27 | 27 | 2 |
| eμ | 681 | 633 | 63 (9%) |

\* vetoed and no single-muon path from our list fired; upper bound.

- Proposal (undecided):
  - Remove the Stage1 veto; rely on compaction deduplication (= reference).
  - Make compaction refuse XZZ data without a dedup inventory.
