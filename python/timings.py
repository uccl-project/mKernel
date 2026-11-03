"""Dump and render the in-kernel timing ring (include/common/timings.cuh).

Three concerns:

  unpack()      ring buffer  -> flat (block, ts, event_id, payload) table
  save()/load() table        -> self-describing .npz
  plot()        table        -> per-warp Gantt PDF

Everything downstream of the flat table is ordinary numpy. Each kernel's warp
roles and phase tables live in LAYOUTS below, keyed by the `kernel` name saved
in the .npz, so recolouring or relabelling never needs the GPU or the .so:

    python python/timings.py traces/gemm_rs_blackwell_M8192_rank0.npz --blocks 0-7
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Matches timing_profile::PAYLOAD_WARP_SHIFT. The ring is partitioned per CTA,
# not per warp: all warps of a block share one head and interleave into the
# block's slot. The warp id packed into each payload's top bits separates them
# again, and keeps the pairing key unique when two warps share a sequence number.
WARP_ID_SHIFT = 28
SEQ_MASK = (1 << WARP_ID_SHIFT) - 1

# Categorical slots in fixed order (dataviz reference palette, light mode),
# assigned per role in phase order; NEUTRAL marks bookkeeping phases.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#a1a1aa"


@dataclass(frozen=True)
class Layout:
    """Which warp plays which role, and the phases each role emits.

    Phases are (begin_event, end_event, color, label, pair_by). Events are
    named, not numbered, so a trace saved before the enum grew still renders:
    the .npz carries its own name -> id map.
    """

    warp_roles: dict  # warp id -> role
    role_order: list  # row order, top to bottom
    phases: dict  # role -> [phase]

    def warps(self, role):
        return {w for w, r in self.warp_roles.items() if r == role}


# Mirrors the warp branch at the bottom of gemm_rs_blackwell() in
# src/gemm_rs_blackwell.cu. Milestones are chained: the end of one phase is the
# begin of the next, and every milestone of one tile carries the same payload.
_SETUP = ("SETUP_BEGIN", "SETUP_DONE", NEUTRAL, "setup", "payload")
GEMM_RS_BLACKWELL = Layout(
    # Only thread 0 emits for the epilogue warpgroup, so it is one row per CTA.
    warp_roles={
        0: "epilogue",
        4: "producer",
        5: "mma",
        6: "mma",
        7: "signal",
        **{w: "reduce-scatter" for w in range(8, 12)},
    },
    role_order=["producer", "mma", "epilogue", "signal", "reduce-scatter"],
    phases={
        "producer": [
            _SETUP,
            # Includes waiting on mma_finish for a free smem stage.
            ("LOAD_TILE_BEGIN", "LOAD_TILE_ISSUED", SLOTS[0], "prod: issue tile loads", "payload"),
            # Per K step, drawn over the tile bar: blocked on the MMA freeing a stage.
            ("LOAD_K_BEGIN", "LOAD_K_STAGE_FREE", SLOTS[6], "prod: k wait stage free", "payload"),
        ],
        # MMA rows exist only for cta_rank 0, which issues the 2-CTA MMA.
        "mma": [
            _SETUP,
            ("MMA_TILE_BEGIN", "MMA_TMEM_FREE", SLOTS[1], "mma: wait tmem", "payload"),
            # Includes waiting on tma_load per K step.
            ("MMA_TMEM_FREE", "MMA_COMMITTED", SLOTS[0], "mma: k-loop", "payload"),
            # Per K step, drawn over the k-loop bar: blocked on that stage's TMA bytes.
            ("MMA_K_BEGIN", "MMA_K_LOADED", SLOTS[7], "mma: k wait load", "payload"),
        ],
        "epilogue": [
            _SETUP,
            ("EPI_SUBTILE_BEGIN", "EPI_MMA_DONE", SLOTS[0], "epi: wait mma", "payload"),
            ("EPI_MMA_DONE", "EPI_TMEM_LOADED", SLOTS[1], "epi: tmem->reg", "payload"),
            ("EPI_TMEM_LOADED", "EPI_STORES_ISSUED", SLOTS[2], "epi: smem + tma store", "payload"),
            # Once per tile, after its last subtile: n is half of the rows above.
            ("EPI_STORES_ISSUED", "EPI_STORE_DRAINED", SLOTS[3], "epi: drain stores", "payload"),
            # Traces from before the signal warp: the epilogue signaled inline.
            ("EPI_STORE_DRAINED", "EPI_SIGNALED", SLOTS[4], "epi: signal owner", "payload"),
            ("EPI_TEARDOWN_BEGIN", "EPI_TEARDOWN_DONE", NEUTRAL, "epi: teardown", "payload"),
        ],
        "signal": [
            _SETUP,
            ("SIG_TILE_BEGIN", "SIG_EPI_DRAINED", SLOTS[0], "sig: wait epilogue", "payload"),
            # The release, split into the fence and the relaxed red after it.
            # Traces from before SIG_FENCED show neither row.
            ("SIG_EPI_DRAINED", "SIG_FENCED", SLOTS[4], "sig: fence", "payload"),
            ("SIG_FENCED", "EPI_SIGNALED", SLOTS[5], "sig: red", "payload"),
        ],
        "reduce-scatter": [
            _SETUP,
            ("RS_TILE_BEGIN", "RS_READY", SLOTS[0], "rs: wait peers", "payload"),
            ("RS_READY", "RS_TILE_DONE", SLOTS[1], "rs: ld_reduce + store", "payload"),
        ],
    },
)

LAYOUTS = {"gemm_rs_blackwell": GEMM_RS_BLACKWELL}


# ---------------------------------------------------------------------------
# Dump
# ---------------------------------------------------------------------------
def unpack(raw_i64, num_blocks: int, events_per_block: int):
    """Ring buffer -> ((N, 4) int64 table, per-block head array).

    `raw_i64` is the int64 array the kernel wrote: two int64s per 16-byte
    record. Columns of the returned table are (block, timestamp_ns, event_id,
    payload).
    """
    raw = np.asarray(raw_i64).ravel().view(np.uint64)
    raw = raw.reshape(num_blocks, events_per_block, 2)

    ts = raw[..., 0]
    packed = raw[..., 1]
    event_ids = (packed >> np.uint64(32)).astype(np.uint32)
    payloads = (packed & np.uint64(0xFFFFFFFF)).astype(np.uint32)

    # The buffer is zeroed by the host and %globaltimer has been counting since
    # GPU boot, so a zero timestamp can only mean "never written". The first
    # such index in a block's slot is that block's event count.
    valid = ts != 0
    heads = valid.argmin(axis=1)
    heads = np.where(valid.all(axis=1), events_per_block, heads)

    rows = []
    for b in range(num_blocks):
        n = int(heads[b])
        if n == 0:
            continue
        rows.append(
            np.stack(
                [
                    np.full(n, b, dtype=np.int64),
                    ts[b, :n].astype(np.int64),
                    event_ids[b, :n].astype(np.int64),
                    payloads[b, :n].astype(np.int64),
                ],
                axis=1,
            )
        )

    records = np.concatenate(rows, axis=0) if rows else np.zeros((0, 4), np.int64)
    return records, heads


def save(path, records, name_to_id, *, kernel: str, **meta):
    """Write the flat table plus enough metadata to render offline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        records=records,
        event_ids=np.array(list(name_to_id.values()), dtype=np.int64),
        event_names=np.array(list(name_to_id.keys())),
        kernel=np.array(kernel),
        **{k: np.asarray(v) for k, v in meta.items()},
    )
    return path


def load(path):
    """.npz -> (records, name_to_id, meta dict)."""
    z = np.load(path, allow_pickle=False)
    records = z["records"]
    name_to_id = {str(n): int(i) for n, i in zip(z["event_names"], z["event_ids"])}
    meta = {k: z[k] for k in z.files if k not in ("records", "event_ids", "event_names")}
    return records, name_to_id, meta


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def build_spans(records, name_to_id, phases, warps=None):
    """Flat table -> [(block, warp, start_ns, dur_ns, color, label)].

    Pairing uses (block << 32) | payload, unique per (block, warp, seq) because
    the kernel packs the warp id into the payload. Vectorized per phase: no
    Python loop over records.
    """
    if records.shape[0] == 0:
        return []

    blk = records[:, 0].astype(np.int64)
    ts = records[:, 1].astype(np.int64)
    eid = records[:, 2].astype(np.int64)
    pld = records[:, 3].astype(np.int64) & 0xFFFFFFFF
    warp = pld >> WARP_ID_SHIFT

    if warps is not None:
        sel = np.isin(warp, sorted(warps))
        blk, ts, eid, pld, warp = (a[sel] for a in (blk, ts, eid, pld, warp))
        if blk.size == 0:
            return []

    key = (blk << 32) | pld

    spans = []
    for start_name, end_name, color, label, mode in phases:
        sid = name_to_id.get(start_name)
        eid_v = name_to_id.get(end_name)
        if sid is None or eid_v is None:
            continue
        s_mask = eid == sid
        e_mask = eid == eid_v
        if not s_mask.any() or not e_mask.any():
            continue

        if mode == "payload":
            s_key, e_key = key[s_mask], key[e_mask]
            s_ts, e_ts = ts[s_mask], ts[e_mask]
            order = np.argsort(e_key, kind="stable")
            e_key_s, e_ts_s = e_key[order], e_ts[order]
            idx = np.searchsorted(e_key_s, s_key)
            in_range = idx < e_key_s.size
            ok = np.zeros(s_key.size, dtype=bool)
            ok[in_range] = e_key_s[idx[in_range]] == s_key[in_range]
            t0 = s_ts[ok]
            durs = e_ts_s[idx[ok]] - t0
            b_out = blk[s_mask][ok]
            w_out = warp[s_mask][ok]

        elif mode == "preceding":
            # End carries no matching payload: pair each end with the most
            # recent begin on the same (block, warp) row.
            row = (blk << 8) | warp
            parts = []
            for r in np.unique(row):
                on_row = row == r
                sts = np.sort(ts[s_mask & on_row])
                ets = np.sort(ts[e_mask & on_row])
                if sts.size == 0 or ets.size == 0:
                    continue
                j = np.searchsorted(sts, ets, side="left") - 1
                valid = j >= 0
                t0v = sts[j[valid]]
                parts.append((int(r) >> 8, int(r) & 0xFF, t0v, ets[valid] - t0v))
            if not parts:
                continue
            b_out = np.concatenate([np.full(t.size, b, np.int64) for b, _, t, _ in parts])
            w_out = np.concatenate([np.full(t.size, w, np.int64) for _, w, t, _ in parts])
            t0 = np.concatenate([t for _, _, t, _ in parts])
            durs = np.concatenate([d for _, _, _, d in parts])
        else:
            raise ValueError(f"unknown pair_by mode {mode!r}")

        # Drop zero- and negative-duration spans: same-tick endpoints or a stray
        # end with no begin. Negatives would draw as inside-out bars.
        keep = durs > 0
        n = int(keep.sum())
        if n == 0:
            continue
        spans += list(
            zip(
                b_out[keep].tolist(),
                w_out[keep].tolist(),
                t0[keep].tolist(),
                durs[keep].tolist(),
                [color] * n,
                [label] * n,
            )
        )
    return spans


def spans_for_layout(records, name_to_id, layout: Layout):
    """Every role's phase table, applied to the warps that play that role."""
    spans = []
    for role in layout.role_order:
        spans += build_spans(records, name_to_id, layout.phases[role], warps=layout.warps(role))
    return spans


def summarize(spans):
    """Per-phase count and mean duration. A count that is not what the tile
    math predicts is a guard mismatch or a missing warp tag, not a slow kernel."""
    agg = {}
    for _, _, _, dur, _, label in spans:
        tot, n = agg.get(label, (0, 0))
        agg[label] = (tot + dur, n + 1)
    lines = [f"  {'phase':<24} {'count':>9} {'total us':>12} {'mean ns':>10}"]
    for label, (total_ns, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"  {label:<24} {n:>9} {total_ns / 1000.0:>12.1f} {total_ns / n:>10.1f}")
    return "\n".join(lines)


def rs_counters(records, name_to_id):
    """Ready-counter values seen by each RS warp -> (N, 6) int64 table of
    (block, warp, tile, threshold, first_poll, at_ready), or None for traces
    without the RS_COUNT_* events.

    The counter events carry a value in place of the tile, so each is matched
    to the RS_TILE_BEGIN / RS_READY its warp emitted just before it.
    """
    ids = {n: name_to_id.get(n) for n in
           ("RS_THRESHOLD", "RS_TILE_BEGIN", "RS_READY", "RS_COUNT_FIRST", "RS_COUNT_READY")}
    if any(v is None for v in ids.values()):
        return None
    pld = records[:, 3] & 0xFFFFFFFF
    warp, seq = pld >> WARP_ID_SHIFT, pld & SEQ_MASK
    rows, threshold, open_tile, last_row = [], {}, {}, {}
    # Rows are in ring order per block, which is emission order per warp.
    for blk, eid, w, s in zip(records[:, 0], records[:, 2], warp, seq):
        key = (int(blk), int(w))
        if eid == ids["RS_THRESHOLD"]:
            threshold[key] = int(s)
        elif eid == ids["RS_TILE_BEGIN"]:
            open_tile[key] = int(s)
        elif eid == ids["RS_COUNT_FIRST"]:
            last_row[key] = [*key, open_tile[key], threshold.get(key, -1), int(s), -1]
            rows.append(last_row[key])
        elif eid == ids["RS_COUNT_READY"]:
            last_row[key][5] = int(s)
    return np.asarray(rows, dtype=np.int64) if rows else None


def summarize_rs_counters(table, num_devices: int):
    """At the first poll a counter should sit at threshold - num_devices plus
    whatever peers have already published; anything at or past the threshold
    then was satisfied before this launch contributed."""
    if table is None:
        return "  (no RS counter events in this trace)"
    thr, first, ready = table[:, 3], table[:, 4], table[:, 5]
    base = thr - num_devices
    pre = int((first >= thr).sum())
    lines = [
        f"  RS counters: {len(table)} tiles, threshold {sorted(set(thr.tolist()))}",
        f"  first poll - (threshold - {num_devices}): "
        f"min {int((first - base).min())} max {int((first - base).max())}",
        f"  at ready - threshold: min {int((ready - thr).min())} max {int((ready - thr).max())}",
        f"  tiles already ready at first poll: {pre}",
    ]
    return "\n".join(lines)


# Mirrors fg::RED_BLOCK: per-K-step payloads are tile * (K // RED_BLOCK) + iter_k.
RED_BLOCK = 64


def kstep_table(records, name_to_id, k_steps: int):
    """Per-K-step MMA view -> dict of arrays, or None for traces without K events.

    One row per (MMA block, warp, tile, k): when the MMA warp started waiting
    for the stage's bytes, how long it waited, and when the last of the two
    cluster CTAs' producers got the stage free and issued its loads. MMA rows
    live on cta_rank 0 (even blocks); its cluster peer is block + 1.
    """
    ids = {n: name_to_id.get(n) for n in
           ("LOAD_K_BEGIN", "LOAD_K_STAGE_FREE", "MMA_K_BEGIN", "MMA_K_LOADED")}
    if any(v is None for v in ids.values()):
        return None
    blk, ts, eid = records[:, 0], records[:, 1], records[:, 2]
    pld = records[:, 3] & 0xFFFFFFFF
    warp, seq = pld >> WARP_ID_SHIFT, pld & SEQ_MASK

    def lookup(name, keys_of):
        m = eid == ids[name]
        return dict(zip(keys_of(m), ts[m].tolist()))

    mma_key = lambda m: zip(blk[m].tolist(), warp[m].tolist(), seq[m].tolist())
    begin = lookup("MMA_K_BEGIN", mma_key)
    loaded = lookup("MMA_K_LOADED", mma_key)
    issued = lookup("LOAD_K_STAGE_FREE", lambda m: zip(blk[m].tolist(), seq[m].tolist()))

    rows = []
    for (b, w, s), t0 in begin.items():
        t1 = loaded.get((b, w, s))
        p0, p1 = issued.get((b, s)), issued.get((b + 1, s))
        if t1 is None or p0 is None or p1 is None:
            continue
        rows.append((b, w, s // k_steps, s % k_steps, t0, t1 - t0, max(p0, p1)))
    if not rows:
        return None
    a = np.asarray(rows, dtype=np.int64)
    return dict(block=a[:, 0], warp=a[:, 1], tile=a[:, 2], k=a[:, 3],
                begin=a[:, 4], wait=a[:, 5], issued=a[:, 6])


def summarize_ksteps(table, stall_ns: int = 500):
    """Where MMA load stalls come from. A stall whose stage was issued long
    before the MMA got there is slow memory; one issued only just before is
    the producer being held back (stage not yet freed), i.e. pipeline depth."""
    if table is None:
        return "  (no per-K-step events in this trace)"
    wait = table["wait"]
    landed = table["begin"] + wait
    in_flight = landed - table["issued"]  # issue -> bytes landed, when the MMA waited
    stalled = wait > stall_ns
    pct = lambda x, q: np.percentile(x, q) / 1000.0 if x.size else float("nan")
    lines = [
        f"  MMA k steps: {wait.size}, wait for load us p50 {pct(wait, 50):.2f} "
        f"p90 {pct(wait, 90):.2f} p99 {pct(wait, 99):.2f} max {wait.max() / 1000.0:.2f}",
        f"  steps stalled > {stall_ns} ns: {int(stalled.sum())} "
        f"({100.0 * stalled.mean():.1f}%), {wait[stalled].sum() / 1000.0:.1f} us of stall in total",
        f"  issue -> landed on stalled steps us: p50 {pct(in_flight[stalled], 50):.2f} "
        f"p90 {pct(in_flight[stalled], 90):.2f} max "
        f"{(in_flight[stalled].max() / 1000.0) if stalled.any() else float('nan'):.2f}",
    ]
    return "\n".join(lines)


def wall_span_us(records):
    """Wall time covered by the trace. Should agree with the measured kernel
    time to within a few percent; a large gap is time in uninstrumented phases."""
    if records.shape[0] == 0:
        return 0.0
    return (records[:, 1].max() - records[:, 1].min()) / 1000.0


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def plot(records, out_path, name_to_id, layout: Layout, *, title="", max_height=48.0):
    """One Y-row per active (block, warp), grouped by role, as PolyCollections."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    if records.shape[0] == 0:
        return None

    records = records.copy()
    records[:, 1] -= records[:, 1].min()  # zero the time axis

    spans = spans_for_layout(records, name_to_id, layout)
    if not spans:
        return None

    # Row assignment: role groups in role_order, blank rows between them.
    active = {(b, w) for b, w, *_ in spans}
    row_to_y, y = {}, 0
    role_rows = {}
    for role in layout.role_order:
        members = sorted(k for k in active if layout.warp_roles.get(k[1]) == role)
        if not members:
            continue
        role_rows[role] = (y, y + len(members) - 1)
        for k in members:
            row_to_y[k] = y
            y += 1
        y += 2

    height = min(max(4.0, y * 0.12), max_height)
    fig, ax = plt.subplots(figsize=(16, height))

    # One PolyCollection per (color, label): every bar of a phase in a single
    # draw call. Rasterized, since a vector PDF of 10^5 bars is unusable.
    by_phase = {}
    for b, w, start, dur, color, label in spans:
        by_phase.setdefault((color, label), []).append(
            (row_to_y[(b, w)], start / 1000.0, dur / 1000.0)
        )
    for (color, label), bars in by_phase.items():
        arr = np.asarray(bars, dtype=np.float64)
        ys, xs, ws = arr[:, 0], arr[:, 1], arr[:, 2]
        h = 0.8
        verts = np.stack(
            [
                np.stack([xs, ys - h / 2], axis=1),
                np.stack([xs + ws, ys - h / 2], axis=1),
                np.stack([xs + ws, ys + h / 2], axis=1),
                np.stack([xs, ys + h / 2], axis=1),
            ],
            axis=1,
        )
        ax.add_collection(
            PolyCollection(
                verts, facecolors=color, edgecolors="none", linewidths=0,
                rasterized=True, label=label,
            )
        )

    for role, (lo, hi) in role_rows.items():
        ax.text(
            -0.01, (lo + hi) / 2, role, transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontweight="bold",
        )

    counts, totals = {}, {}
    for _, _, _, dur, _, label in spans:
        totals[label] = totals.get(label, 0) + dur / 1000.0
        counts[label] = counts.get(label, 0) + 1
    handles = [
        Patch(facecolor=c, label=f"{l} (n={counts[l]} avg={totals[l] / counts[l]:.2f}us)")
        for (c, l) in by_phase
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    xmax = max((s[2] + s[3]) for s in spans) / 1000.0
    ax.set_xlim(0, xmax)
    ax.set_ylim(y, -1)  # inverted: block 0 on top
    ax.set_xlabel("time (us)")
    ax.set_ylabel("CTA / warp")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.3)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def parse_blocks(spec: str):
    """'0-7,16' -> {0..7, 16}. Empty string -> None (keep everything)."""
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz", type=str, help="trace written by save()")
    ap.add_argument("--out", type=str, default="", help="output PDF (default: alongside the npz)")
    ap.add_argument(
        "--blocks", type=str, default="",
        help="only render these CTAs, e.g. '0-7,16'",
    )
    args = ap.parse_args(argv)

    records, name_to_id, meta = load(args.npz)
    kernel = str(meta["kernel"])
    if kernel not in LAYOUTS:
        print(f"no layout for kernel {kernel!r}; known: {sorted(LAYOUTS)}", file=sys.stderr)
        return 1
    layout = LAYOUTS[kernel]
    blocks = parse_blocks(args.blocks)
    if blocks is not None:
        records = records[np.isin(records[:, 0], sorted(blocks))]

    title = str(meta["title"]) if "title" in meta else Path(args.npz).name

    print(f"{records.shape[0]} events, wall span {wall_span_us(records):.1f} us")
    print(summarize(spans_for_layout(records, name_to_id, layout)))
    if "world_size" in meta:
        print(summarize_rs_counters(rs_counters(records, name_to_id), int(meta["world_size"])))
    if "K" in meta:
        k_steps = int(meta["K"]) // RED_BLOCK
        print(summarize_ksteps(kstep_table(records, name_to_id, k_steps)))

    out = Path(args.out) if args.out else Path(args.npz).with_suffix(".pdf")
    written = plot(records, out, name_to_id, layout, title=title)
    if written is None:
        print("no spans paired -- nothing to draw")
        return 1
    print(f"wrote {written} ({written.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
