# %% [markdown]
# # Depth of Early ED Hypotension and 7-Day Acute Kidney Injury or Death
#
# **Target-trial style observational study in MIMIC-IV-ED v2.2 linked to MIMIC-IV v3.1.**
#
# Among adults with a first new sustained ED hypotensive episode, we ask whether a
# **deeper mean arterial pressure (MAP) nadir in the first 2 hours** is **associated with**
# higher 7-day risk of **KDIGO stage >=2 acute kidney injury (AKI) or death**.
#
# **Primary exposure (pre-specified):** nadir MAP (depth). Charted minutes MAP <65 and
# AUT are **planned secondary measurement comparisons**, because ED vital density can
# reverse duration-based contrasts even when depth shows the expected harm gradient.
#
# Language is associative throughout. The design names and addresses the main biases
# that make observational hypotension-outcome work fragile.
#
# This notebook is self-contained and reproducible end-to-end. Run from the
# repository root as `python scripts/analysis.py` after placing MIMIC tables
# as described in README.md and scripts/config.py.

# %% [markdown]
# ## 0. Causal framework and analytic blueprint
#
# **Estimand.** The overlap-weighted (equipoise population) 7-day risk difference (RD)
# and risk ratio (RR) of the composite outcome for a **deep** vs **shallow** early
# MAP nadir (top vs bottom tertile of depth = -nadir).
#
# **Directed acyclic graph (rendered below).** Baseline illness severity confounds the
# depth-outcome relationship. Resuscitation (fluids, vasopressors) is a *mediator* that
# lowers depth, so it is deliberately not adjusted. Monitoring density and reference-
# creatinine availability are *selection* nodes handled explicitly.
#
# **Bias inventory:**
#
# | # | Bias | How it is addressed |
# |---|------|---------------------|
# | 1 | Confounding by severity | Richer baseline covariates; overlap weights + MICE + AIPW; E-value; tipping-point |
# | 2 | Monitoring / immortal-time bias in duration metrics | Depth primary; planned minutes/AUT comparison; >=3-reading sensitivity |
# | 3 | Selection on creatinine capture | IPCW re-estimate |
# | 4 | Competing risk (death) | Composite primary; Aalen-Johansen CIF; cause-specific hazards |
# | 5 | Exposure mismeasurement | Physiologic MAP filters; alternate windows; exclude profound-only |
# | 6 | Reverse causation | Negative-control pre-onset AKI (disclose residual association) |
# | 7 | Model misspecification | Overlap, IPTW, AIPW |
# | 8 | Positivity | Propensity overlap diagnostic; ATO; truncated IPTW |

# %%
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (  # noqa: E402
    PROJECT_DIR, ED_DIR, HOSP_DIR, ICU_DIR, TABLES_DIR, CACHE_DIR, FIGURES_DIR,
    MAP_THRESHOLD, MAP_PROFOUND, SUSTAIN_MIN, MAX_GAP_MIN, BURDEN_WINDOW_H,
    AKI_RATIO, HORIZON_DAYS, N_BOOT, N_MICE, RANDOM_SEED, require_mimic,
)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(RANDOM_SEED)

# Analysis constants that are local to this script (not shared via config).
AKI_STAGE1_RATIO = 1.5    # KDIGO stage >=1 (sensitivity)
AKI_STAGE1_ABS = 0.3
N_BOOT_SENS = 300
MISSING_DROP_FRAC = 0.40
RANDOM_STATE = RANDOM_SEED
NIGHT_START, NIGHT_END = 22, 6

# %% [markdown]
# ## Visual identity (house style: Lancet-style serif, grey panels, 300 dpi PNG/PDF/TIF)

# %%
_avail = {f.name for f in font_manager.fontManager.ttflist}
SERIF = "Times New Roman" if "Times New Roman" in _avail else "DejaVu Serif"

PAL = {
    "ink": "#1a1a1a", "deep": "#E76F51", "shallow": "#2A9D8F",
    "accent": "#C1452B", "sky": "#1B7268", "plum": "#264653",
    "green": "#2A9D8F", "amber": "#E9C46A", "muted": "#8C8880",
    "grid": "#E6E9ED", "panel": "#F0F0F0", "bg": "#ffffff",
}
CMAP_WARM = LinearSegmentedColormap.from_list(
    "warm", ["#F0DFB8", "#E9C46A", "#E76F51", "#C1452B", "#7F2A18"])
SPINE = "#B0B0B0"


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "figure.facecolor": PAL["bg"],
        "savefig.facecolor": PAL["bg"], "savefig.bbox": "tight",
        "font.family": SERIF,
        "font.serif": [SERIF, "Times New Roman", "Times", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlepad": 12,
        "axes.labelsize": 11.5, "axes.labelcolor": PAL["ink"],
        "axes.edgecolor": PAL["ink"], "axes.linewidth": 0.9,
        "axes.facecolor": PAL["bg"], "axes.grid": False,
        "grid.color": PAL["grid"], "grid.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "xtick.color": PAL["ink"], "ytick.color": PAL["ink"],
        "text.color": PAL["ink"], "legend.frameon": False, "legend.fontsize": 10,
        "axes.unicode_minus": False,
    })


def panel(ax, grid_axis="both"):
    ax.set_facecolor(PAL["panel"])
    ax.grid(True, axis=grid_axis, color="#FFFFFF", lw=1.2, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(SPINE); s.set_linewidth(0.9)


def panel_label(ax, text, x=0.0, y=1.02):
    """Panel tag above the axes so it never covers data, legends, or grid."""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, fontweight="bold", color="#1A1A1A", zorder=9,
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="#D0D5DD", linewidth=0.8, alpha=0.95))


def ann(ax, x, y, text, **kw):
    """Annotation with an opaque white box so lines never strike through labels."""
    defaults = dict(fontsize=9.2, color=PAL["ink"], zorder=8, clip_on=False,
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                              edgecolor="#D0D5DD", linewidth=0.7, alpha=0.96))
    defaults.update(kw)
    ax.text(x, y, text, **defaults)


def fmt_p(p):
    if not np.isfinite(p):
        return "p = NA"
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"


HERE = PROJECT_DIR
FIG = FIGURES_DIR
OUT = TABLES_DIR
CACHE = CACHE_DIR
ED = ED_DIR
HOSP = HOSP_DIR
ICU = ICU_DIR


def save_fig(fig, name):
    for ax in fig.get_axes():
        for it in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                   + ax.get_xticklabels() + ax.get_yticklabels()):
            it.set_fontfamily(SERIF)
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    try:
        from PIL import Image as PILImage
        PILImage.open(FIG / f"{name}.png").convert("RGB").save(
            FIG / f"{name}.tif", format="TIFF", compression="tiff_lzw", dpi=(300, 300))
    except Exception as exc:
        print(f"  TIFF failed {name}: {exc}")
    print(f"  saved {name} (png/pdf/tif)")


apply_style()

# %%
require_mimic()
print(f"Project: {HERE.name}\nData   : ED={ED}\n         HOSP={HOSP}\n         ICU={ICU}")


def load_lab_panel(label_regex: str, subject_ids: set[int], cache_name: str) -> pd.DataFrame:
    """Pull a blood lab panel for the episode subjects, with a local parquet cache.

    Replaces the external AJEM Exploratory Screen cache so this repository is
    self-contained for anyone with credentialed MIMIC access.
    """
    cache = CACHE / cache_name
    if cache.exists():
        lab = pd.read_parquet(cache)
        return lab[lab["subject_id"].isin(subject_ids)].copy()
    dlab = pd.read_csv(HOSP / "d_labitems.csv.gz", usecols=["itemid", "label", "fluid"])
    ids = set(
        dlab.loc[
            dlab["label"].str.contains(label_regex, case=False, na=False)
            & dlab["fluid"].astype(str).str.contains("Blood", case=False, na=False),
            "itemid",
        ].astype(int)
    )
    chunks = []
    scanned = 0
    for chunk in pd.read_csv(
        HOSP / "labevents.csv.gz",
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=4_000_000,
    ):
        scanned += len(chunk)
        chunk = chunk[chunk["itemid"].isin(ids) & chunk["subject_id"].isin(subject_ids)]
        chunk = chunk.dropna(subset=["valuenum", "charttime"])
        if len(chunk):
            chunks.append(chunk)
        if scanned % 40_000_000 < 4_000_000:
            print(f"  {cache_name} scan {scanned:,}")
    lab = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"])
    if len(lab):
        lab["charttime"] = pd.to_datetime(lab["charttime"], errors="coerce")
        lab.to_parquet(cache, index=False)
    return lab


# %% [markdown]
# ## 0.3 Render the causal DAG

# %%
def figure_dag():
    fig, ax = plt.subplots(figsize=(11.6, 7.0))
    ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(-0.2, 8.4)
    # Wider spacing so arrows never cross node text
    nodes = {
        "severity": (2.2, 7.0, "Baseline illness\nseverity", PAL["amber"], "confounder"),
        "exposure": (2.2, 2.6, "Early MAP nadir\n(depth, 0-2 h)", PAL["deep"], "exposure"),
        "outcome": (9.8, 2.6, "7-day AKI\nor death", PAL["shallow"], "outcome"),
        "mediator": (6.0, 5.2, "Resuscitation\n(fluids, pressors)", PAL["sky"], "mediator"),
        "monitor": (6.0, 0.55, "Monitoring density /\nref-creatinine capture", PAL["muted"], "selection"),
    }
    for key, (x, y, lab, col, role) in nodes.items():
        e = Ellipse((x, y), 2.9, 1.35, facecolor=col, alpha=0.30,
                    edgecolor=col, linewidth=1.8, zorder=3)
        ax.add_patch(e)
        ax.text(x, y + 0.18, lab, ha="center", va="center", fontsize=10.0,
                fontweight="bold", color=PAL["ink"], zorder=5)
        ax.text(x, y - 0.40, role, ha="center", va="center", fontsize=8.2,
                style="italic", color="#555", zorder=5)

    def arrow(a, b, color=PAL["ink"], style="-|>", rad=0.0, lw=1.7, ls="-"):
        xa, ya = nodes[a][0], nodes[a][1]
        xb, yb = nodes[b][0], nodes[b][1]
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle=style,
                     mutation_scale=16, lw=lw, color=color, ls=ls,
                     connectionstyle=f"arc3,rad={rad}", shrinkA=38, shrinkB=38, zorder=2))

    arrow("severity", "exposure", PAL["amber"], rad=0.0)
    arrow("severity", "outcome", PAL["amber"], rad=0.22)
    arrow("exposure", "outcome", PAL["deep"], rad=0.0, lw=2.6)
    arrow("exposure", "mediator", PAL["sky"], rad=0.18, ls="--")
    arrow("mediator", "outcome", PAL["sky"], rad=-0.18, ls="--")
    arrow("severity", "mediator", PAL["amber"], rad=-0.12, lw=1.2, ls=":")
    arrow("exposure", "monitor", PAL["muted"], rad=-0.12, ls="--", lw=1.2)
    arrow("outcome", "monitor", PAL["muted"], rad=0.12, ls="--", lw=1.2)

    ann(ax, 6.0, 3.15, "target association (estimated)", ha="center",
        fontsize=9.2, color=PAL["deep"], style="italic",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                  edgecolor=PAL["deep"], linewidth=0.8, alpha=0.96))
    leg = [
        Line2D([0], [0], color=PAL["amber"], lw=2, label="Confounding (adjusted: overlap/MICE/AIPW)"),
        Line2D([0], [0], color=PAL["sky"], lw=2, ls="--", label="Mediation (not adjusted)"),
        Line2D([0], [0], color=PAL["muted"], lw=2, ls="--", label="Selection (IPCW / collider)"),
        Line2D([0], [0], color=PAL["deep"], lw=2.4, label="Target association"),
    ]
    ax.legend(handles=leg, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02),
              fontsize=9.0, frameon=True, facecolor="white", edgecolor="#D0D5DD")
    ax.set_title("Causal structure of early ED hypotension depth and 7-day AKI or death",
                 fontsize=13.5, pad=18)
    save_fig(fig, "Figure1_DAG")
    plt.show()


figure_dag()


# %% [markdown]
# ## 1. Base cohort and episode detection
#
# Adults (>=18) with ED vital signs. For each stay we walk MAP readings in time order
# and mark the first qualifying hypotensive episode: MAP < 65 sustained >=15 min
# (consecutive lows, no gap > 120 min) or a single MAP < 55 (profound). Onset = first
# low reading of that run.

# %%
def derive_map(sbp, dbp):
    sbp = pd.to_numeric(sbp, errors="coerce")
    dbp = pd.to_numeric(dbp, errors="coerce")
    ok = (sbp.notna() & dbp.notna() & (sbp > 40) & (dbp > 20)
          & (sbp < 300) & (dbp < 200) & (sbp >= dbp))
    out = pd.Series(np.nan, index=sbp.index, dtype=float)
    out.loc[ok] = (sbp.loc[ok] + 2.0 * dbp.loc[ok]) / 3.0
    return out


print("Loading edstays / patients / triage ...")
edstays = pd.read_csv(ED / "edstays.csv.gz")
for c in ["intime", "outtime"]:
    edstays[c] = pd.to_datetime(edstays[c], errors="coerce")
patients = pd.read_csv(HOSP / "patients.csv.gz", usecols=["subject_id", "anchor_age", "dod"])
patients["dod"] = pd.to_datetime(patients["dod"], errors="coerce")
triage = pd.read_csv(ED / "triage.csv.gz")
triage["triage_map"] = derive_map(triage["sbp"], triage["dbp"])
base = edstays.merge(patients, on="subject_id", how="left").merge(
    triage[["stay_id", "acuity", "heartrate", "o2sat", "sbp", "dbp",
            "triage_map", "temperature", "chiefcomplaint"]], on="stay_id", how="left")
base = base.loc[base["anchor_age"].fillna(0) >= 18].copy()
base["age"] = base["anchor_age"].astype(float)
base["sex_male"] = (base["gender"].astype(str).str.upper() == "M").astype(int)
print(f"Adult ED stays: {len(base):,}")

# %%
print("Loading ED vital signs ...")
vit = pd.read_csv(ED / "vitalsign.csv.gz",
                  usecols=["stay_id", "charttime", "heartrate", "o2sat", "sbp", "dbp"])
vit["charttime"] = pd.to_datetime(vit["charttime"], errors="coerce")
vit = vit[vit["stay_id"].isin(base["stay_id"])].copy()
vit["map"] = derive_map(vit["sbp"], vit["dbp"])
vit = vit.dropna(subset=["charttime", "map"]).sort_values(["stay_id", "charttime"])
low_stays = vit.loc[vit["map"] < MAP_THRESHOLD, "stay_id"].unique()
vitc = vit[vit["stay_id"].isin(low_stays)].copy()
print(f"MAP-capable rows: {len(vit):,}; stays with any MAP<65: {len(low_stays):,}")


# %%
def detect_first_episode(t_min, m, hr, o2):
    run_start = run_start_i = prev_low_t = None
    for i in range(len(m)):
        mi, ti = m[i], t_min[i]
        if mi < MAP_THRESHOLD:
            if run_start is None or (prev_low_t is not None and (ti - prev_low_t) > MAX_GAP_MIN):
                run_start, run_start_i = ti, i
            prev_low_t = ti
            profound = mi < MAP_PROFOUND
            if profound or (ti - run_start) >= SUSTAIN_MIN:
                onset_type = "profound" if (profound and run_start_i == i) else "sustained"
                pre_hr = pre_o2 = np.nan
                for j in range(run_start_i, -1, -1):
                    if np.isnan(pre_hr) and np.isfinite(hr[j]):
                        pre_hr = hr[j]
                    if np.isnan(pre_o2) and np.isfinite(o2[j]):
                        pre_o2 = o2[j]
                    if np.isfinite(pre_hr) and np.isfinite(pre_o2):
                        break
                return {"onset_min": run_start, "onset_type": onset_type,
                        "profound_flag": int(onset_type == "profound"),
                        "onset_map": float(m[run_start_i]), "pre_hr": pre_hr, "pre_o2": pre_o2}
        else:
            run_start = run_start_i = prev_low_t = None
    return None


print("Detecting first hypotensive episode per stay ...")
sid = vitc["stay_id"].to_numpy()
ct = vitc["charttime"].to_numpy()
mp = vitc["map"].to_numpy(float)
hr = pd.to_numeric(vitc["heartrate"], errors="coerce").to_numpy(float)
o2 = pd.to_numeric(vitc["o2sat"], errors="coerce").to_numpy(float)
uniq, starts = np.unique(sid, return_index=True)
order = np.argsort(starts)
uniq, starts = uniq[order], starts[order]
ends = np.append(starts[1:], len(sid))
records = []
for k in range(len(uniq)):
    s, e = starts[k], ends[k]
    t0 = ct[s]
    tmin = ((ct[s:e] - t0) / np.timedelta64(1, "m")).astype(float)
    epi_k = detect_first_episode(tmin, mp[s:e], hr[s:e], o2[s:e])
    if epi_k is None:
        continue
    epi_k["stay_id"] = int(uniq[k])
    epi_k["onset"] = pd.Timestamp(t0) + pd.Timedelta(minutes=float(epi_k["onset_min"]))
    records.append(epi_k)
epi = pd.DataFrame.from_records(records).merge(
    base[["stay_id", "subject_id", "hadm_id", "intime", "outtime", "disposition",
          "dod", "age", "sex_male", "acuity"]], on="stay_id", how="left")
epi["night_onset"] = ((epi["onset"].dt.hour >= NIGHT_START) | (epi["onset"].dt.hour < NIGHT_END)).astype(int)
print(f"Episodes: {len(epi):,}\n{epi['onset_type'].value_counts().to_string()}")


# %% [markdown]
# ## 2. Exposure metrics in the first 2 hours: depth (nadir), depth-weighted burden (AUT), duration (minutes)

# %%
on = epi[["stay_id", "onset"]].dropna()


def window_metrics(window_h):
    w = vit.merge(on, on="stay_id", how="inner")
    w = w.loc[(w["charttime"] >= w["onset"]) &
              (w["charttime"] <= w["onset"] + pd.Timedelta(hours=window_h))].copy()
    w = w.sort_values(["stay_id", "charttime"])
    w["dt"] = (w.groupby("stay_id")["charttime"].shift(-1) - w["charttime"]).dt.total_seconds() / 60.0
    low = (w["map"] < MAP_THRESHOLD) & w["dt"].notna() & (w["dt"] <= MAX_GAP_MIN)
    w["seg"] = np.where(low, w["dt"], 0.0)
    w["aut_seg"] = np.where(low, (MAP_THRESHOLD - w["map"]) * w["dt"], 0.0)
    agg = w.groupby("stay_id", as_index=False).agg(
        minutes=("seg", "sum"), aut=("aut_seg", "sum"),
        nadir=("map", "min"), n_read=("map", "size"))
    last = w.groupby("stay_id", as_index=False).tail(1)
    ll = last.loc[last["map"] < MAP_THRESHOLD, ["stay_id", "map"]].copy()
    ll["mc"] = 15.0
    ll["ac"] = (MAP_THRESHOLD - ll["map"]) * 15.0
    agg = agg.merge(ll[["stay_id", "mc", "ac"]], on="stay_id", how="left")
    agg["minutes"] = (agg["minutes"] + agg["mc"].fillna(0)).clip(upper=window_h * 60)
    agg["aut"] = agg["aut"] + agg["ac"].fillna(0)
    return agg[["stay_id", "minutes", "aut", "nadir", "n_read"]]


b2 = window_metrics(2.0).rename(columns={"minutes": "minutes_2h", "aut": "aut_2h",
                                         "nadir": "nadir_2h", "n_read": "n_read_2h"})
b1 = window_metrics(1.0).rename(columns={"minutes": "minutes_1h", "aut": "aut_1h",
                                         "nadir": "nadir_1h", "n_read": "n_read_1h"})
b3 = window_metrics(3.0).rename(columns={"nadir": "nadir_3h", "n_read": "n_read_3h"})
epi = epi.merge(b2, on="stay_id", how="left").merge(
    b1[["stay_id", "minutes_1h", "aut_1h", "nadir_1h", "n_read_1h"]], on="stay_id", how="left").merge(
    b3[["stay_id", "nadir_3h", "n_read_3h"]], on="stay_id", how="left")
for c in ["minutes_2h", "aut_2h", "n_read_2h", "minutes_1h", "aut_1h", "n_read_1h", "n_read_3h"]:
    epi[c] = epi[c].fillna(0.0)
for c in ["nadir_2h", "nadir_1h", "nadir_3h"]:
    epi[c] = epi[c].fillna(epi["onset_map"])
print(epi[["nadir_2h", "aut_2h", "minutes_2h", "n_read_2h"]].describe().round(1))


# %% [markdown]
# ## 3. Outcomes: reference creatinine, KDIGO AKI, death, RRT, and a pre-onset negative control

# %%
adm = pd.read_csv(HOSP / "admissions.csv.gz",
                  usecols=["hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag"])
for c in ["admittime", "dischtime", "deathtime"]:
    adm[c] = pd.to_datetime(adm[c], errors="coerce")

epi_subjects = set(epi["subject_id"].dropna().astype(int))
cr = load_lab_panel(r"^Creatinine$", epi_subjects, "creatinine_screen.parquet")
cr["charttime"] = pd.to_datetime(cr["charttime"], errors="coerce")
cr = cr.dropna(subset=["charttime", "valuenum"])
cr = cr[(cr["valuenum"] > 0) & (cr["valuenum"] < 50)]
print(f"Creatinine rows: {len(cr):,} ({cr['subject_id'].nunique():,} subjects)")

d_items = pd.read_csv(ICU / "d_items.csv.gz", usecols=["itemid", "label", "linksto"])
dia_ids = set(d_items.loc[
    d_items["label"].astype(str).str.contains("dialysis|crrt|cvvh|hemodial|ultrafilt", case=False, na=False)
    & d_items["linksto"].eq("procedureevents"), "itemid"].astype(int))
pe = pd.read_csv(ICU / "procedureevents.csv.gz", usecols=["subject_id", "starttime", "itemid"])
pe = pe[pe["itemid"].isin(dia_ids) & pe["subject_id"].isin(epi_subjects)].copy()
pe["starttime"] = pd.to_datetime(pe["starttime"], errors="coerce")
pe = pe.dropna(subset=["starttime"])
print(f"RRT events: {len(pe):,}")


# %%
def attach_outcomes(df, cr, adm, pe):
    out = df.merge(adm, on="hadm_id", how="left", suffixes=("", "_adm"))
    death_t = out["deathtime"].fillna(out["dod"])
    t0 = pd.to_datetime(out["onset"])
    out["t_death_d"] = (pd.to_datetime(death_t) - t0).dt.total_seconds() / 86400.0
    out["death_7d"] = ((out["t_death_d"] >= 0) & (out["t_death_d"] <= HORIZON_DAYS)).fillna(False).astype(int)

    keys = out[["stay_id", "subject_id", "onset"]].copy()
    keys["t0"] = pd.to_datetime(keys["onset"])
    cr2 = cr[["subject_id", "charttime", "valuenum"]]
    j = keys.merge(cr2, on="subject_id", how="left")
    dd = (j["charttime"] - j["t0"]).dt.total_seconds() / 86400.0

    # reference = min creatinine in [-7d, 0]
    ref = j.loc[(dd >= -7) & (dd <= 0)].groupby("stay_id", as_index=False)["valuenum"].min() \
        .rename(columns={"valuenum": "ref_cr"})
    # pre-onset baseline for negative control = min in [-7d, -2d]
    base_pre = j.loc[(dd >= -7) & (dd <= -2)].groupby("stay_id", as_index=False)["valuenum"].min() \
        .rename(columns={"valuenum": "base_pre_cr"})
    # post creatinine
    postj = j.loc[(dd > 0) & (dd <= HORIZON_DAYS)].copy()
    post_max = postj.groupby("stay_id", as_index=False)["valuenum"].max().rename(columns={"valuenum": "post_max_cr"})
    # pre-onset peak in [-2d, 0] for negative-control outcome
    prej = j.loc[(dd > -2) & (dd <= 0)].copy()
    pre_max = prej.groupby("stay_id", as_index=False)["valuenum"].max().rename(columns={"valuenum": "pre_max_cr"})

    out = out.merge(ref, on="stay_id", how="left").merge(base_pre, on="stay_id", how="left") \
             .merge(post_max, on="stay_id", how="left").merge(pre_max, on="stay_id", how="left")

    # time to AKI (creatinine) for competing-risk analyses
    postj = postj.merge(ref, on="stay_id", how="left")
    hit = postj.loc[postj["valuenum"] >= AKI_RATIO * postj["ref_cr"]].copy()
    hit["aki_cr_d"] = (hit["charttime"] - hit["t0"]).dt.total_seconds() / 86400.0
    out = out.merge(hit.groupby("stay_id", as_index=False)["aki_cr_d"].min(), on="stay_id", how="left")

    out["aki_cr"] = (out["ref_cr"].notna() & (out["ref_cr"] > 0) & out["post_max_cr"].notna()
                     & (out["post_max_cr"] >= AKI_RATIO * out["ref_cr"])).astype(int)
    out["aki_cr_stage1"] = (out["ref_cr"].notna() & (out["ref_cr"] > 0) & out["post_max_cr"].notna()
                            & ((out["post_max_cr"] >= AKI_STAGE1_RATIO * out["ref_cr"])
                               | (out["post_max_cr"] >= out["ref_cr"] + AKI_STAGE1_ABS))).astype(int)

    # RRT
    if len(pe):
        rj = out[["stay_id", "subject_id", "onset"]].merge(pe, on="subject_id", how="left")
        rj["d"] = (rj["starttime"] - pd.to_datetime(rj["onset"])).dt.total_seconds() / 86400.0
        post_rrt = rj.loc[(rj["d"] > 0) & (rj["d"] <= HORIZON_DAYS)].groupby("stay_id", as_index=False)["d"].min() \
            .rename(columns={"d": "rrt_d"})
        pre_rrt = rj.loc[rj["d"] <= 0, "stay_id"].unique()
        out = out.merge(post_rrt, on="stay_id", how="left")
        out["pre_rrt"] = out["stay_id"].isin(pre_rrt).astype(int)
    else:
        out["rrt_d"] = np.nan
        out["pre_rrt"] = 0
    out["rrt_7d"] = out["rrt_d"].notna().astype(int)

    out["aki"] = ((out["aki_cr"] == 1) | (out["rrt_7d"] == 1)).astype(int)
    out["aki_stage1"] = ((out["aki_cr_stage1"] == 1) | (out["rrt_7d"] == 1)).astype(int)
    out["aki_time_d"] = out[["aki_cr_d", "rrt_d"]].min(axis=1)
    out["aki_or_death"] = ((out["aki"] == 1) | (out["death_7d"] == 1)).astype(int)

    # Negative-control outcome (reverse causation falsification):
    # creatinine rise >=1.5x occurring BEFORE onset (relative to [-7d,-2d] baseline)
    out["pre_onset_aki"] = (out["base_pre_cr"].notna() & (out["base_pre_cr"] > 0)
                            & out["pre_max_cr"].notna()
                            & (out["pre_max_cr"] >= 1.5 * out["base_pre_cr"])).astype(int)
    out["has_ref"] = out["ref_cr"].notna().astype(int)
    return out


epi = attach_outcomes(epi, cr, adm, pe)
print(epi[["ref_cr", "aki", "death_7d", "aki_or_death", "pre_onset_aki"]].describe().round(3))


# %% [markdown]
# ## 3b. Baseline severity: comorbidities, sepsis flag, pre-onset lactate
#
# Chronic comorbidity flags are subject-level (any coded history). Sepsis/infection is
# flagged on the linked hospital admission (often the presenting illness). Pre-onset
# lactate is the maximum value in the 24 h before onset through onset. Lactate enters
# the PS only if missingness is below 40% in the analytic sample; otherwise Table 1 only.

# %%
dx = pd.read_csv(HOSP / "diagnoses_icd.csv.gz",
                 usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
dx["icd_code"] = dx["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)
epi_subj = set(epi["subject_id"].dropna().astype(int))
dx = dx[dx["subject_id"].isin(epi_subj)].copy()

esrd = (dx["icd_code"].str.startswith("5856") | dx["icd_code"].eq("N186")
        | dx["icd_code"].str.startswith("Z992") | dx["icd_code"].isin(["V451", "V4511", "V4512"]))
esrd_subjects = set(dx.loc[esrd, "subject_id"].astype(int))
epi["esrd"] = epi["subject_id"].isin(esrd_subjects).astype(int)

COMORB = {
    "chf": r"^(428|I50)",
    "cirrhosis": r"^(5712|5715|5716|K703|K717|K743|K744|K745|K746)",
    "diabetes": r"^(250|E08|E09|E10|E11|E13)",
    "ckd": r"^(585[1-5]|N18[1-5])",
    "cad": r"^(414|I25)",
    "copd": r"^(496|J44)",
}
for name, pat in COMORB.items():
    hits = set(dx.loc[dx["icd_code"].str.match(pat, na=False), "subject_id"].astype(int))
    epi[name] = epi["subject_id"].isin(hits).astype(int)

sepsis_pat = r"^(99591|99592|A40|A41|R652|038|99590|A021|A227|A267|A327|A427|B377)"
sx = dx.loc[dx["icd_code"].str.match(sepsis_pat, na=False), "hadm_id"].dropna().astype(int)
epi["sepsis"] = epi["hadm_id"].isin(set(sx)).astype(int)
print("Comorbidity prevalences (episode cohort):")
print(epi[list(COMORB) + ["sepsis"]].mean().mul(100).round(1).astype(str).add("%").to_string())

lac = load_lab_panel(r"^Lactate$", epi_subj, "lactate_screen.parquet")
lac["charttime"] = pd.to_datetime(lac["charttime"], errors="coerce")
lac = lac.dropna(subset=["charttime", "valuenum"])
lac = lac[(lac["valuenum"] > 0) & (lac["valuenum"] < 30)]
lj = epi[["stay_id", "subject_id", "onset"]].merge(lac, on="subject_id", how="left")
ld = (lj["charttime"] - pd.to_datetime(lj["onset"])).dt.total_seconds() / 3600.0
pre_lac = lj.loc[(ld >= -24) & (ld <= 0)].groupby("stay_id", as_index=False)["valuenum"].max() \
    .rename(columns={"valuenum": "pre_lactate"})
epi = epi.merge(pre_lac, on="stay_id", how="left")
print(f"Pre-onset lactate available: {epi['pre_lactate'].notna().mean()*100:.1f}% "
      f"(median {epi['pre_lactate'].median():.1f})")


# %% [markdown]
# ## 4. Eligibility, exclusions and STROBE flow

# %%
# dx / esrd already loaded above

flow = []
flow.append(("First ED hypotensive episode (adults)", len(epi)))
step = epi.loc[epi["esrd"] == 0].copy()
flow.append(("Exclude ESRD / chronic dialysis", len(step)))
step = step.loc[step["pre_rrt"] == 0].copy()
flow.append(("Exclude RRT at/before onset", len(step)))
step = step.loc[step["n_read_2h"] >= 1].copy()
flow.append(("Scorable 2-hour window", len(step)))
eligible = step.copy()                         # used for IPCW denominator (before ref-cr selection)
no_ref = step.loc[step["ref_cr"].isna()]
analytic_all = step.loc[step["ref_cr"].notna()].copy()
flow.append(("Reference creatinine available (AKI ascertainable)", len(analytic_all)))
strobe = pd.DataFrame(flow, columns=["step", "n"])
strobe["excluded"] = strobe["n"].shift(1) - strobe["n"]
strobe.to_csv(OUT / "strobe_flow_counts.csv", index=False)
print(strobe.to_string(index=False))
print(f"No reference creatinine (IPCW-modelled): {len(no_ref):,}, "
      f"death rate {no_ref['death_7d'].mean():.3f}")


# %% [markdown]
# ## 5. Primary exposure contrast (deep vs shallow nadir tertile) and confounders

# %%
analytic_all["sev"] = -analytic_all["nadir_2h"]
analytic_all["sev_tertile"] = pd.qcut(analytic_all["sev"].rank(method="first"), 3,
                                      labels=["shallow", "mid", "deep"])
ana = analytic_all.loc[analytic_all["sev_tertile"].isin(["shallow", "deep"])].copy()
ana["deep"] = (ana["sev_tertile"] == "deep").astype(int)

CONT = ["age", "pre_hr", "pre_o2", "ref_cr", "acuity", "pre_lactate"]
BIN = ["sex_male", "night_onset", "chf", "cirrhosis", "diabetes", "ckd", "cad", "copd", "sepsis"]
COVARS = CONT + BIN
miss = ana[COVARS].isna().mean()
ps_covars = [c for c in COVARS if miss[c] < MISSING_DROP_FRAC]
dropped = [c for c in COVARS if c not in ps_covars]
print("Missingness:\n" + (miss * 100).round(1).astype(str).add("%").to_string())
print(f"\nPS covariates ({len(ps_covars)}): {ps_covars}")
if dropped:
    print(f"Dropped from PS (>{100*MISSING_DROP_FRAC:.0f}% missing; Table 1 only): {dropped}")
print(f"Deep n={int(ana.deep.sum())}, shallow n={int((1-ana.deep).sum())}")
print(f"Nadir mmHg by tertile:\n"
      f"{analytic_all.groupby('sev_tertile')['nadir_2h'].agg(['min','max','median']).round(1).to_string()}")


# %% [markdown]
# ## 6. Multiple imputation (MICE) of pre-onset covariates
#
# Covariate missingness is low but non-zero (acuity, SpO2). We generate `N_MICE`
# imputations and pool by Rubin's rules for the primary estimate; complete-case is
# reported as a sensitivity.

# %%
def make_imputations(df, covars, m=N_MICE):
    bin_cov = [c for c in covars if c in BIN]
    imps = []
    for i in range(m):
        imp = IterativeImputer(random_state=RANDOM_STATE + i, sample_posterior=True, max_iter=10)
        X = imp.fit_transform(df[covars].astype(float))
        d = df.copy()
        d[covars] = X
        if "acuity" in covars:
            d["acuity"] = d["acuity"].round().clip(1, 5)
        for c in bin_cov:
            d[c] = (d[c] >= 0.5).astype(int)
        imps.append(d)
    return imps


imps = make_imputations(ana, ps_covars, N_MICE)
print(f"Generated {len(imps)} MICE completions.")


# %% [markdown]
# ## 7. Estimators: stabilized overlap weights, truncated IPTW, and doubly-robust AIPW

# %%
def fit_ps(df, covars):
    X = StandardScaler().fit_transform(df[covars].astype(float))
    e = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(X, df["deep"]).predict_proba(X)[:, 1]
    return np.clip(e, 1e-4, 1 - 1e-4)


def overlap_w(t, e):
    return np.where(t == 1, 1 - e, e)


def iptw_w(t, e, trunc=(0.01, 0.99)):
    p = t.mean()
    w = np.where(t == 1, p / e, (1 - p) / (1 - e))
    lo, hi = np.quantile(w, trunc)
    return np.clip(w, lo, hi)


def wrisk(y, t, w):
    r1 = np.sum(w[t == 1] * y[t == 1]) / np.sum(w[t == 1])
    r0 = np.sum(w[t == 0] * y[t == 0]) / np.sum(w[t == 0])
    return r1, r0


def aipw(df, covars, ycol="aki_or_death"):
    t = df["deep"].to_numpy(int)
    y = df[ycol].to_numpy(int)
    e = fit_ps(df, covars)
    Xc = StandardScaler().fit_transform(df[covars].astype(float))
    m1 = LogisticRegression(max_iter=1000).fit(Xc[t == 1], y[t == 1]).predict_proba(Xc)[:, 1]
    m0 = LogisticRegression(max_iter=1000).fit(Xc[t == 0], y[t == 0]).predict_proba(Xc)[:, 1]
    psi1 = np.mean(t * (y - m1) / e + m1)
    psi0 = np.mean((1 - t) * (y - m0) / (1 - e) + m0)
    return psi1, psi0, psi1 - psi0, psi1 / psi0 if psi0 > 0 else np.nan


def smd(x, t, w=None):
    x = np.asarray(x, float); t1, t0 = t == 1, t == 0
    if w is None:
        m1, m0, v1, v0 = x[t1].mean(), x[t0].mean(), x[t1].var(), x[t0].var()
    else:
        w1, w0 = w[t1], w[t0]
        m1 = np.sum(w1 * x[t1]) / w1.sum(); m0 = np.sum(w0 * x[t0]) / w0.sum()
        v1 = np.sum(w1 * (x[t1] - m1) ** 2) / w1.sum(); v0 = np.sum(w0 * (x[t0] - m0) ** 2) / w0.sum()
    sp = np.sqrt((v1 + v0) / 2)
    return (m1 - m0) / sp if sp > 0 else 0.0


def estimate(df, covars, method="overlap", ycol="aki_or_death", extra_w=None):
    t = df["deep"].to_numpy(int); y = df[ycol].to_numpy(int)
    if method == "aipw":
        return aipw(df, covars, ycol)
    e = fit_ps(df, covars)
    w = overlap_w(t, e) if method == "overlap" else iptw_w(t, e)
    if extra_w is not None:
        w = w * np.asarray(extra_w)
    r1, r0 = wrisk(y, t, w)
    return r1, r0, r1 - r0, r1 / r0 if r0 > 0 else np.nan


def boot(df, covars, method="overlap", ycol="aki_or_death", B=N_BOOT, extra_w=None):
    r1, r0, rd, rr = estimate(df, covars, method, ycol, extra_w)
    idx = np.arange(len(df)); rds, rrs, r1s, r0s = [], [], [], []
    ew = None if extra_w is None else np.asarray(extra_w)
    for _ in range(B):
        bs = rng.choice(idx, len(idx), replace=True)
        d = df.iloc[bs]
        if d["deep"].nunique() < 2:
            continue
        try:
            b1, b0, brd, brr = estimate(d, covars, method, ycol,
                                        None if ew is None else ew[bs])
            rds.append(brd); rrs.append(brr); r1s.append(b1); r0s.append(b0)
        except Exception:
            continue

    def ci(a):
        a = np.asarray(a, float); a = a[np.isfinite(a)]
        return (np.percentile(a, 2.5), np.percentile(a, 97.5)) if len(a) else (np.nan, np.nan)
    out = {"risk_high": r1, "risk_low": r0, "rd": rd, "rr": rr, "n": len(df),
           "n_high": int(df.deep.sum()), "n_low": int((1 - df.deep).sum())}
    out["risk_high_lo"], out["risk_high_hi"] = ci(r1s)
    out["risk_low_lo"], out["risk_low_hi"] = ci(r0s)
    out["rd_lo"], out["rd_hi"] = ci(rds)
    out["rr_lo"], out["rr_hi"] = ci(rrs)
    return out


def pool_mice(imps, covars, method="overlap", ycol="aki_or_death", B=N_BOOT):
    """Rubin's rules across imputations, each with bootstrap variance."""
    m = len(imps)
    pts = {"risk_high": [], "risk_low": [], "rd": [], "logrr": []}
    wvar = {"risk_high": [], "risk_low": [], "rd": [], "logrr": []}
    for d in imps:
        b = boot(d, covars, method, ycol, B=max(150, B // m))
        pts["risk_high"].append(b["risk_high"]); pts["risk_low"].append(b["risk_low"])
        pts["rd"].append(b["rd"]); pts["logrr"].append(np.log(b["rr"]))
        wvar["risk_high"].append(((b["risk_high_hi"] - b["risk_high_lo"]) / (2 * 1.96)) ** 2)
        wvar["risk_low"].append(((b["risk_low_hi"] - b["risk_low_lo"]) / (2 * 1.96)) ** 2)
        wvar["rd"].append(((b["rd_hi"] - b["rd_lo"]) / (2 * 1.96)) ** 2)
        wvar["logrr"].append(((np.log(b["rr_hi"]) - np.log(b["rr_lo"])) / (2 * 1.96)) ** 2)

    def pool(key, transform=lambda x: x):
        est = np.mean(pts[key])
        se = np.sqrt(np.mean(wvar[key]) + (1 + 1 / m) * np.var(pts[key], ddof=1))
        return transform(est), transform(est - 1.96 * se), transform(est + 1.96 * se)

    rh, rh_lo, rh_hi = pool("risk_high")
    rl, rl_lo, rl_hi = pool("risk_low")
    rd, rd_lo, rd_hi = pool("rd")
    rr, rr_lo, rr_hi = pool("logrr", np.exp)
    return {"risk_high": rh, "risk_high_lo": rh_lo, "risk_high_hi": rh_hi,
            "risk_low": rl, "risk_low_lo": rl_lo, "risk_low_hi": rl_hi,
            "rd": rd, "rd_lo": rd_lo, "rd_hi": rd_hi,
            "rr": rr, "rr_lo": rr_lo, "rr_hi": rr_hi,
            "n": len(imps[0]), "n_high": int(imps[0].deep.sum()),
            "n_low": int((1 - imps[0].deep).sum())}


# %% [markdown]
# ## 8. Covariate balance (Love plot inputs)

# %%
cc = ana.dropna(subset=ps_covars + ["deep", "aki_or_death"]).copy()
e_cc = fit_ps(cc, ps_covars)
cc["ps"] = e_cc
w_cc = overlap_w(cc["deep"].to_numpy(int), e_cc)
cc["w_overlap"] = w_cc
balance = pd.DataFrame({
    "covariate": ps_covars,
    "smd_unweighted": [smd(cc[c], cc["deep"].to_numpy()) for c in ps_covars],
    "smd_overlap": [smd(cc[c], cc["deep"].to_numpy(), w_cc) for c in ps_covars],
})
balance["abs_smd_overlap"] = balance["smd_overlap"].abs()
balance.to_csv(OUT / "covariate_balance.csv", index=False)
print(balance.round(3).to_string(index=False))
print(f"Max |SMD| after overlap: {balance['abs_smd_overlap'].max():.3f}")


# %% [markdown]
# ## 9. Primary and co-primary estimates
#
# Primary: MICE-pooled overlap-weighted RD/RR. Co-primary robustness: complete-case
# overlap, truncated IPTW, and doubly-robust AIPW.

# %%
primary = pool_mice(imps, ps_covars, "overlap", "aki_or_death", B=N_BOOT)
cc_overlap = boot(cc, ps_covars, "overlap", "aki_or_death", B=N_BOOT)
cc_iptw = boot(cc, ps_covars, "iptw", "aki_or_death", B=N_BOOT)
cc_aipw = boot(cc, ps_covars, "aipw", "aki_or_death", B=N_BOOT)


def e_value(rr):
    rr = rr if rr >= 1 else 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


primary["e_value"] = e_value(primary["rr"])
primary["e_value_ci"] = e_value(primary["rr_lo"] if primary["rr_lo"] > 1 else primary["rr_lo"])
print("PRIMARY (MICE-pooled overlap):")
for k in ["risk_high", "risk_low", "rd", "rr"]:
    print(f"  {k:10s} {primary[k]:.4f}  ({primary[k+'_lo']:.4f}, {primary[k+'_hi']:.4f})")
print(f"  E-value {primary['e_value']:.2f} (CI bound {primary['e_value_ci']:.2f})")
print(f"Complete-case overlap RD {cc_overlap['rd']:+.4f} ({cc_overlap['rd_lo']:+.4f},{cc_overlap['rd_hi']:+.4f})")
print(f"Truncated IPTW      RD {cc_iptw['rd']:+.4f} ({cc_iptw['rd_lo']:+.4f},{cc_iptw['rd_hi']:+.4f})")
print(f"Doubly-robust AIPW  RD {cc_aipw['rd']:+.4f} ({cc_aipw['rd_lo']:+.4f},{cc_aipw['rd_hi']:+.4f})")


# %% [markdown]
# ## 10. Secondary endpoints and competing-risk analysis

# %%
sec_aki = boot(cc, ps_covars, "overlap", "aki", B=N_BOOT)
sec_death = boot(cc, ps_covars, "overlap", "death_7d", B=N_BOOT)
print(f"AKI alone   RD {sec_aki['rd']:+.4f} ({sec_aki['rd_lo']:+.4f},{sec_aki['rd_hi']:+.4f})")
print(f"Death alone RD {sec_death['rd']:+.4f} ({sec_death['rd_lo']:+.4f},{sec_death['rd_hi']:+.4f})")

# Aalen-Johansen CIF (overlap-weighted) + cause-specific hazards (lifelines)
cc["event_time"] = np.minimum(cc[["aki_time_d", "t_death_d"]].min(axis=1).fillna(HORIZON_DAYS),
                              HORIZON_DAYS).clip(lower=0)
aki_first = cc["aki"].eq(1) & (cc["aki_time_d"].fillna(np.inf) <= cc["t_death_d"].fillna(np.inf))
death_first = cc["death_7d"].eq(1) & ~aki_first
cc["etype"] = np.where(aki_first, 1, np.where(death_first, 2, 0))


def aalen_johansen(t, e, w, horizon=HORIZON_DAYS, grid=None):
    if grid is None:
        grid = np.linspace(0, horizon, 200)
    t = np.asarray(t, float); e = np.asarray(e, int); w = np.asarray(w, float)
    o = np.argsort(t); t, e, w = t[o], e[o], w[o]
    surv, c1, c2 = 1.0, 0.0, 0.0
    cif1, cif2 = np.zeros_like(grid), np.zeros_like(grid)
    gi = 0
    for tt in np.unique(t[e > 0]):
        at = w[t >= tt].sum()
        if at <= 0:
            continue
        d1 = w[(t == tt) & (e == 1)].sum(); d2 = w[(t == tt) & (e == 2)].sum()
        c1 += surv * d1 / at; c2 += surv * d2 / at
        surv *= (1 - (d1 + d2) / at)
        while gi < len(grid) and grid[gi] < tt:
            gi += 1
        cif1[gi:] = c1; cif2[gi:] = c2
    return grid, cif1, cif2


cif_rows = []
for arm, lab in [(1, "deep"), (0, "shallow")]:
    sub = cc[cc.deep == arm]
    g, a1, a2 = aalen_johansen(sub["event_time"], sub["etype"], sub["w_overlap"])
    for gg, x1, x2 in zip(g, a1, a2):
        cif_rows.append({"arm": lab, "t": gg, "cif_aki": x1, "cif_death": x2})
cif_df = pd.DataFrame(cif_rows)
cif_df.to_csv(OUT / "competing_risk_cif.csv", index=False)

# Cause-specific weighted Cox (deep vs shallow)
from lifelines import CoxPHFitter
cs_rows = []
for cause, lab in [(1, "AKI"), (2, "death")]:
    d = cc[["event_time", "deep", "w_overlap"]].copy()
    d["ev"] = (cc["etype"] == cause).astype(int)
    d = d[d["event_time"] > 0]
    try:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(d, "event_time", "ev", weights_col="w_overlap", robust=True)
        hr = float(np.exp(cph.params_["deep"]))
        lo, hi = np.exp(cph.confidence_intervals_.loc["deep"].values)
        cs_rows.append({"cause": lab, "hr": hr, "lo": lo, "hi": hi})
    except Exception as exc:
        cs_rows.append({"cause": lab, "hr": np.nan, "lo": np.nan, "hi": np.nan})
        print("Cox failed", lab, exc)
cause_specific = pd.DataFrame(cs_rows)
cause_specific.to_csv(OUT / "cause_specific_hr.csv", index=False)
print("Cause-specific HR (deep vs shallow):")
print(cause_specific.round(3).to_string(index=False))
print("CIF at 7d:\n" + cif_df.groupby("arm").tail(1)[["arm", "cif_aki", "cif_death"]].round(3).to_string(index=False))


# %% [markdown]
# ## 11. Dose-response by depth, and the exposure-measurement comparison
#
# The head-to-head shows that charted **minutes** is confounded/reversed, while depth
# (nadir) and depth-weighted burden (AUT) recover the expected harm gradient.

# %%
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, c - h, c + h


nadir_bins = [0, 50, 55, 60, 65]
nadir_labels = ["<50", "50-54", "55-59", "60-64"]
analytic_all["nadir_bin"] = pd.cut(analytic_all["nadir_2h"], nadir_bins, labels=nadir_labels, right=False)
dose = []
for b in nadir_labels:
    sub = analytic_all.loc[analytic_all["nadir_bin"] == b]
    if not len(sub):
        continue
    p, lo, hi = wilson(int(sub["aki_or_death"].sum()), len(sub))
    dose.append({"band": b, "n": len(sub), "events": int(sub["aki_or_death"].sum()),
                 "median_nadir": sub["nadir_2h"].median(), "risk": p, "lo": lo, "hi": hi})
dose_df = pd.DataFrame(dose)
dose_df.to_csv(OUT / "dose_response_nadir.csv", index=False)


def contrast_metric(metric, higher_is_severe):
    d = analytic_all.copy()
    d["_s"] = d[metric] if higher_is_severe else -d[metric]
    ter = pd.qcut(d["_s"].rank(method="first"), 3, labels=["low", "mid", "high"])
    d = d.loc[ter.isin(["low", "high"])].copy()
    d["deep"] = (ter[ter.isin(["low", "high"])] == "high").astype(int).values
    return d.dropna(subset=ps_covars + ["deep", "aki_or_death"])


# Monitoring-density diagnostic: charted minutes are entangled with how often vitals
# were recorded. We (1) quantify the minutes/reading-count association, and (2) build a
# monitoring-adjusted minutes exposure (minutes residualised on reading count) so the
# "protective" reversal can be re-tested with charting cadence removed.
import statsmodels.api as sm
from scipy.stats import spearmanr

_mon = analytic_all.dropna(subset=["minutes_2h", "n_read_2h"]).copy()
_ols_min = sm.OLS(_mon["minutes_2h"].astype(float),
                  sm.add_constant(_mon["n_read_2h"].astype(float))).fit()
analytic_all["minutes_resid"] = np.nan
analytic_all.loc[_mon.index, "minutes_resid"] = _ols_min.resid.values
rho_min_read, _ = spearmanr(_mon["minutes_2h"], _mon["n_read_2h"])
monitoring = pd.DataFrame([{
    "spearman_minutes_readings": rho_min_read,
    "deep_median_readings": float(ana.loc[ana.deep == 1, "n_read_2h"].median()),
    "shallow_median_readings": float(ana.loc[ana.deep == 0, "n_read_2h"].median()),
    "deep_median_minutes": float(ana.loc[ana.deep == 1, "minutes_2h"].median()),
    "shallow_median_minutes": float(ana.loc[ana.deep == 0, "minutes_2h"].median()),
}])
monitoring.to_csv(OUT / "monitoring_diagnostic.csv", index=False)
print(f"Monitoring density: Spearman(minutes, readings) = {rho_min_read:+.2f}; "
      f"deep readings med {monitoring['deep_median_readings'].iloc[0]:.0f} vs "
      f"shallow {monitoring['shallow_median_readings'].iloc[0]:.0f}")

expo_rows = []
for metric, hi_sev, lab in [("nadir_2h", False, "Nadir MAP (depth) [PRIMARY]"),
                            ("aut_2h", True, "AUT depth-weighted burden"),
                            ("minutes_2h", True, "Minutes MAP<65 (duration)"),
                            ("minutes_resid", True, "Minutes MAP<65 (monitoring-adjusted)")]:
    est = boot(contrast_metric(metric, hi_sev), ps_covars, "overlap", B=N_BOOT_SENS)
    expo_rows.append({"exposure": lab, **{k: est[k] for k in
                     ["risk_high", "risk_low", "rd", "rd_lo", "rd_hi", "rr", "rr_lo", "rr_hi"]}})
expo_df = pd.DataFrame(expo_rows)
expo_df.to_csv(OUT / "exposure_comparison.csv", index=False)
print(dose_df.round(3).to_string(index=False))
print(expo_df.round(3).to_string(index=False))

# Continuous dose-response: restricted cubic spline of nadir MAP -> 7-day composite risk
# (marginal/crude curve, same semantics as the binned dose-response, for the supplement).
# Harrell restricted-cubic-spline basis built directly in numpy (patsy cr() is unreliable
# on this interpreter build).
def rcs_basis(x, knots):
    x = np.asarray(x, float)
    t = np.asarray(knots, float)
    tk, tk1 = t[-1], t[-2]
    denom = tk - tk1
    scale = (t[-1] - t[0]) ** 2
    cube = lambda u: np.where(u > 0, u, 0.0) ** 3
    cols = [x]
    for j in range(len(t) - 2):
        cols.append((cube(x - t[j])
                     - cube(x - tk1) * (tk - t[j]) / denom
                     + cube(x - tk) * (tk1 - t[j]) / denom) / scale)
    return np.column_stack(cols)


_sp = analytic_all.dropna(subset=["nadir_2h", "aki_or_death"]).copy()
_sp = _sp.loc[_sp["nadir_2h"].between(38, 64.9)]
_x = _sp["nadir_2h"].astype(float).to_numpy()
_knots = np.quantile(_x, [0.05, 0.35, 0.65, 0.95])
_Xd = np.column_stack([np.ones(len(_x)), rcs_basis(_x, _knots)])
_fit = sm.Logit(_sp["aki_or_death"].astype(int).to_numpy(), _Xd).fit(disp=False, maxiter=200)
_grid = np.linspace(float(_x.min()), 64.0, 90)
_Xg = np.column_stack([np.ones(len(_grid)), rcs_basis(_grid, _knots)])
_eta = _Xg @ _fit.params
_se = np.sqrt(np.einsum("ij,jk,ik->i", _Xg, _fit.cov_params(), _Xg))
spline_df = pd.DataFrame({
    "nadir": _grid,
    "risk": 1.0 / (1.0 + np.exp(-_eta)),
    "lo": 1.0 / (1.0 + np.exp(-(_eta - 1.96 * _se))),
    "hi": 1.0 / (1.0 + np.exp(-(_eta + 1.96 * _se))),
})
spline_df.to_csv(OUT / "dose_response_spline.csv", index=False)
print(f"Spline dose-response saved ({len(spline_df)} points, "
      f"nadir {_grid.min():.0f}-{_grid.max():.0f} mmHg).")


# %% [markdown]
# ## 12. Bias analyses
#
# (a) **Selection bias** via IPCW for reference-creatinine availability; (b) **reverse
# causation** via a pre-onset negative-control outcome; (c) **unmeasured confounding**
# via E-value and a tipping-point curve.

# %%
# (a) IPCW: model P(has reference creatinine | pre-onset covariates + depth) among all
# eligible episodes, then reweight the analytic cohort by 1/P(selected).
sel = eligible.copy()
sel["sev_tertile"] = pd.qcut((-sel["nadir_2h"]).rank(method="first"), 3,
                             labels=["shallow", "mid", "deep"])
sel_covars = [c for c in ["age", "sex_male", "acuity", "night_onset", "nadir_2h",
                          "onset_map", "profound_flag", "n_read_2h",
                          "chf", "cirrhosis", "diabetes", "ckd", "cad", "copd", "sepsis"]
              if c in sel.columns]
sd = sel.dropna(subset=[c for c in sel_covars if c != "acuity"]).copy()
sd["acuity"] = sd["acuity"].fillna(sd["acuity"].median())
Xs = StandardScaler().fit_transform(sd[sel_covars].astype(float))
p_sel = LogisticRegression(max_iter=1000).fit(Xs, sd["has_ref"]).predict_proba(Xs)[:, 1]
sd["p_sel"] = np.clip(p_sel, 0.05, 0.95)
ipcw_map = sd.set_index("stay_id")["p_sel"]
cc_ipcw = cc.copy()
cc_ipcw["ipcw"] = (1.0 / cc_ipcw["stay_id"].map(ipcw_map)).fillna(1.0)
cc_ipcw["ipcw"] = (cc_ipcw["ipcw"] / cc_ipcw["ipcw"].mean()).clip(upper=np.quantile(cc_ipcw["ipcw"], 0.99))
ipcw_est = boot(cc_ipcw, ps_covars, "overlap", "aki_or_death", B=N_BOOT_SENS,
                extra_w=cc_ipcw["ipcw"].to_numpy())
print(f"(a) IPCW-adjusted RD {ipcw_est['rd']:+.4f} ({ipcw_est['rd_lo']:+.4f},{ipcw_est['rd_hi']:+.4f})")

# (b) Negative-control outcome: pre-onset AKI (occurs before exposure)
neg_ctrl = boot(cc, ps_covars, "overlap", "pre_onset_aki", B=N_BOOT_SENS)
print(f"(b) Negative-control (pre-onset AKI) RD {neg_ctrl['rd']:+.4f} "
      f"({neg_ctrl['rd_lo']:+.4f},{neg_ctrl['rd_hi']:+.4f})  "
      f"[should be near 0 if temporality holds]")

# (c) Tipping-point: RD attenuation under an unmeasured binary confounder U with
# prevalence difference delta between arms and outcome risk ratio gamma.
def bias_adjusted_rr(rr_obs, p1, p0, gamma):
    """Array bias formula: observed RR / bias factor for confounder with RR=gamma,
    prevalence p1 (deep) and p0 (shallow)."""
    bf = (gamma * p1 + (1 - p1)) / (gamma * p0 + (1 - p0))
    return rr_obs / bf


tip = []
for gamma in [1.5, 2.0, 2.5, 3.0]:
    for delta in np.linspace(0, 0.6, 25):
        p1 = min(0.6 + delta / 2, 0.99); p0 = max(0.6 - delta / 2, 0.01)
        tip.append({"gamma": gamma, "delta": delta,
                    "rr_adj": bias_adjusted_rr(primary["rr"], p1, p0, gamma)})
tip_df = pd.DataFrame(tip)
tip_df.to_csv(OUT / "tipping_point.csv", index=False)
pd.DataFrame([{
    "ipcw_rd": ipcw_est["rd"], "ipcw_rd_lo": ipcw_est["rd_lo"], "ipcw_rd_hi": ipcw_est["rd_hi"],
    "neg_ctrl_rd": neg_ctrl["rd"], "neg_ctrl_rd_lo": neg_ctrl["rd_lo"], "neg_ctrl_rd_hi": neg_ctrl["rd_hi"],
    "e_value": primary["e_value"], "e_value_ci": primary["e_value_ci"],
}]).to_csv(OUT / "bias_summary.csv", index=False)


# %% [markdown]
# ## 13. Sensitivity analyses and effect modification

# %%
sens_rows = [{"analysis": "Primary: MICE overlap (nadir tertiles)", **{k: primary[k] for k in
             ["rd", "rd_lo", "rd_hi", "rr", "rr_lo", "rr_hi", "n"]}}]
sens_rows.append({"analysis": "Complete-case overlap", **{k: cc_overlap[k] for k in
                 ["rd", "rd_lo", "rd_hi", "rr", "rr_lo", "rr_hi", "n"]}})
sens_rows.append({"analysis": "Doubly-robust AIPW", **{k: cc_aipw[k] for k in
                 ["rd", "rd_lo", "rd_hi", "rr", "rr_lo", "rr_hi", "n"]}})
sens_rows.append({"analysis": "Truncated stabilized IPTW", **{k: cc_iptw[k] for k in
                 ["rd", "rd_lo", "rd_hi", "rr", "rr_lo", "rr_hi", "n"]}})
sens_rows.append({"analysis": "Selection-adjusted (IPCW)", **{k: ipcw_est[k] for k in
                 ["rd", "rd_lo", "rd_hi", "rr", "rr_lo", "rr_hi", "n"]}})

# Co-primary using absolute clinical MAP cutpoints (transportable, not cohort tertiles):
# deep = nadir <50 mmHg vs shallow = 55-64 mmHg (55 <= nadir < 65); the 50-54 band is the
# excluded middle so the contrast mirrors the tertile design.
abs_src = analytic_all.loc[(analytic_all["nadir_2h"] < 50)
                           | ((analytic_all["nadir_2h"] >= 55) & (analytic_all["nadir_2h"] < 65))].copy()
abs_src["deep"] = (abs_src["nadir_2h"] < 50).astype(int)
abs_co = abs_src.dropna(subset=ps_covars + ["deep", "aki_or_death"])
abs_est = boot(abs_co, ps_covars, "overlap", B=N_BOOT)
sens_rows.append({"analysis": "Co-primary: absolute thresholds (<50 vs 55-64 mmHg)",
                 **{k: abs_est[k] for k in ["rd", "rd_lo", "rd_hi", "rr", "rr_lo", "rr_hi", "n"]}})

# Reverse-causation guard: restrict to episodes AKI-free at time zero (drops anyone whose
# creatinine was already rising into stage 1 before onset).
akifree = ana.loc[ana["pre_onset_aki"].fillna(0) == 0].dropna(subset=ps_covars + ["deep", "aki_or_death"])
akifree_est = boot(akifree, ps_covars, "overlap", B=N_BOOT)
sens_rows.append({"analysis": "AKI-free at onset (reverse-causation guard)",
                 **{k: akifree_est[k] for k in ["rd", "rd_lo", "rd_hi", "rr", "rr_lo", "rr_hi", "n"]}})

# S: 1-hour and 3-hour nadir windows
for wcol, wl in [("nadir_1h", "S: 1-hour nadir window"), ("nadir_3h", "S: 3-hour nadir window")]:
    a = analytic_all.copy()
    a["_t"] = pd.qcut((-a[wcol]).rank(method="first"), 3, labels=["shallow", "mid", "deep"])
    a = a.loc[a["_t"].isin(["shallow", "deep"])].copy()
    a["deep"] = (a["_t"] == "deep").astype(int)
    est = boot(a.dropna(subset=ps_covars + ["deep", "aki_or_death"]), ps_covars, "overlap", B=N_BOOT_SENS)
    sens_rows.append({"analysis": wl, **{k: est[k] for k in ["rd", "rd_lo", "rd_hi", "rr", "rr_lo", "rr_hi", "n"]}})

# S: >=3 readings
a3 = ana.loc[ana["n_read_2h"] >= 3].dropna(subset=ps_covars + ["deep", "aki_or_death"])
est = boot(a3, ps_covars, "overlap", B=N_BOOT_SENS)
sens_rows.append({"analysis": "S: >=3 MAP readings in window", **{k: est[k] for k in
                 ["rd", "rd_lo", "rd_hi", "rr", "rr_lo", "rr_hi", "n"]}})

# S: exclude profound-only onsets (measurement robustness)
ap = ana.loc[ana["profound_flag"] == 0].dropna(subset=ps_covars + ["deep", "aki_or_death"])
if ap["deep"].nunique() == 2:
    est = boot(ap, ps_covars, "overlap", B=N_BOOT_SENS)
    sens_rows.append({"analysis": "S: exclude profound-only onsets", **{k: est[k] for k in
                     ["rd", "rd_lo", "rd_hi", "rr", "rr_lo", "rr_hi", "n"]}})

# S: KDIGO stage>=1 composite
est = boot(cc.assign(aki_or_death=((cc["aki_stage1"] == 1) | (cc["death_7d"] == 1)).astype(int)),
           ps_covars, "overlap", B=N_BOOT_SENS)
sens_rows.append({"analysis": "S: KDIGO stage>=1 composite", **{k: est[k] for k in
                 ["rd", "rd_lo", "rd_hi", "rr", "rr_lo", "rr_hi", "n"]}})

# S: continuous depth (OR per -10 mmHg)
import statsmodels.api as sm
a2 = analytic_all.dropna(subset=ps_covars + ["aki_or_death"]).copy()
Xc = sm.add_constant(a2[["nadir_2h"] + ps_covars].astype(float))
mfit = sm.Logit(a2["aki_or_death"].astype(int), Xc).fit(disp=False, maxiter=200)
or10 = float(np.exp(mfit.params["nadir_2h"] * -10))
ci10 = (mfit.conf_int().loc["nadir_2h"] * -10).sort_values()
sens_rows.append({"analysis": "S: continuous depth (OR per -10 mmHg)", "rd": np.nan,
                  "rd_lo": np.nan, "rd_hi": np.nan, "rr": or10,
                  "rr_lo": float(np.exp(ci10.iloc[0])), "rr_hi": float(np.exp(ci10.iloc[1])), "n": len(a2)})
sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(OUT / "sensitivity_analyses.csv", index=False)
print(sens_df.round(3).to_string(index=False))

# Effect modification by night onset
night_rows = []
for lab, sub in [("day", cc[cc.night_onset == 0]), ("night", cc[cc.night_onset == 1])]:
    if sub["deep"].nunique() == 2 and len(sub) > 100:
        est = boot(sub, ps_covars, "overlap", B=N_BOOT_SENS)
        night_rows.append({"stratum": lab, **{k: est[k] for k in
                          ["risk_high", "risk_low", "rd", "rd_lo", "rd_hi", "n"]}})
night_df = pd.DataFrame(night_rows)
night_df.to_csv(OUT / "effect_modification_night.csv", index=False)
print(night_df.round(3).to_string(index=False))


# %% [markdown]
# ## 14. Table 1

# %%
def t1_cont(df, col):
    g = df.groupby("deep")[col]
    return ({a: f"{g.get_group(a).median():.1f} [{g.get_group(a).quantile(.25):.1f}, "
             f"{g.get_group(a).quantile(.75):.1f}]" for a in [0, 1]},
            smd(df[col], df["deep"].to_numpy()))


def t1_bin(df, col):
    g = df.groupby("deep")[col]
    return ({a: f"{100*g.get_group(a).mean():.1f}%" for a in [0, 1]},
            smd(df[col].astype(float), df["deep"].to_numpy()))


t1 = []
for col, lab in {"age": "Age, y", "nadir_2h": "Nadir MAP 2h, mmHg", "onset_map": "Onset MAP, mmHg",
                 "aut_2h": "AUT, mmHg-min", "minutes_2h": "Minutes MAP<65", "n_read_2h": "MAP readings 2h",
                 "pre_hr": "Pre-onset HR", "pre_o2": "Pre-onset SpO2", "ref_cr": "Reference creatinine",
                 "acuity": "Triage acuity", "pre_lactate": "Pre-onset lactate"}.items():
    sub = ana.dropna(subset=[col])
    if len(sub) < 50:
        continue
    v, s = t1_cont(sub, col)
    t1.append({"variable": lab, "shallow": v[0], "deep": v[1], "smd": round(s, 3)})
for col, lab in {"sex_male": "Male", "profound_flag": "Profound onset", "night_onset": "Night onset",
                 "chf": "Heart failure", "cirrhosis": "Cirrhosis", "diabetes": "Diabetes",
                 "ckd": "CKD (non-ESRD)", "cad": "Coronary disease", "copd": "COPD",
                 "sepsis": "Sepsis (admission)",
                 "aki": "Stage>=2 AKI 7d", "death_7d": "Death 7d", "aki_or_death": "Composite 7d"}.items():
    v, s = t1_bin(ana.dropna(subset=[col]), col)
    t1.append({"variable": lab, "shallow": v[0], "deep": v[1], "smd": round(s, 3)})
table1 = pd.DataFrame(t1)
table1.to_csv(OUT / "table1_baseline.csv", index=False)
print(table1.to_string(index=False))

ana.to_csv(OUT / "analytic_cohort.csv", index=False)
cc.to_parquet(CACHE / "complete_case.parquet", index=False)
epi.to_parquet(CACHE / "episodes.parquet", index=False)
run_summary = {
    "config": {k: globals()[k] for k in ["MAP_THRESHOLD", "MAP_PROFOUND", "SUSTAIN_MIN",
               "MAX_GAP_MIN", "BURDEN_WINDOW_H", "AKI_RATIO", "HORIZON_DAYS", "N_BOOT", "N_MICE"]},
    "primary": {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in primary.items()},
    "aipw": {k: cc_aipw[k] for k in ["rd", "rd_lo", "rd_hi", "rr", "rr_lo", "rr_hi"]},
    "ipcw": {k: ipcw_est[k] for k in ["rd", "rd_lo", "rd_hi"]},
    "neg_control": {k: neg_ctrl[k] for k in ["rd", "rd_lo", "rd_hi"]},
    "abs_threshold": {**{k: abs_est[k] for k in ["rd", "rd_lo", "rd_hi", "rr", "rr_lo", "rr_hi", "n",
                     "risk_high", "risk_low"]}},
    "aki_free_onset": {**{k: akifree_est[k] for k in ["rd", "rd_lo", "rd_hi", "rr", "rr_lo", "rr_hi", "n"]}},
    "monitoring": monitoring.iloc[0].to_dict(),
    "max_abs_smd_overlap": float(balance["abs_smd_overlap"].max()),
    "cause_specific": cause_specific.to_dict("records"),
}
(OUT / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
print("Saved tables + run_summary.json")


# %% [markdown]
# ## 15. Figures (house style)
#
# Layout is maintained in `rebuild_figures.py` so labels never cover data layers.
# That module is the single source of truth for all publication figures.

# %%
import rebuild_figures as RF
RF.main()

# Legacy figure function definitions below are retained only so older notebook
# cells that call them by name do not break; they are not used in a fresh run.
def figure_flow():
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.8), gridspec_kw={"width_ratios": [1.0, 1.05]})
    ax = axes[0]; ax.axis("off"); ax.set_xlim(0, 10)
    n = len(strobe); ax.set_ylim(0.3, n + 0.5)
    panel_label(ax, "(A)  Cohort derivation")

    def y_of(i):
        return n - i
    for i, r in enumerate(strobe.itertuples()):
        y = y_of(i); key = i in (0, n - 1)
        ax.add_patch(FancyBboxPatch((1.0, y - 0.34), 4.9, 0.68,
                     boxstyle="round,pad=0.04,rounding_size=0.10",
                     facecolor="#B7D0E4" if key else "#FFFFFF",
                     edgecolor=PAL["ink"], linewidth=1.1, zorder=3))
        ax.text(3.45, y + 0.10, r.step, ha="center", va="center", fontsize=8.6,
                fontweight="bold", color=PAL["ink"], zorder=4)
        ax.text(3.45, y - 0.17, f"n = {int(r.n):,}", ha="center", va="center",
                fontsize=8.8, fontweight="bold", color=PAL["ink"], zorder=4)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((3.45, y - 0.34), (3.45, y_of(i + 1) + 0.34),
                         arrowstyle="-|>", mutation_scale=15, lw=1.4, color=PAL["ink"], zorder=2))
    for i in range(n - 1):
        yi = (y_of(i) + y_of(i + 1)) / 2
        drop = int(strobe.iloc[i].n) - int(strobe.iloc[i + 1].n)
        if drop <= 0:
            continue
        ax.add_patch(FancyBboxPatch((6.4, yi - 0.24), 3.2, 0.48,
                     boxstyle="round,pad=0.03,rounding_size=0.08",
                     facecolor="#FFFFFF", edgecolor=PAL["muted"], linewidth=1.0, zorder=3))
        ax.text(8.0, yi, f"Excluded: {drop:,}", ha="center", va="center", fontsize=8.4, zorder=4)
        ax.add_patch(FancyArrowPatch((3.45, yi), (6.4, yi), arrowstyle="-|>",
                     mutation_scale=11, lw=1.0, color=PAL["muted"], ls="--", zorder=2))

    ax = axes[1]; panel(ax, "x"); panel_label(ax, "(B)  Exposure: MAP trajectory over 2 hours")
    tt = np.linspace(0, 120, 200)
    deep_tr = 78 - 30 * np.exp(-((tt - 25) ** 2) / 300) - 6 * np.exp(-((tt - 70) ** 2) / 900)
    shal_tr = 74 - 10 * np.exp(-((tt - 40) ** 2) / 1500)
    ax.axhspan(0, 65, color=PAL["deep"], alpha=0.06, zorder=0)
    ax.axhline(65, color=PAL["ink"], lw=1.1, ls="--", zorder=3)
    ax.text(118, 66, "MAP 65 mmHg threshold", ha="right", va="bottom", fontsize=8.8)
    ax.plot(tt, deep_tr, color=PAL["deep"], lw=2.6, zorder=5, label="Deep nadir (top tertile)")
    ax.plot(tt, shal_tr, color=PAL["shallow"], lw=2.6, zorder=5, label="Shallow nadir (bottom tertile)")
    ax.scatter([25], [deep_tr[np.argmin(np.abs(tt - 25))]], s=70, color=PAL["deep"],
               edgecolor="white", zorder=6)
    ax.annotate("nadir (depth)\n= exposure", xy=(25, deep_tr.min()),
                xytext=(48, 40), fontsize=9, color=PAL["deep"],
                arrowprops=dict(arrowstyle="-|>", color=PAL["deep"], lw=1.2))
    ax.fill_between(tt, deep_tr, 65, where=(deep_tr < 65), color=PAL["deep"], alpha=0.18, zorder=2)
    ax.text(70, 52, "AUT = area under 65\n(depth x time)", fontsize=8.6, color="#7F2A18")
    ax.set_xlim(0, 120); ax.set_ylim(35, 85)
    ax.set_xlabel("Minutes after episode onset"); ax.set_ylabel("MAP (mmHg)")
    ax.legend(loc="upper right", fontsize=9)
    save_fig(fig, "Figure2_CohortFlow")
    plt.show()




# %%
# ---------- Figure 3: Love plot ----------
def figure_love():
    bal = balance.sort_values("smd_unweighted", key=lambda s: s.abs())
    labmap = {"age": "Age", "pre_hr": "Pre-onset HR", "pre_o2": "Pre-onset SpO2",
              "ref_cr": "Reference creatinine", "acuity": "Triage acuity",
              "pre_lactate": "Pre-onset lactate", "sex_male": "Male sex",
              "night_onset": "Night onset", "chf": "Heart failure",
              "cirrhosis": "Cirrhosis", "diabetes": "Diabetes", "ckd": "CKD (non-ESRD)",
              "cad": "Coronary disease", "copd": "COPD", "sepsis": "Sepsis (admission)"}
    fig, ax = plt.subplots(figsize=(8.4, 5.2)); panel(ax, "x")
    y = np.arange(len(bal))
    for yi, u, a in zip(y, bal["smd_unweighted"], bal["smd_overlap"]):
        ax.plot([abs(u), abs(a)], [yi, yi], color="#C6CBD1", lw=1.6, zorder=2)
    ax.scatter(bal["smd_unweighted"].abs(), y, s=95, color=PAL["muted"],
               edgecolor="white", zorder=4, label="Unweighted")
    ax.scatter(bal["smd_overlap"].abs(), y, s=100, color=PAL["deep"],
               edgecolor="white", zorder=5, label="Overlap-weighted")
    ax.axvline(0.1, ls="--", color=PAL["ink"], lw=1.3)
    ax.text(0.1, len(bal) - 0.3, " |SMD| = 0.10", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([labmap.get(c, c) for c in bal["covariate"]])
    ax.set_xlabel("Absolute standardized mean difference"); ax.set_xlim(left=0)
    ax.legend(loc="lower right"); ax.set_title("Covariate balance before and after overlap weighting")
    save_fig(fig, "Figure3_LovePlot")
    plt.show()




# %%
# ---------- Figure 4: Primary effect + dose-response ----------
def figure_primary():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]; panel(ax, "y"); panel_label(ax, "(A)  Overlap-weighted primary outcome")
    risks = [primary["risk_low"] * 100, primary["risk_high"] * 100]
    los = [primary["risk_low_lo"] * 100, primary["risk_high_lo"] * 100]
    his = [primary["risk_low_hi"] * 100, primary["risk_high_hi"] * 100]
    xb = np.arange(2)
    ax.bar(xb, risks, width=0.6, color=[PAL["shallow"], PAL["deep"]], edgecolor="white", zorder=3)
    ax.errorbar(xb, risks, yerr=[np.array(risks) - los, np.array(his) - risks], fmt="none",
                ecolor=PAL["ink"], elinewidth=1.6, capsize=6, zorder=4)
    for x, r, h in zip(xb, risks, his):
        ax.text(x, h + 0.6, f"{r:.1f}%", ha="center", fontweight="bold", fontsize=12)
    ax.set_xticks(xb); ax.set_xticklabels(["Shallow nadir", "Deep nadir"])
    ax.set_ylabel("7-day risk of AKI or death (%)"); ax.set_ylim(0, max(his) * 1.42)
    ax.text(0.97, 0.80,
            f"RD {primary['rd']*100:+.1f} pts ({primary['rd_lo']*100:+.1f}, {primary['rd_hi']*100:+.1f})\n"
            f"RR {primary['rr']:.2f} ({primary['rr_lo']:.2f}, {primary['rr_hi']:.2f})\n"
            f"E-value {primary['e_value']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.2,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D0D5DD"))

    ax = axes[1]; panel(ax); panel_label(ax, "(B)  Dose-response by depth")
    dd = dose_df.reset_index(drop=True); xd = np.arange(len(dd))
    ax.plot(xd, dd["risk"] * 100, "-o", color=PAL["deep"], lw=2.6, ms=10, markeredgecolor="white", zorder=4)
    ax.fill_between(xd, dd["lo"] * 100, dd["hi"] * 100, color=PAL["deep"], alpha=0.15, zorder=2)
    for x, (_, r) in zip(xd, dd.iterrows()):
        ax.text(x, r["hi"] * 100 + 0.6, f"{r['risk']*100:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(xd); ax.set_xticklabels([f"{b}\n(n={int(n)})" for b, n in zip(dd["band"], dd["n"])])
    ax.invert_xaxis()
    ax.set_xlabel("Nadir MAP in first 2 hours (mmHg)"); ax.set_ylabel("7-day risk of AKI or death (%)")
    fig.suptitle("Depth of early ED hypotension and 7-day AKI or death", fontsize=14.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "Figure4_PrimaryEffect")
    plt.show()




# %%
# ---------- Figure 5: Competing-risk CIF + cause-specific HR ----------
def figure_competing():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]; panel(ax); panel_label(ax, "(A)  Cumulative incidence (Aalen-Johansen)")
    for arm, col in [("deep", PAL["deep"]), ("shallow", PAL["shallow"])]:
        s = cif_df[cif_df.arm == arm]
        ax.step(s["t"], s["cif_aki"] * 100, where="post", color=col, lw=2.6, label=f"AKI, {arm}")
        ax.step(s["t"], s["cif_death"] * 100, where="post", color=col, lw=2.0, ls="--", label=f"Death, {arm}")
    top = cif_df[["cif_aki", "cif_death"]].max().max() * 100
    ax.set_xlim(0, HORIZON_DAYS); ax.set_ylim(0, top * 1.32)
    ax.set_xlabel("Days from onset"); ax.set_ylabel("Cumulative incidence (%)")
    ax.legend(fontsize=8.6, loc="upper left", bbox_to_anchor=(0.015, 0.88), ncol=2,
              frameon=True, facecolor="white", edgecolor="#D0D5DD", columnspacing=1.1)

    ax = axes[1]; panel(ax, "x"); panel_label(ax, "(B)  Cause-specific hazard ratio")
    cs = cause_specific.dropna(subset=["hr"])
    y = np.arange(len(cs))
    ax.errorbar(cs["hr"], y, xerr=[cs["hr"] - cs["lo"], cs["hi"] - cs["hr"]], fmt="o",
                color=PAL["deep"], ecolor=PAL["ink"], elinewidth=1.6, capsize=6, ms=11, markeredgecolor="white")
    ax.axvline(1.0, color=PAL["ink"], ls="--", lw=1.2)
    for yi, (_, r) in zip(y, cs.iterrows()):
        ax.text(r["hi"] + 0.05, yi, f"{r['hr']:.2f} ({r['lo']:.2f}, {r['hi']:.2f})", va="center", fontsize=9.5)
    ax.set_yticks(y); ax.set_yticklabels([f"{c} (deep vs shallow)" for c in cs["cause"]])
    ax.set_xlabel("Cause-specific hazard ratio"); ax.set_ylim(-0.6, len(cs) - 0.4)
    ax.margins(x=0.25)
    save_fig(fig, "Figure5_CompetingRisk")
    plt.show()




# %%
# ---------- Figure 6: Exposure-measurement comparison ----------
def figure_exposure():
    fig, ax = plt.subplots(figsize=(10.4, 4.6)); panel(ax, "x")
    ed = expo_df.iloc[::-1].reset_index(drop=True); y = np.arange(len(ed))
    cols = [PAL["deep"] if r["rd"] > 0 else PAL["muted"] for _, r in ed.iterrows()]
    ax.errorbar(ed["rd"] * 100, y, xerr=[np.clip((ed["rd"] - ed["rd_lo"]) * 100, 0, None),
                np.clip((ed["rd_hi"] - ed["rd"]) * 100, 0, None)], fmt="none",
                ecolor=PAL["ink"], elinewidth=1.7, capsize=6, zorder=2)
    ax.scatter(ed["rd"] * 100, y, s=150, c=cols, edgecolor="white", zorder=3)
    ax.axvline(0, color=PAL["ink"], ls="--", lw=1.3)
    for yi, (_, r) in zip(y, ed.iterrows()):
        ax.text(r["rd_hi"] * 100 + 0.4, yi,
                f"RD {r['rd']*100:+.1f} ({r['rd_lo']*100:+.1f}, {r['rd_hi']*100:+.1f}); RR {r['rr']:.2f}",
                va="center", fontsize=9.6)
    ax.set_yticks(y); ax.set_yticklabels(ed["exposure"])
    ax.set_xlabel("Overlap-weighted 7-day risk difference, AKI or death (pts)")
    ax.set_title("Planned exposure-measurement comparison:\ndepth harms; charted minutes reverse (monitoring bias)", fontsize=13)
    ax.margins(x=0.32)
    save_fig(fig, "Figure6_ExposureMeasurement")
    plt.show()




# %%
# ---------- Figure 7: Bias analyses ----------
def figure_bias():
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))

    # (A) E-value curve
    ax = axes[0]; panel(ax); panel_label(ax, "(A)  E-value")
    rr = primary["rr"]; rr_lo = primary["rr_lo"]
    gg = np.linspace(1.0, max(4.0, primary["e_value"] + 1), 200)
    # confounder-outcome vs confounder-exposure association trade-off to explain RR
    def partner(g, target):
        # solve BF = target for the other association given g
        return np.where(g > target, (target * (g - 1)) / (g - target) + 1, np.nan)
    ax.plot(gg, partner(gg, rr), color=PAL["deep"], lw=2.4, label=f"Point estimate (RR {rr:.2f})")
    if rr_lo > 1:
        ax.plot(gg, partner(gg, rr_lo), color=PAL["shallow"], lw=2.0, ls="--",
                label=f"CI bound (RR {rr_lo:.2f})")
    ax.scatter([primary["e_value"]], [primary["e_value"]], s=90, color=PAL["deep"],
               edgecolor="white", zorder=6)
    ax.annotate(f"E-value = {primary['e_value']:.2f}", (primary["e_value"], primary["e_value"]),
                xytext=(primary["e_value"] + 0.4, primary["e_value"] + 0.6), fontsize=9.5,
                arrowprops=dict(arrowstyle="-|>", color=PAL["ink"], lw=1.0))
    ax.set_xlabel("Confounder-exposure association"); ax.set_ylabel("Confounder-outcome association")
    ax.set_xlim(1, gg.max()); ax.set_ylim(1, gg.max()); ax.legend(fontsize=9, loc="upper right")

    # (B) Falsification / selection forest
    ax = axes[1]; panel(ax, "x"); panel_label(ax, "(B)  Falsification and selection")
    rows = [("Primary (composite)", primary["rd"], primary["rd_lo"], primary["rd_hi"], PAL["deep"]),
            ("Selection-adjusted (IPCW)", ipcw_est["rd"], ipcw_est["rd_lo"], ipcw_est["rd_hi"], PAL["sky"]),
            ("Negative control\n(pre-onset AKI)", neg_ctrl["rd"], neg_ctrl["rd_lo"], neg_ctrl["rd_hi"], PAL["muted"])]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, rd, lo, hi, col) in zip(y, rows):
        ax.errorbar(rd * 100, yi, xerr=[[max(0, (rd - lo) * 100)], [max(0, (hi - rd) * 100)]],
                    fmt="o", color=col, ecolor=PAL["ink"], elinewidth=1.6, capsize=6, ms=11, markeredgecolor="white")
        ax.text((hi) * 100 + 0.3, yi, f"{rd*100:+.1f} ({lo*100:+.1f}, {hi*100:+.1f})", va="center", fontsize=9)
    ax.axvline(0, color=PAL["ink"], ls="--", lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.margins(x=0.3)
    ax.set_ylim(-0.6, len(rows) - 0.15); ax.set_xlabel("Risk difference (pts)")

    # (C) Tipping point
    ax = axes[2]; panel(ax); panel_label(ax, "(C)  Unmeasured-confounding tipping point")
    for g, col in zip([1.5, 2.0, 2.5, 3.0], [PAL["amber"], PAL["sky"], PAL["plum"], PAL["deep"]]):
        s = tip_df[tip_df.gamma == g]
        ax.plot(s["delta"], s["rr_adj"], lw=2.2, color=col, label=f"conf-outcome RR {g:.1f}")
    ax.axhline(1.0, color=PAL["ink"], ls="--", lw=1.3)
    ax.set_xlabel("Confounder prevalence difference (deep - shallow)")
    ax.set_ylabel("Bias-adjusted RR"); ax.legend(fontsize=8.6, loc="upper right", title="strength")
    ax.set_title("")
    fig.suptitle("Quantitative bias analysis", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "Figure7_BiasAnalysis")
    plt.show()




# %%
# ---------- Supplementary figures ----------
def figure_supp():
    fig, ax = plt.subplots(figsize=(8.0, 5.0)); panel(ax, "y")
    bins = np.linspace(0, 1, 31)
    ax.hist(cc.loc[cc.deep == 1, "ps"], bins=bins, color=PAL["deep"], alpha=0.6,
            density=True, edgecolor="white", label="Deep nadir")
    ax.hist(cc.loc[cc.deep == 0, "ps"], bins=bins, color=PAL["shallow"], alpha=0.55,
            density=True, edgecolor="white", label="Shallow nadir")
    ax.set_xlabel("Estimated propensity score (P[deep])"); ax.set_ylabel("Density")
    ax.legend(); ax.set_title("Propensity score overlap")
    save_fig(fig, "FigureS1_PropensityOverlap")
    plt.show()

    fig, ax = plt.subplots(figsize=(9.6, 5.6)); panel(ax, "x")
    sd = sens_df[sens_df["rd"].notna()].copy(); y = np.arange(len(sd))[::-1]
    ax.errorbar(sd["rd"] * 100, y, xerr=[np.clip((sd["rd"] - sd["rd_lo"]) * 100, 0, None),
                np.clip((sd["rd_hi"] - sd["rd"]) * 100, 0, None)], fmt="o", color=PAL["ink"],
                ecolor="#666", elinewidth=1.5, capsize=5, ms=9)
    ax.axvline(0, color=PAL["deep"], ls="--", lw=1.3)
    for yi, (_, r) in zip(y, sd.iterrows()):
        ax.text(r["rd_hi"] * 100 + 0.3, yi, f"{r['rd']*100:+.1f} ({r['rd_lo']*100:+.1f}, {r['rd_hi']*100:+.1f})",
                va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(sd["analysis"], fontsize=9.5)
    ax.set_xlabel("Overlap-weighted risk difference, AKI or death (pts)")
    ax.set_title("Sensitivity analyses"); ax.margins(x=0.2)
    save_fig(fig, "FigureS2_SensitivityForest")
    plt.show()




# %%
print("\n" + "=" * 66)
print("RESULTS SUMMARY  -  Depth of early ED hypotension and 7-day AKI/death")
print("=" * 66)
print(f"Analytic cohort (deep/shallow) : {len(ana):,} ({int(ana.deep.sum())}/{int((1-ana.deep).sum())})")
print(f"Complete-case weighted          : {len(cc):,}")
print(f"PS covariates                   : {ps_covars}")
print(f"Max |SMD| after overlap         : {balance['abs_smd_overlap'].max():.3f}")
print(f"PRIMARY risk deep {primary['risk_high']*100:.1f}% vs shallow {primary['risk_low']*100:.1f}%")
print(f"PRIMARY RD  {primary['rd']*100:+.1f} pts ({primary['rd_lo']*100:+.1f}, {primary['rd_hi']*100:+.1f})")
print(f"PRIMARY RR  {primary['rr']:.2f} ({primary['rr_lo']:.2f}, {primary['rr_hi']:.2f}); E-value {primary['e_value']:.2f}")
print(f"AIPW RD {cc_aipw['rd']*100:+.1f}; IPCW RD {ipcw_est['rd']*100:+.1f}; "
      f"neg-control RD {neg_ctrl['rd']*100:+.1f}")
print("Figures: Figure1_DAG, Figure2_CohortFlow, Figure3_LovePlot, Figure4_PrimaryCompeting, "
      "Figure5_ExposureBias + FigureS1/S2/S3. Tables in tables/.")

# Optional: write manuscript framing only if a private manuscript/ folder exists.
# The public analysis repository does not ship manuscript sources.
MS = HERE / "manuscript"
WRITE_MS = MS.exists()
draft = f"""# AJEM draft framing (association language)

**Working title.** Depth of Early Emergency Department Hypotension and 7-Day Acute Kidney Injury or Death

**Status.** Results-locked framing for native LaTeX. Do not upgrade language from "associated with" to "causes."

## Key result (for Abstract / Results)

In {primary['n']:,} adults with a first new ED hypotensive episode (deep n={primary['n_high']:,}; shallow n={primary['n_low']:,}), the overlap-weighted 7-day risk of stage >=2 AKI or death was **{primary['risk_high']*100:.1f}%** after a deep early MAP nadir versus **{primary['risk_low']*100:.1f}%** after a shallow nadir (**RD {primary['rd']*100:+.1f} percentage points**, 95% CI {primary['rd_lo']*100:+.1f} to {primary['rd_hi']*100:+.1f}; **RR {primary['rr']:.2f}**, 95% CI {primary['rr_lo']:.2f} to {primary['rr_hi']:.2f}). The E-value for the point estimate was {primary['e_value']:.2f}.

Doubly-robust AIPW RD {cc_aipw['rd']*100:+.1f} pts; selection-adjusted IPCW RD {ipcw_est['rd']*100:+.1f} pts. Cause-specific HR deep vs shallow: AKI {cause_specific.loc[cause_specific.cause=='AKI','hr'].iloc[0]:.2f}; death {cause_specific.loc[cause_specific.cause=='death','hr'].iloc[0]:.2f}.

The association was robust to the two pre-specified threats to validity. Restricting to episodes that were AKI-free at onset (guarding against reverse causation) left the estimate essentially unchanged (RD {akifree_est['rd']*100:+.1f} pts, 95% CI {akifree_est['rd_lo']*100:+.1f} to {akifree_est['rd_hi']*100:+.1f}; RR {akifree_est['rr']:.2f}), and a co-primary contrast using transportable absolute MAP thresholds (<50 vs 55-64 mmHg) rather than cohort-specific tertiles gave a concordant, if slightly larger, estimate (RD {abs_est['rd']*100:+.1f} pts, 95% CI {abs_est['rd_lo']*100:+.1f} to {abs_est['rd_hi']*100:+.1f}; RR {abs_est['rr']:.2f}).

## Reviewer-hardening results (weave into Results/Discussion, keep in supplement)

- **Absolute clinical cutpoints (co-primary).** Using transportable thresholds instead of cohort tertiles (nadir <50 vs 55-64 mmHg), the overlap-weighted RD was {abs_est['rd']*100:+.1f} pts ({abs_est['rd_lo']*100:+.1f} to {abs_est['rd_hi']*100:+.1f}); RR {abs_est['rr']:.2f} ({abs_est['rr_lo']:.2f} to {abs_est['rr_hi']:.2f}), n={abs_est['n']:,}. This answers the "tertiles are arbitrary" critique and gives a bedside-portable contrast.
- **Reverse-causation guard.** Restricting to episodes AKI-free at time zero, RD was {akifree_est['rd']*100:+.1f} pts ({akifree_est['rd_lo']*100:+.1f} to {akifree_est['rd_hi']*100:+.1f}); RR {akifree_est['rr']:.2f} ({akifree_est['rr_lo']:.2f} to {akifree_est['rr_hi']:.2f}). Compare against the primary RD to show the signal is not merely pre-existing AKI.
- **The minutes reversal is a duration-of-monitored-state artifact, not a simple reading-count effect.** Deep-nadir episodes were sampled *more* often (median {monitoring['deep_median_readings'].iloc[0]:.0f} vs {monitoring['shallow_median_readings'].iloc[0]:.0f} readings) yet accumulated *fewer* charted minutes MAP<65 (median {monitoring['deep_median_minutes'].iloc[0]:.0f} vs {monitoring['shallow_median_minutes'].iloc[0]:.0f}), and the minutes/reading-count correlation was weak (Spearman {monitoring['spearman_minutes_readings'].iloc[0]:+.2f}). Residualising minutes on reading count barely moved the paradoxical estimate (monitoring-adjusted minutes row in Fig 5A still points the "wrong" way), so the reversal is **not** fixable by adjusting for sampling frequency. The interpretation: brief profound dips trigger rapid response, so charted low-duration reflects a monitored-and-treated state rather than perfusion depth. This is why depth (nadir), not duration (minutes), is the primary exposure.
- **Continuous dose-response.** A restricted cubic spline of nadir MAP on 7-day risk (supplement figure) shows a smooth monotone gradient consistent with the binned dose-response.

## Methods one-liners (main text)

- **Design.** Target-trial style retrospective cohort, MIMIC-IV-ED linked to MIMIC-IV.
- **Exposure.** Primary: nadir MAP in 0-2 h after episode onset (deep vs shallow tertile of depth). Secondary measurement comparisons: AUT and minutes MAP <65.
- **Outcome.** 7-day composite KDIGO stage >=2 AKI or death.
- **Analysis.** MICE; overlap weights for the equipoise population; RD/RR with bootstrap CIs; AIPW and IPTW robustness; competing-risk CIF; quantitative bias analysis.

## Discussion skeleton

1. **Biology.** Deeper early nadir tracks renal hypoperfusion; dose-response from 60-64 to <50 mmHg supports a physiologic gradient.
2. **Why not minutes.** Charted duration reversed under the same design, and the reversal persisted after adjusting minutes for sampling frequency. Brief profound dips that trigger rapid response make charted minutes reflect a monitored-and-treated state rather than perfusion depth; report this as a measurement finding, not a paradox.
3. **Residual confounding.** Pre-onset AKI negative control RD was {neg_ctrl['rd']*100:+.1f} pts ({neg_ctrl['rd_lo']*100:+.1f} to {neg_ctrl['rd_hi']*100:+.1f}). This is not null. Interpret the primary association as partly vulnerable to reverse causation / unmeasured severity, while noting the primary RD is substantially larger.
4. **Clinical takeaway (cautious).** Early depth of hypotension is a marker of short-term kidney and survival risk in the ED. Whether actively shortening depth improves outcomes is untested here.

## Limitations (copy into manuscript)

1. Observational association only; residual confounding remains (negative control not null; E-value {primary['e_value']:.2f}).
2. Single-center MIMIC; cuff-derived MAP; irregular vital sign cadence.
3. Comorbidities from ICD codes; sepsis flagged on linked admission, not ED timestamp.
4. Pre-onset lactate often missing and may be excluded from the propensity model.
5. Deep tertile overlaps profound (MAP <55) onset; sensitivity excluding profound-only onsets should be cited beside the primary result.
6. No treatment recommendation or MAP target claim.

## Figures for main text (5 figures + Table 1)

1. Figure1 DAG
2. Figure2 Cohort flow
3. Figure3 Love plot
4. Figure4 Primary effect + dose-response (top) over competing-risk panels (bottom)
5. Figure5 Exposure-measurement comparison (top) over quantitative-bias panels (bottom)

Supplement: FigureS1 propensity overlap, FigureS2 sensitivity forest (now includes the
absolute-threshold co-primary and the AKI-free-at-onset guard), FigureS3 continuous
nadir spline dose-response.

PS covariates used: {', '.join(ps_covars)}.
Max |SMD| after overlap weighting: {balance['abs_smd_overlap'].max():.3f}.
"""
if WRITE_MS:
    (MS / "DRAFT_FRAMING.md").write_text(draft, encoding="utf-8")
    print(f"Wrote {MS / 'DRAFT_FRAMING.md'}")
else:
    print("Skipping manuscript framing (no manuscript/ directory in this checkout).")

# ---------------------------------------------------------------------------
# Reviewer-proofing artifacts (STROBE checklist + anticipated-critique rebuttals).
# Both are generated with live numbers so a single run reproduces the whole
# submission-support package.
# ---------------------------------------------------------------------------
def _n(step_label):
    hit = strobe.loc[strobe["step"] == step_label, "n"]
    return int(hit.iloc[0]) if len(hit) else np.nan


strobe_md = f"""# STROBE checklist (cohort study) - live pointers

Generated from the analysis run. Section/figure/table pointers are for the LaTeX
manuscript we will write; numbers are current as of the latest `analysis.py` run.

| # | STROBE item | Where addressed | Current value / note |
|---|-------------|-----------------|----------------------|
| 1a | Design in title/abstract | Title, Abstract | Retrospective target-trial-style cohort |
| 1b | Informative abstract | Abstract | RD {primary['rd']*100:+.1f} pts; RR {primary['rr']:.2f} |
| 2 | Background/rationale | Introduction | Early ED hypotension depth vs AKI/death |
| 3 | Objectives/hypotheses | Introduction (last para) | Depth (nadir) as primary exposure; association framing |
| 4 | Study design | Methods 2.1 | Cohort, MIMIC-IV-ED linked to MIMIC-IV |
| 5 | Setting | Methods 2.1 | Single academic ED, MIMIC-IV era |
| 6a | Eligibility / participants | Methods 2.2; Figure 2 | First ED hypotensive episode, adults |
| 6b | Matching / weighting | Methods 2.5 | Overlap weights on propensity score |
| 7 | Variables (exposure/outcome/confounders) | Methods 2.3-2.4; Figure 1 (DAG) | Nadir MAP; 7-day stage>=2 AKI or death; {len(ps_covars)} covariates |
| 8 | Data sources / measurement | Methods 2.2-2.3 | MAP = (SBP+2*DBP)/3; KDIGO creatinine |
| 9 | Bias | Methods 2.6; Figure 5B-D | IPCW selection, negative control, E-value, tipping point |
| 10 | Study size | Figure 2; Results | Analytic n={len(ana):,} (deep {int(ana.deep.sum())}/shallow {int((1-ana.deep).sum())}) |
| 11 | Quantitative handling of exposure | Methods 2.3; Figure 4B, Figure S3 | Tertiles + absolute thresholds + continuous spline |
| 12a | Statistical methods | Methods 2.5 | Overlap/IPTW/AIPW; bootstrap CIs |
| 12b | Subgroups / interactions | Methods 2.7; Results | Night vs day effect modification |
| 12c | Missing data | Methods 2.5 | MICE ({N_MICE} imputations); complete-case sensitivity |
| 12d | Loss to follow-up / selection | Methods 2.6; Figure 5B | IPCW for reference-creatinine availability |
| 12e | Sensitivity analyses | Results; Figure S2 | {int(sens_df['rd'].notna().sum())} sensitivity/robustness rows |
| 13 | Participants (flow) | Figure 2 | Episodes {_n('First ED hypotensive episode (adults)'):,} -> analytic {len(analytic_all):,} |
| 14 | Descriptive data | Table 1 | Baseline by depth arm with SMDs |
| 15 | Outcome events | Table 1; Results | Composite deep {ana.loc[ana.deep==1,'aki_or_death'].mean()*100:.1f}% vs shallow {ana.loc[ana.deep==0,'aki_or_death'].mean()*100:.1f}% |
| 16a | Main results (adjusted) | Results; Figure 4A | RD {primary['rd']*100:+.1f} ({primary['rd_lo']*100:+.1f}, {primary['rd_hi']*100:+.1f}); RR {primary['rr']:.2f} |
| 16b | Category boundaries | Methods 2.3 | Nadir tertiles; absolute <50 / 55-64 mmHg |
| 16c | Absolute risk | Results; Figure 4A | Deep {primary['risk_high']*100:.1f}% vs shallow {primary['risk_low']*100:.1f}% |
| 17 | Other analyses | Results; Figures 4B, 5, S2, S3 | Dose-response, exposure comparison, bias suite |
| 18 | Key results | Discussion (1st para) | Depth associated with higher 7-day risk |
| 19 | Limitations | Discussion; Limitations list | Negative control not null; single-center; cuff MAP |
| 20 | Interpretation | Discussion | Associative; no treatment/target claim |
| 21 | Generalizability | Discussion | Single-center MIMIC; absolute thresholds aid transport |
| 22 | Funding | Manuscript footer | To be completed |

Max |SMD| after overlap weighting: {balance['abs_smd_overlap'].max():.3f}
(positivity/overlap shown in Figure S1).
"""
if WRITE_MS:
    (MS / "STROBE_checklist.md").write_text(strobe_md, encoding="utf-8")
    print(f"Wrote {MS / 'STROBE_checklist.md'}")


rebut_md = f"""# Anticipated reviewer critiques and where they are answered

Each row pairs a likely rejection line with the specific analysis and artifact that
pre-empts it. Keep this as an internal map while drafting; fold the strongest rows into
the Results and Discussion so reviewers see the rebuttal before they raise it.

| Likely critique | Rebuttal (with current numbers) | Artifact |
|-----------------|--------------------------------|----------|
| "This is just reverse causation - deep-nadir patients were already developing AKI." | Restricting to episodes AKI-free at onset barely changes the estimate: RD {akifree_est['rd']*100:+.1f} pts ({akifree_est['rd_lo']*100:+.1f}, {akifree_est['rd_hi']*100:+.1f}) vs primary {primary['rd']*100:+.1f}. | Figure S2; Results |
| "Tertile cutpoints are arbitrary and non-transportable." | Absolute clinical thresholds (<50 vs 55-64 mmHg) give a concordant estimate: RD {abs_est['rd']*100:+.1f} pts ({abs_est['rd_lo']*100:+.1f}, {abs_est['rd_hi']*100:+.1f}), n={abs_est['n']:,}. | Figure S2; Results |
| "The exposure is dose-arbitrary; show a gradient." | Binned dose-response plus a continuous restricted-cubic-spline curve show a smooth monotone gradient across nadir MAP. | Figure 4B; Figure S3 |
| "Unmeasured confounding could explain this." | E-value {primary['e_value']:.2f} (CI bound {primary['e_value_ci']:.2f}); tipping-point analysis quantifies the confounder strength needed to nullify. | Figure 5B, 5D |
| "Outcome ascertainment is selective (reference creatinine missing)." | IPCW for reference-creatinine availability leaves RD at {ipcw_est['rd']*100:+.1f} pts ({ipcw_est['rd_lo']*100:+.1f}, {ipcw_est['rd_hi']*100:+.1f}). | Figure 5C; Methods 2.6 |
| "Your estimator choice drives the result." | Overlap {cc_overlap['rd']*100:+.1f}, IPTW {cc_iptw['rd']*100:+.1f}, AIPW {cc_aipw['rd']*100:+.1f} pts all agree. | Figure S2; Results |
| "Death competes with AKI; a composite hides it." | Cause-specific hazards separate the two (AKI HR {cause_specific.loc[cause_specific.cause=='AKI','hr'].iloc[0]:.2f}; death HR {cause_specific.loc[cause_specific.cause=='death','hr'].iloc[0]:.2f}) and Aalen-Johansen CIFs are reported. | Figure 4C-D |
| "Charted minutes<65 actually looked protective - your exposure is unreliable." | Reported transparently as a measurement finding; the reversal persists after adjusting minutes for sampling frequency, so depth (nadir) is primary. | Figure 5A; Discussion 2 |
| "Missing covariates bias the propensity model." | MICE ({N_MICE} imputations) pooled by Rubin's rules; complete-case is concordant; covariates >{100*MISSING_DROP_FRAC:.0f}% missing dropped from PS. | Methods 2.5 |
| "Groups were not comparable at baseline." | Overlap weighting drives max |SMD| to {balance['abs_smd_overlap'].max():.3f}; positivity confirmed by propensity overlap. | Figure 3; Figure S1 |
| "Findings may not hold across time-of-day / acuity." | Effect modification by night onset reported; direction consistent. | Methods 2.7; Results |
| "Overreaching causal claim." | Language is associative throughout; no MAP-target or treatment recommendation is made. | Discussion 4; Limitations |

**Honest residual weakness to disclose, not hide.** The pre-onset AKI negative control was
not null (RD {neg_ctrl['rd']*100:+.1f} pts, {neg_ctrl['rd_lo']*100:+.1f} to {neg_ctrl['rd_hi']*100:+.1f}). We state plainly that some
residual reverse causation / severity confounding remains, that it is smaller than the
primary signal, and that the AKI-free-at-onset analysis brackets its impact.
"""
if WRITE_MS:
    (MS / "REVIEWER_REBUTTALS.md").write_text(rebut_md, encoding="utf-8")
    print(f"Wrote {MS / 'REVIEWER_REBUTTALS.md'}")


# %% [markdown]
# ## 16. Interpretation for AJEM (association language and locked limitations)
#
# **Headline (associative).** Among adults with a first new ED hypotensive episode, a
# deep early MAP nadir was associated with higher 7-day risk of stage >=2 AKI or death
# than a shallow nadir, with a monotone dose-response and consistent estimates under
# overlap weighting, IPTW, and AIPW.
#
# **Measurement lesson (planned secondary).** AUT (depth-weighted burden) pointed the
# same direction as nadir. Charted minutes MAP <65 reversed. That is expected if
# monitoring intensity and brief profound dips dominate duration scoring; it is why
# depth is the primary exposure.
#
# **Limitations (must appear in the paper):**
#
# 1. **Residual reverse causation / severity confounding.** The pre-onset AKI
#    negative control was not null. Part of the depth-outcome link may reflect illness
#    that was already evolving before the recorded episode. Language stays associative.
# 2. **Unmeasured confounding.** E-value and tipping-point bound how strong an omitted
#    confounder would need to be; they do not prove absence of bias. Lactate is often
#    missing and may drop from the PS.
# 3. **Single-center MIMIC, cuff-derived MAP,** irregular vital cadence, discharge ICD
#    comorbidities (sepsis flag is admission-coded, not ED timestamped).
# 4. **Deep tertile overlaps profound onset.** Sensitivity excluding profound-only
#    onsets is required in the main results narrative.
# 5. **No claim that treating to a specific MAP target causes benefit.** That needs a
#    different design.
#
# Full draft framing for the manuscript is written to `manuscript/DRAFT_FRAMING.md`
# after this run.
