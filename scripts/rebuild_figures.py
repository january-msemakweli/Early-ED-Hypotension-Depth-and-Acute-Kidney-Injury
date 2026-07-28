"""Regenerate publication figures with non-overlapping labels (from saved tables)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import FIGURES_DIR, TABLES_DIR, CACHE_DIR, HORIZON_DAYS  # noqa: E402

FIG = FIGURES_DIR
OUT = TABLES_DIR
CACHE = CACHE_DIR

_avail = {f.name for f in font_manager.fontManager.ttflist}
SERIF = "Times New Roman" if "Times New Roman" in _avail else "DejaVu Serif"
# Teal / coral / slate palette stolen from Hospital Variation in Liberation
# from Mechanical Ventilation (their chosen Figure 2 stack scheme).
PAL = {
    "ink": "#1a1a1a",
    "deep": "#E76F51",       # coral: deep nadir / harm accent
    "shallow": "#2A9D8F",    # teal: shallow nadir / favourable
    "accent": "#C1452B",     # darker coral
    "sky": "#1B7268",        # deep teal (secondary)
    "plum": "#264653",       # slate (dark neutral anchor)
    "green": "#2A9D8F",
    "amber": "#E9C46A",
    "muted": "#8C8880",      # warm neutral
    "grid": "#E6E9ED", "panel": "#F0F0F0", "bg": "#ffffff",
}
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


def panel_label(ax, text):
    """Put the panel tag in the axes title slot so it never covers data."""
    ax.set_title(text, loc="left", fontsize=12, fontweight="bold", pad=8,
                 color="#1A1A1A")


def ann(ax, x, y, text, **kw):
    defaults = dict(fontsize=9.2, color=PAL["ink"], zorder=8, clip_on=False,
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                              edgecolor="#D0D5DD", linewidth=0.7, alpha=0.96))
    defaults.update(kw)
    ax.text(x, y, text, **defaults)


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
    plt.close(fig)
    print(f"  saved {name}")


def figure_dag():
    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(-0.15, 8.35)
    nodes = {
        "severity": (2.2, 6.85, "Baseline illness\nseverity", PAL["amber"], "confounder"),
        "exposure": (2.2, 2.5, "Early MAP nadir\n(depth, 0-2 h)", PAL["deep"], "exposure"),
        "outcome": (9.8, 2.5, "7-day AKI\nor death", PAL["shallow"], "outcome"),
        "mediator": (6.0, 5.05, "Resuscitation\n(fluids, pressors)", PAL["sky"], "mediator"),
        "monitor": (6.0, 0.55, "Monitoring density /\nref-creatinine capture", PAL["muted"], "selection"),
    }
    for key, (x, y, lab, col, role) in nodes.items():
        ax.add_patch(Ellipse((x, y), 2.9, 1.35, facecolor=col, alpha=0.30,
                             edgecolor=col, linewidth=1.8, zorder=3))
        ax.text(x, y + 0.18, lab, ha="center", va="center", fontsize=10.0,
                fontweight="bold", color=PAL["ink"], zorder=5)
        ax.text(x, y - 0.40, role, ha="center", va="center", fontsize=8.2,
                style="italic", color="#555", zorder=5)

    def arrow(a, b, color=PAL["ink"], rad=0.0, lw=1.7, ls="-"):
        xa, ya = nodes[a][0], nodes[a][1]
        xb, yb = nodes[b][0], nodes[b][1]
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>",
                     mutation_scale=16, lw=lw, color=color, ls=ls,
                     connectionstyle=f"arc3,rad={rad}", shrinkA=40, shrinkB=40, zorder=2))

    arrow("severity", "exposure", PAL["amber"])
    arrow("severity", "outcome", PAL["amber"], rad=0.22)
    arrow("exposure", "outcome", PAL["deep"], lw=2.6)
    arrow("exposure", "mediator", PAL["sky"], rad=0.18, ls="--")
    arrow("mediator", "outcome", PAL["sky"], rad=-0.18, ls="--")
    arrow("severity", "mediator", PAL["amber"], rad=-0.12, lw=1.2, ls=":")
    arrow("exposure", "monitor", PAL["muted"], rad=-0.12, ls="--", lw=1.2)
    arrow("outcome", "monitor", PAL["muted"], rad=0.12, ls="--", lw=1.2)
    # Sit on the target path only (below the long confounding arc)
    ann(ax, 6.0, 1.95, "target association (estimated)", ha="center",
        fontsize=9.0, color=PAL["deep"], style="italic",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                  edgecolor=PAL["deep"], linewidth=0.8, alpha=0.96))
    leg = [
        Line2D([0], [0], color=PAL["amber"], lw=2, label="Confounding (adjusted: overlap/MICE/AIPW)"),
        Line2D([0], [0], color=PAL["sky"], lw=2, ls="--", label="Mediation (not adjusted)"),
        Line2D([0], [0], color=PAL["muted"], lw=2, ls="--", label="Selection (IPCW / collider)"),
        Line2D([0], [0], color=PAL["deep"], lw=2.4, label="Target association"),
    ]
    ax.legend(handles=leg, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995),
              fontsize=9.0, frameon=True, facecolor="white", edgecolor="#D0D5DD")
    save_fig(fig, "Figure1_DAG")


def figure_flow(strobe):
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.6), gridspec_kw={"width_ratios": [1.0, 1.05]})
    fig.subplots_adjust(wspace=0.26, top=0.90, bottom=0.11)
    ax = axes[0]; ax.axis("off"); ax.set_xlim(0, 10)
    n = len(strobe); ax.set_ylim(0.1, n + 0.95)
    panel_label(ax, "(A)  Cohort derivation")
    wrap = {
        "Reference creatinine available (AKI ascertainable)":
            "Reference creatinine available\n(AKI ascertainable)",
        "First ED hypotensive episode (adults)":
            "First ED hypotensive episode\n(adults)",
    }

    def y_of(i):
        return n - i
    for i, r in enumerate(strobe.itertuples()):
        y = y_of(i); key = i in (0, n - 1)
        ax.add_patch(FancyBboxPatch((0.95, y - 0.38), 5.0, 0.76,
                     boxstyle="round,pad=0.04,rounding_size=0.10",
                     facecolor="#DCEFEC" if key else "#FFFFFF",
                     edgecolor=PAL["ink"], linewidth=1.1, zorder=3))
        ax.text(3.45, y + 0.12, wrap.get(r.step, r.step), ha="center", va="center",
                fontsize=8.4, fontweight="bold", color=PAL["ink"], zorder=4, linespacing=1.25)
        ax.text(3.45, y - 0.22, f"n = {int(r.n):,}", ha="center", va="center",
                fontsize=9.0, fontweight="bold", color=PAL["ink"], zorder=4)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((3.45, y - 0.38), (3.45, y_of(i + 1) + 0.38),
                         arrowstyle="-|>", mutation_scale=15, lw=1.4, color=PAL["ink"], zorder=2))
    for i in range(n - 1):
        yi = (y_of(i) + y_of(i + 1)) / 2
        drop = int(strobe.iloc[i].n) - int(strobe.iloc[i + 1].n)
        if drop <= 0:
            continue
        ax.add_patch(FancyArrowPatch((3.45, yi), (6.5, yi), arrowstyle="-|>",
                     mutation_scale=12, lw=1.2, color=PAL["muted"], ls=(0, (4, 2)), zorder=2))
        ax.add_patch(FancyBboxPatch((6.5, yi - 0.22), 3.15, 0.44,
                     boxstyle="round,pad=0.03,rounding_size=0.08",
                     facecolor="#F7F5F2", edgecolor=PAL["muted"], linewidth=1.1, zorder=3))
        ax.text(8.075, yi, f"Excluded  {drop:,}", ha="center", va="center",
                fontsize=8.6, color=PAL["ink"], zorder=4)

    ax = axes[1]; panel(ax, "x"); panel_label(ax, "(B)  Exposure: MAP trajectory over 2 hours")
    tt = np.linspace(0, 120, 240)
    deep_tr = 78 - 34 * np.exp(-((tt - 25) ** 2) / 300)
    shal_tr = 74 - 12 * np.exp(-((tt - 40) ** 2) / 1400)
    y_nadir = float(deep_tr.min())
    ax.axhspan(35, 65, color=PAL["deep"], alpha=0.05, zorder=0)
    ax.fill_between(tt, deep_tr, 65, where=(deep_tr < 65), color=PAL["deep"], alpha=0.16, zorder=1)
    ax.axhline(65, color=PAL["ink"], lw=1.1, ls="--", zorder=3)
    ann(ax, 118, 66.2, "MAP 65 mmHg threshold", ha="right", va="bottom", fontsize=8.6)
    ax.plot(tt, deep_tr, color=PAL["deep"], lw=2.8, zorder=5, label="Deep nadir (top tertile)")
    ax.plot(tt, shal_tr, color=PAL["shallow"], lw=2.8, zorder=5, label="Shallow nadir (bottom tertile)")
    ax.scatter([25], [y_nadir], s=80, color=PAL["deep"], edgecolor="white",
               linewidth=1.2, zorder=6)
    ax.annotate("nadir (depth) = exposure", xy=(25, y_nadir), xytext=(47, 45.5),
                fontsize=9.0, color=PAL["deep"], va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=PAL["deep"],
                          lw=0.8, alpha=0.97),
                arrowprops=dict(arrowstyle="-|>", color=PAL["deep"], lw=1.3,
                                shrinkA=2, shrinkB=6,
                                connectionstyle="arc3,rad=-0.25"))
    ann(ax, 26, 58.5, "AUT = area under 65\n(depth x time)", fontsize=8.4,
        color="#7F2A18", ha="center", va="center")  # dark coral for AUT label
    ax.set_xlim(0, 120); ax.set_ylim(35, 90)
    ax.set_xlabel("Minutes after episode onset"); ax.set_ylabel("MAP (mmHg)")
    ax.legend(loc="upper right", fontsize=8.8, frameon=True, facecolor="white",
              edgecolor="#D0D5DD")
    save_fig(fig, "Figure2_CohortFlow")


def figure_love(balance):
    bal = balance.sort_values("smd_unweighted", key=lambda s: s.abs())
    labmap = {"age": "Age", "pre_hr": "Pre-onset HR", "pre_o2": "Pre-onset SpO2",
              "ref_cr": "Reference creatinine", "acuity": "Triage acuity",
              "pre_lactate": "Pre-onset lactate", "sex_male": "Male sex",
              "night_onset": "Night onset", "chf": "Heart failure",
              "cirrhosis": "Cirrhosis", "diabetes": "Diabetes", "ckd": "CKD (non-ESRD)",
              "cad": "Coronary disease", "copd": "COPD", "sepsis": "Sepsis (admission)"}
    fig, ax = plt.subplots(figsize=(8.8, 5.8)); panel(ax, "x")
    y = np.arange(len(bal))
    for yi, u, a in zip(y, bal["smd_unweighted"], bal["smd_overlap"]):
        ax.plot([abs(u), abs(a)], [yi, yi], color="#C6CBD1", lw=1.6, zorder=2)
    ax.scatter(bal["smd_unweighted"].abs(), y, s=95, color=PAL["muted"],
               edgecolor="white", zorder=4, label="Unweighted")
    ax.scatter(bal["smd_overlap"].abs(), y, s=100, color=PAL["deep"],
               edgecolor="white", zorder=5, label="Overlap-weighted")
    ax.axvline(0.1, ls="--", color=PAL["ink"], lw=1.3, zorder=1)
    ann(ax, 0.112, len(bal) - 0.7, "|SMD| = 0.10", ha="left", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([labmap.get(c, c) for c in bal["covariate"]])
    ax.set_xlabel("Absolute standardized mean difference")
    ax.set_xlim(-0.015, max(0.34, float(bal["smd_unweighted"].abs().max()) * 1.12))
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#D0D5DD")
    ax.set_title("Covariate balance before and after overlap weighting", pad=10)
    fig.tight_layout()
    save_fig(fig, "Figure3_LovePlot")


def figure_primary_competing(primary, dose_df, cif_df, cause_specific):
    """Stacked figure: primary effect (top) over competing-risk panels (bottom).
    Each block keeps its original standalone proportions and look."""
    fig = plt.figure(figsize=(13.2, 11.4))
    gsT = fig.add_gridspec(1, 2, top=0.88, bottom=0.535, left=0.07, right=0.97,
                           wspace=0.28, width_ratios=[1, 1.15])
    gsB = fig.add_gridspec(1, 2, top=0.445, bottom=0.07, left=0.07, right=0.97,
                           wspace=0.22, width_ratios=[1.25, 1.0])

    # ---------- Top block header ----------
    fig.text(0.5, 0.965, "Depth of early ED hypotension and 7-day AKI or death",
             ha="center", va="center", fontsize=14.5, fontweight="bold", color=PAL["ink"])
    fig.text(0.5, 0.925,
             f"RD {primary['rd']*100:+.1f} pts ({primary['rd_lo']*100:+.1f}, {primary['rd_hi']*100:+.1f})"
             f"     RR {primary['rr']:.2f} ({primary['rr_lo']:.2f}, {primary['rr_hi']:.2f})"
             f"     E-value {primary['e_value']:.2f}",
             ha="center", va="center", fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D0D5DD", alpha=0.98))

    # ---------- (A) Overlap-weighted primary outcome ----------
    ax = fig.add_subplot(gsT[0, 0]); panel(ax, "y"); panel_label(ax, "(A)  Overlap-weighted primary outcome")
    risks = [primary["risk_low"] * 100, primary["risk_high"] * 100]
    los = [primary["risk_low_lo"] * 100, primary["risk_high_lo"] * 100]
    his = [primary["risk_low_hi"] * 100, primary["risk_high_hi"] * 100]
    xb = np.arange(2)
    ax.bar(xb, risks, width=0.55, color=[PAL["shallow"], PAL["deep"]], edgecolor="white", zorder=3)
    ax.errorbar(xb, risks, yerr=[np.array(risks) - los, np.array(his) - risks], fmt="none",
                ecolor=PAL["ink"], elinewidth=1.6, capsize=6, zorder=4)
    for x, r, h in zip(xb, risks, his):
        ax.text(x, h + 0.85, f"{r:.1f}%", ha="center", fontweight="bold", fontsize=12, zorder=5)
    ax.set_xticks(xb); ax.set_xticklabels(["Shallow nadir", "Deep nadir"])
    ax.set_ylabel("7-day risk of AKI or death (%)")
    ax.set_ylim(0, max(his) * 1.38)

    # ---------- (B) Dose-response by depth ----------
    ax = fig.add_subplot(gsT[0, 1]); panel(ax); panel_label(ax, "(B)  Dose-response by depth")
    dd = dose_df.reset_index(drop=True); xd = np.arange(len(dd))
    ax.plot(xd, dd["risk"] * 100, "-o", color=PAL["deep"], lw=2.6, ms=10,
            markeredgecolor="white", zorder=4)
    ax.fill_between(xd, dd["lo"] * 100, dd["hi"] * 100, color=PAL["deep"], alpha=0.15, zorder=2)
    for x, (_, r) in zip(xd, dd.iterrows()):
        ann(ax, x, r["hi"] * 100 + 0.85, f"{r['risk']*100:.1f}%", ha="center",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92))
    ax.set_xticks(xd)
    ax.set_xticklabels([f"{b}\n(n={int(n)})" for b, n in zip(dd["band"], dd["n"])])
    ax.invert_xaxis()
    ax.set_ylim(dd["lo"].min() * 100 - 1.2, dd["hi"].max() * 100 + 3.6)
    ax.set_xlabel("Nadir MAP in first 2 hours (mmHg)")
    ax.set_ylabel("7-day risk of AKI or death (%)")

    # ---------- (C) Cumulative incidence ----------
    ax = fig.add_subplot(gsB[0, 0]); panel(ax); panel_label(ax, "(C)  Cumulative incidence (Aalen-Johansen)")
    for arm, col in [("deep", PAL["deep"]), ("shallow", PAL["shallow"])]:
        s = cif_df[cif_df.arm == arm]
        ax.step(s["t"], s["cif_aki"] * 100, where="post", color=col, lw=2.6, label=f"AKI, {arm}")
        ax.step(s["t"], s["cif_death"] * 100, where="post", color=col, lw=2.0, ls="--",
                label=f"Death, {arm}")
    top = cif_df[["cif_aki", "cif_death"]].max().max() * 100
    ax.set_xlim(0, HORIZON_DAYS); ax.set_ylim(0, top * 1.38)
    ax.set_xlabel("Days from onset"); ax.set_ylabel("Cumulative incidence (%)")
    ax.legend(fontsize=8.5, loc="lower right", ncol=2, frameon=True, facecolor="white",
              edgecolor="#D0D5DD", columnspacing=1.0)

    # ---------- (D) Cause-specific hazard ratio ----------
    ax = fig.add_subplot(gsB[0, 1]); panel(ax, "x"); panel_label(ax, "(D)  Cause-specific hazard ratio")
    cs = cause_specific.dropna(subset=["hr"]).reset_index(drop=True)
    y = np.arange(len(cs))
    ax.errorbar(cs["hr"], y, xerr=[cs["hr"] - cs["lo"], cs["hi"] - cs["hr"]], fmt="o",
                color=PAL["deep"], ecolor=PAL["ink"], elinewidth=1.6, capsize=6, ms=11,
                markeredgecolor="white", zorder=3)
    ax.axvline(1.0, color=PAL["ink"], ls="--", lw=1.2)
    x_right = float(cs["hi"].max()) + 0.08
    for yi, r in cs.iterrows():
        ax.text(x_right, yi, f"{r['hr']:.2f} ({r['lo']:.2f}, {r['hi']:.2f})",
                va="center", ha="left", fontsize=9.4, color=PAL["ink"],
                clip_on=False, zorder=8)
    ax.set_yticks(y)
    ax.set_yticklabels([str(c).upper() if str(c).lower() == "aki" else str(c).capitalize()
                        for c in cs["cause"]])
    ax.set_xlabel("Cause-specific HR (deep vs shallow)")
    ax.set_ylim(-0.7, len(cs) - 0.3)
    ax.set_xlim(0.75, x_right + 0.85)

    save_fig(fig, "Figure4_PrimaryCompeting")


def figure_measure_bias(expo_df, primary, tip_df, ipcw, neg):
    """Stacked figure: exposure-measurement comparison (top) over the three
    quantitative-bias panels (bottom). Replaces the old Figures 6 and 7."""
    fig = plt.figure(figsize=(15.6, 10.6))
    # Two stacked blocks that each keep the original standalone proportions:
    # the exposure forest on top (full height, own title) and the three
    # bias panels below (own "Quantitative bias analysis" header).
    gsT = fig.add_gridspec(1, 1, top=0.905, bottom=0.575, left=0.06, right=0.98)
    gsB = fig.add_gridspec(1, 3, top=0.505, bottom=0.075, left=0.06, right=0.98,
                           wspace=0.32)

    # ---------- Top: exposure-measurement comparison (spans all columns) ----------
    axT = fig.add_subplot(gsT[0, 0]); panel(axT, "x")
    ed = expo_df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(ed))
    cols = [PAL["deep"] if r["rd"] > 0 else PAL["muted"] for _, r in ed.iterrows()]
    axT.errorbar(ed["rd"] * 100, y,
                 xerr=[np.clip((ed["rd"] - ed["rd_lo"]) * 100, 0, None),
                       np.clip((ed["rd_hi"] - ed["rd"]) * 100, 0, None)],
                 fmt="none", ecolor=PAL["ink"], elinewidth=1.7, capsize=6, zorder=2)
    axT.scatter(ed["rd"] * 100, y, s=150, c=cols, edgecolor="white", zorder=3)
    axT.axvline(0, color=PAL["ink"], ls="--", lw=1.3)
    x_lab = float(ed["rd_hi"].max() * 100) + 0.9
    for yi, (_, r) in zip(y, ed.iterrows()):
        axT.text(x_lab, yi,
                 f"RD {r['rd']*100:+.1f} ({r['rd_lo']*100:+.1f}, {r['rd_hi']*100:+.1f}); RR {r['rr']:.2f}",
                 va="center", ha="left", fontsize=9.6, color=PAL["ink"], clip_on=False, zorder=8)
    axT.set_yticks(y); axT.set_yticklabels(ed["exposure"])
    axT.set_xlabel("Overlap-weighted 7-day risk difference, AKI or death (pts)")
    axT.set_title("Planned exposure-measurement comparison:\n"
                  "depth harms; charted minutes reverse (measurement artifact)",
                  fontsize=13, pad=10)
    xmin = float(ed["rd_lo"].min() * 100) - 1.2
    axT.set_xlim(xmin, x_lab + 7.5)
    axT.set_ylim(-0.6, len(ed) - 0.4)

    # ---------- (A) E-value ----------
    ax = fig.add_subplot(gsB[0, 0]); panel(ax); panel_label(ax, "(A)  E-value")
    rr, rr_lo = primary["rr"], primary["rr_lo"]
    gg = np.linspace(1.0, max(4.2, primary["e_value"] + 1.2), 220)

    def partner(g, target):
        return np.where(g > target, (target * (g - 1)) / (g - target) + 1, np.nan)

    ax.plot(gg, partner(gg, rr), color=PAL["deep"], lw=2.4, label=f"Point estimate (RR {rr:.2f})")
    if rr_lo > 1:
        ax.plot(gg, partner(gg, rr_lo), color=PAL["shallow"], lw=2.0, ls="--",
                label=f"CI bound (RR {rr_lo:.2f})")
    ev = primary["e_value"]
    ax.scatter([ev], [ev], s=90, color=PAL["deep"], edgecolor="white", zorder=6)
    ax.annotate(f"E-value = {ev:.2f}", xy=(ev, ev), xytext=(ev + 0.55, ev - 0.85),
                fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#D0D5DD", alpha=0.96),
                arrowprops=dict(arrowstyle="-|>", color=PAL["ink"], lw=1.0,
                                connectionstyle="arc3,rad=-0.2"))
    ax.set_xlabel("Confounder-exposure association")
    ax.set_ylabel("Confounder-outcome association")
    ax.set_xlim(1, gg.max()); ax.set_ylim(1, gg.max())
    ax.legend(fontsize=8.4, loc="upper right", frameon=True, facecolor="white",
              edgecolor="#D0D5DD")

    # ---------- (C) Falsification and selection ----------
    ax = fig.add_subplot(gsB[0, 1]); panel(ax, "x"); panel_label(ax, "(B)  Falsification and selection")
    rows = [("Primary\n(composite)", primary["rd"], primary["rd_lo"], primary["rd_hi"], PAL["deep"]),
            ("Selection-adjusted\n(IPCW)", ipcw["rd"], ipcw["rd_lo"], ipcw["rd_hi"], PAL["sky"]),
            ("Negative control\n(pre-onset AKI)", neg["rd"], neg["rd_lo"], neg["rd_hi"], PAL["muted"])]
    y = np.arange(len(rows))[::-1]
    hi_max = max(r[3] for r in rows) * 100
    x_lab = hi_max + 0.7
    for yi, (lab, rd, lo, hi, col) in zip(y, rows):
        ax.errorbar(rd * 100, yi, xerr=[[max(0, (rd - lo) * 100)], [max(0, (hi - rd) * 100)]],
                    fmt="o", color=col, ecolor=PAL["ink"], elinewidth=1.6, capsize=6, ms=11,
                    markeredgecolor="white", zorder=3)
        ax.text(x_lab, yi, f"{rd*100:+.1f} ({lo*100:+.1f}, {hi*100:+.1f})",
                va="center", ha="left", fontsize=9, color=PAL["ink"], clip_on=False, zorder=8)
    ax.axvline(0, color=PAL["ink"], ls="--", lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Risk difference (pts)")
    ax.set_xlim(-2.5, x_lab + 4.2)
    ax.set_ylim(-0.7, len(rows) - 0.2)

    # ---------- (D) Unmeasured-confounding tipping point ----------
    ax = fig.add_subplot(gsB[0, 2]); panel(ax); panel_label(ax, "(C)  Unmeasured-confounding tipping point")
    for g, col in zip([1.5, 2.0, 2.5, 3.0], [PAL["amber"], PAL["sky"], PAL["plum"], PAL["deep"]]):
        s = tip_df[tip_df.gamma == g]
        ax.plot(s["delta"], s["rr_adj"], lw=2.2, color=col, label=f"RR {g:.1f}")
    ax.axhline(1.0, color=PAL["ink"], ls="--", lw=1.3)
    ax.set_xlabel("Confounder prevalence difference (deep - shallow)")
    ax.set_ylabel("Bias-adjusted RR")
    ax.legend(fontsize=8.2, loc="upper right", title="Confounder-outcome",
              frameon=True, facecolor="white", edgecolor="#D0D5DD")

    save_fig(fig, "Figure5_ExposureBias")


def figure_supp(cc, sens_df, spline_df=None):
    fig, ax = plt.subplots(figsize=(8.0, 5.0)); panel(ax, "y")
    bins = np.linspace(0, 1, 31)
    ax.hist(cc.loc[cc.deep == 1, "ps"], bins=bins, color=PAL["deep"], alpha=0.6,
            density=True, edgecolor="white", label="Deep nadir")
    ax.hist(cc.loc[cc.deep == 0, "ps"], bins=bins, color=PAL["shallow"], alpha=0.55,
            density=True, edgecolor="white", label="Shallow nadir")
    ax.set_xlabel("Estimated propensity score (P[deep])"); ax.set_ylabel("Density")
    ax.legend(frameon=True, facecolor="white", edgecolor="#D0D5DD")
    ax.set_title("Propensity score overlap", pad=10)
    fig.tight_layout()
    save_fig(fig, "FigureS1_PropensityOverlap")

    sd = sens_df[sens_df["rd"].notna()].copy().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.6, max(6.0, 0.52 * len(sd) + 1.8))); panel(ax, "x")
    y = np.arange(len(sd))[::-1]
    ax.errorbar(sd["rd"] * 100, y,
                xerr=[np.clip((sd["rd"] - sd["rd_lo"]) * 100, 0, None),
                      np.clip((sd["rd_hi"] - sd["rd"]) * 100, 0, None)],
                fmt="o", color=PAL["ink"], ecolor="#666", elinewidth=1.5, capsize=5, ms=9, zorder=3)
    ax.axvline(0, color=PAL["deep"], ls="--", lw=1.3)
    x_lab = float(sd["rd_hi"].max() * 100) + 0.6
    for yi, (_, r) in zip(y, sd.iterrows()):
        ann(ax, x_lab, yi,
            f"{r['rd']*100:+.1f} ({r['rd_lo']*100:+.1f}, {r['rd_hi']*100:+.1f})",
            va="center", ha="left", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(sd["analysis"], fontsize=9.3)
    ax.set_xlabel("Overlap-weighted risk difference, AKI or death (pts)")
    ax.set_title("Sensitivity analyses", pad=10)
    ax.set_xlim(float(sd["rd_lo"].min() * 100) - 1.0, x_lab + 5.5)
    ax.set_ylim(-0.7, len(sd) - 0.3)
    fig.tight_layout()
    save_fig(fig, "FigureS2_SensitivityForest")

    if spline_df is not None and len(spline_df):
        fig, ax = plt.subplots(figsize=(8.6, 5.4)); panel(ax)
        sp = spline_df.sort_values("nadir")
        ax.fill_between(sp["nadir"], sp["lo"] * 100, sp["hi"] * 100,
                        color=PAL["deep"], alpha=0.16, zorder=2)
        ax.plot(sp["nadir"], sp["risk"] * 100, color=PAL["deep"], lw=2.8, zorder=4)
        ax.axvline(65, color=PAL["ink"], ls="--", lw=1.1, zorder=3)
        ann(ax, 64.4, ax.get_ylim()[1], "MAP 65 threshold", ha="right", va="top", fontsize=8.6)
        ax.invert_xaxis()
        ax.set_xlabel("Nadir MAP in first 2 hours (mmHg)")
        ax.set_ylabel("7-day risk of AKI or death (%)")
        ax.set_title("Continuous dose-response (restricted cubic spline)", pad=10)
        fig.tight_layout()
        save_fig(fig, "FigureS3_DoseSpline")


def main():
    apply_style()
    primary = json.loads((OUT / "run_summary.json").read_text(encoding="utf-8"))["primary"]
    bias = pd.read_csv(OUT / "bias_summary.csv").iloc[0]
    ipcw = {"rd": bias["ipcw_rd"], "rd_lo": bias["ipcw_rd_lo"], "rd_hi": bias["ipcw_rd_hi"]}
    neg = {"rd": bias["neg_ctrl_rd"], "rd_lo": bias["neg_ctrl_rd_lo"], "rd_hi": bias["neg_ctrl_rd_hi"]}
    primary["e_value"] = float(bias["e_value"])

    strobe = pd.read_csv(OUT / "strobe_flow_counts.csv")
    balance = pd.read_csv(OUT / "covariate_balance.csv")
    dose = pd.read_csv(OUT / "dose_response_nadir.csv")
    cif = pd.read_csv(OUT / "competing_risk_cif.csv")
    cs = pd.read_csv(OUT / "cause_specific_hr.csv")
    expo = pd.read_csv(OUT / "exposure_comparison.csv")
    tip = pd.read_csv(OUT / "tipping_point.csv")
    sens = pd.read_csv(OUT / "sensitivity_analyses.csv")
    spline_path = OUT / "dose_response_spline.csv"
    spline = pd.read_csv(spline_path) if spline_path.exists() else None
    cc = pd.read_parquet(CACHE / "complete_case.parquet")

    print("Rebuilding figures with non-overlapping labels ...")
    figure_dag()
    figure_flow(strobe)
    figure_love(balance)
    figure_primary_competing(primary, dose, cif, cs)
    figure_measure_bias(expo, primary, tip, ipcw, neg)
    figure_supp(cc, sens, spline)
    print("Done.")


if __name__ == "__main__":
    main()
