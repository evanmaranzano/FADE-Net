"""Generate vector-first figures for the FADE-Net Chinese journal manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "paper" / "evidence" / "fivefold_summary.json"
OUTPUT_DIR = ROOT / "docs" / "paper" / "figures"

FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/simsun.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/msyh.ttc"),
)
FONT_PATH = next((path for path in FONT_CANDIDATES if path.is_file()), None)
ZH = font_manager.FontProperties(fname=str(FONT_PATH)) if FONT_PATH else font_manager.FontProperties()
ZH_BOLD = font_manager.FontProperties(fname=str(FONT_PATH), weight="bold") if FONT_PATH else font_manager.FontProperties(weight="bold")

COLORS = {
    "ink": "#20262e",
    "muted": "#66717f",
    "line": "#52606d",
    "grid": "#d8dee5",
    "blue": "#597b9d",
    "blue_light": "#e6eef5",
    "green": "#6f8f7a",
    "green_light": "#e8f0ea",
    "orange": "#aa7b4f",
    "orange_light": "#f3ebe2",
    "purple": "#786b8e",
    "purple_light": "#edeaf2",
    "gray_light": "#f3f5f7",
    "teacher": "#8c6d72",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    face: str = "white",
    edge: str = COLORS["line"],
    linestyle: str = "-",
    fontsize: float = 7.0,
    bold: bool = False,
    radius: float = 0.8,
    linewidth: float = 0.9,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.18,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontproperties=ZH_BOLD if bold else ZH,
        linespacing=1.22,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    linestyle: str = "-",
    linewidth: float = 0.9,
    mutation_scale: float = 7.5,
    connectionstyle: str = "arc3",
    zorder: int = 1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            mutation_scale=mutation_scale,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
    )


def panel_label(ax: plt.Axes, x: float, y: float, label: str) -> None:
    ax.text(x, y, label, fontsize=8.5, fontweight="bold", fontfamily="Times New Roman", va="top")


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=400, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")
    panel_label(ax, 0.8, 63.2, "(a)")
    ax.text(5.2, 61.6, "推理主链", fontsize=7.5, fontproperties=ZH_BOLD, va="center")
    ax.plot([13, 98], [58.7, 58.7], color=COLORS["grid"], linewidth=0.6)

    box(ax, 2.0, 29.0, 10.0, 10.0, "输入人脸\n256×256", face=COLORS["gray_light"], bold=True)
    box(ax, 16.0, 27.0, 14.0, 14.0, "MobileNetV4\nConv-Small / Medium", face=COLORS["blue_light"], edge=COLORS["blue"], bold=True)
    arrow(ax, (12.0, 34.0), (16.0, 34.0))

    box(ax, 34.0, 44.0, 9.0, 7.0, "$F_1$\n浅层", face=COLORS["gray_light"], fontsize=6.6)
    box(ax, 34.0, 30.5, 9.0, 7.0, "$F_2$\n中层", face=COLORS["gray_light"], fontsize=6.6)
    box(ax, 34.0, 17.0, 9.0, 7.0, "$F_3$\n深层", face=COLORS["gray_light"], fontsize=6.6)
    for target_y in (47.5, 34.0, 20.5):
        arrow(ax, (30.0, 34.0), (34.0, target_y), connectionstyle="arc3,rad=0.05")

    box(ax, 47.0, 25.5, 13.0, 17.0, "DCSR\n分布条件\n尺度路由", face=COLORS["green_light"], edge=COLORS["green"], bold=True)
    for source_y, target_y in ((47.5, 39.5), (34.0, 34.0), (20.5, 28.5)):
        arrow(ax, (43.0, source_y), (47.0, target_y))

    box(ax, 46.0, 49.0, 15.0, 8.0, "粗分布头  $p^{c}$", face=COLORS["orange_light"], edge=COLORS["orange"], fontsize=6.8)
    arrow(ax, (42.8, 23.0), (49.0, 49.0), color=COLORS["orange"], connectionstyle="arc3,rad=-0.15")
    arrow(ax, (53.5, 49.0), (53.5, 42.5), color=COLORS["orange"])

    box(ax, 64.0, 27.0, 11.5, 14.0, "主分布头\n$p^{m}$ → " + r"$\mu$", face=COLORS["purple_light"], edge=COLORS["purple"], bold=True)
    arrow(ax, (60.0, 34.0), (64.0, 34.0))
    box(ax, 79.0, 26.0, 11.5, 16.0, "CGBR\n门控有界\n残差细化", face=COLORS["orange_light"], edge=COLORS["orange"], bold=True)
    arrow(ax, (75.5, 34.0), (79.0, 34.0))
    box(ax, 93.0, 29.0, 6.0, 10.0, "年龄\n" + r"$\hat y$", face=COLORS["gray_light"], bold=True)
    arrow(ax, (90.5, 34.0), (93.0, 34.0))

    panel_label(ax, 0.8, 13.8, "(b)")
    ax.text(5.2, 12.2, "仅训练时启用的同折蒸馏路径", fontsize=7.5, fontproperties=ZH_BOLD, va="center")
    box(ax, 18.0, 3.0, 16.0, 8.0, "FaRL ViT-B/16\n同折教师", face="white", edge=COLORS["teacher"], linestyle="--", fontsize=6.8)
    box(ax, 42.0, 3.0, 14.0, 8.0, "教师分布  $p^{T}$", face="white", edge=COLORS["teacher"], linestyle="--", fontsize=6.8)
    box(ax, 64.0, 3.0, 13.0, 8.0, "KL 蒸馏", face="white", edge=COLORS["teacher"], linestyle="--", fontsize=6.8)
    arrow(ax, (34.0, 7.0), (42.0, 7.0), color=COLORS["teacher"], linestyle="--")
    arrow(ax, (56.0, 7.0), (64.0, 7.0), color=COLORS["teacher"], linestyle="--")
    arrow(ax, (77.0, 7.0), (69.5, 27.0), color=COLORS["teacher"], linestyle="--", connectionstyle="arc3,rad=0.12")
    ax.text(81.5, 6.9, "部署时移除", fontsize=6.2, fontproperties=ZH, color=COLORS["teacher"], va="center")
    save_figure(fig, "fig1_fade_net_architecture")


def mechanism_figure() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"wspace": 0.08})
    for ax in axes:
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 48)
        ax.axis("off")

    left, right = axes
    panel_label(left, 0.5, 47.3, "(a)")
    left.text(4.2, 45.2, "分布条件尺度路由（DCSR）", fontsize=7.8, fontproperties=ZH_BOLD, va="center")
    for index, (label, y) in enumerate((("$F_1$", 33.5), ("$F_2$", 23.5), ("$F_3$", 13.5))):
        box(left, 2.0, y, 6.5, 6.0, label, face=COLORS["gray_light"], fontsize=7.0)
        box(left, 11.5, y, 9.0, 6.0, "适配器", face=COLORS["blue_light"], edge=COLORS["blue"], fontsize=6.5)
        arrow(left, (8.5, y + 3.0), (11.5, y + 3.0))
        arrow(left, (20.5, y + 3.0), (29.0, 26.0), connectionstyle=f"arc3,rad={(index - 1) * 0.10}")
    box(left, 29.0, 19.5, 11.0, 13.0, "分组加权\n" + r"$\sum_i\alpha_{g,i}F_{g,i}$", face=COLORS["green_light"], edge=COLORS["green"], fontsize=6.8, bold=True)
    box(left, 42.0, 22.0, 7.0, 8.0, "$F^{f}$", face=COLORS["gray_light"], fontsize=7.0, bold=True)
    arrow(left, (40.0, 26.0), (42.0, 26.0))
    box(left, 2.5, 2.5, 18.0, 6.5, r"$[\operatorname{GAP}(F_3),s(p^c),e(p^c)]$", face=COLORS["orange_light"], edge=COLORS["orange"], fontsize=5.9)
    box(left, 24.0, 2.5, 10.0, 6.5, "MLP", face="white", edge=COLORS["orange"], fontsize=6.5)
    box(left, 37.0, 2.5, 12.0, 6.5, "组内 Softmax\n" + r"$\alpha_{g,i}$", face="white", edge=COLORS["orange"], fontsize=6.0)
    arrow(left, (20.5, 5.75), (24.0, 5.75), color=COLORS["orange"])
    arrow(left, (34.0, 5.75), (37.0, 5.75), color=COLORS["orange"])
    arrow(left, (43.0, 9.0), (35.0, 19.5), color=COLORS["orange"], connectionstyle="arc3,rad=0.08")
    left.text(3.0, 10.6, "$p^c$ 停止梯度", fontsize=5.9, fontproperties=ZH, color=COLORS["muted"])

    panel_label(right, 0.5, 47.3, "(b)")
    right.text(4.2, 45.2, "修正需求引导有界残差（CGBR）", fontsize=7.8, fontproperties=ZH_BOLD, va="center")
    box(right, 2.0, 32.0, 11.0, 7.0, "主分布 $p^m$", face=COLORS["purple_light"], edge=COLORS["purple"], fontsize=6.8)
    box(right, 2.0, 19.0, 11.0, 7.0, "融合特征 $F^f$", face=COLORS["green_light"], edge=COLORS["green"], fontsize=6.8)
    box(right, 17.0, 31.0, 12.0, 9.0, "不确定性描述\n$s(p^m),e(p^m)$", face="white", edge=COLORS["purple"], fontsize=6.2)
    arrow(right, (13.0, 35.5), (17.0, 35.5), color=COLORS["purple"])
    box(right, 33.0, 32.0, 8.5, 7.0, "门控 $g$\nSigmoid", face=COLORS["orange_light"], edge=COLORS["orange"], fontsize=6.3)
    arrow(right, (29.0, 35.5), (33.0, 35.5), color=COLORS["orange"])
    box(right, 17.0, 17.0, 12.0, 9.0, "特征—分布\n联合表征", face="white", edge=COLORS["green"], fontsize=6.2)
    arrow(right, (13.0, 22.5), (17.0, 22.5), color=COLORS["green"])
    arrow(right, (23.0, 31.0), (23.0, 26.0), color=COLORS["purple"])
    box(right, 33.0, 18.0, 8.5, 7.0, "残差 $r$\n" + r"$3\tanh(\cdot)$", face=COLORS["blue_light"], edge=COLORS["blue"], fontsize=6.1)
    arrow(right, (29.0, 21.5), (33.0, 21.5), color=COLORS["blue"])
    box(right, 8.0, 4.0, 34.0, 8.0, r"$\hat y=\operatorname{clip}(\mu+g\,r,\,0,\,80)$", face=COLORS["gray_light"], fontsize=7.0, bold=True)
    arrow(right, (37.25, 32.0), (34.0, 12.0), color=COLORS["orange"], connectionstyle="arc3,rad=0.08")
    arrow(right, (37.25, 18.0), (36.0, 12.0), color=COLORS["blue"], connectionstyle="arc3,rad=-0.06")
    arrow(right, (7.5, 32.0), (16.0, 12.0), color=COLORS["purple"], connectionstyle="arc3,rad=-0.12")
    right.text(2.5, 28.4, "$p^m$ 停止梯度", fontsize=5.9, fontproperties=ZH, color=COLORS["muted"])
    save_figure(fig, "fig2_dcsr_cgbr_mechanism")


def results_figure() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    small = report["results"]["small"]["folds"]
    medium = report["results"]["medium"]["folds"]
    ensemble = report["results"]["ensemble"]["folds"]
    x = list(range(5))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"wspace": 0.22})
    ax1, ax2 = axes
    ax1.text(-0.13, 1.04, "(a)", transform=ax1.transAxes, fontsize=8.5, fontweight="bold", fontfamily="Times New Roman", va="top")
    curves = (
        ("Small 1×", [f["test_1x_mae"] for f in small], COLORS["muted"], "o", "-"),
        ("Small TTA", [f["selected_test_mae"] for f in small], COLORS["muted"], "o", "--"),
        ("Medium 1×", [f["test_1x_mae"] for f in medium], COLORS["blue"], "s", "-"),
        ("Medium TTA", [f["selected_test_mae"] for f in medium], COLORS["blue"], "s", "--"),
    )
    for label, values, color, marker, linestyle in curves:
        ax1.plot(x, values, label=label, color=color, marker=marker, linestyle=linestyle, linewidth=1.05, markersize=3.8, markerfacecolor="white", markeredgewidth=0.9)
    ax1.set_title("单模型逐折结果", fontproperties=ZH_BOLD, fontsize=8.0, pad=5)
    ax1.set_xticks(x, [f"Fold{i}" for i in x], fontfamily="Times New Roman", fontsize=7)
    ax1.set_ylabel("MAE / 岁", fontproperties=ZH, fontsize=7.5)
    ax1.set_ylim(3.09, 3.25)
    ax1.grid(axis="y", color=COLORS["grid"], linewidth=0.5, linestyle="--")
    ax1.tick_params(axis="y", labelsize=7)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(prop=font_manager.FontProperties(fname=str(FONT_PATH), size=6.2) if FONT_PATH else None, frameon=False, loc="upper right", borderaxespad=0.2, handlelength=2.2)

    ax2.text(-0.13, 1.04, "(b)", transform=ax2.transAxes, fontsize=8.5, fontweight="bold", fontfamily="Times New Roman", va="top")
    ax2.plot(x, [f["test_1x_mae"] for f in ensemble], label="等权融合 1×", color=COLORS["purple"], marker="^", linestyle="-", linewidth=1.1, markersize=4.2, markerfacecolor="white", markeredgewidth=0.9)
    ax2.plot(x, [f["selected_test_mae"] for f in ensemble], label="等权融合 TTA", color=COLORS["purple"], marker="^", linestyle="--", linewidth=1.1, markersize=4.2, markerfacecolor="white", markeredgewidth=0.9)
    ax2.axhline(3.10, color=COLORS["orange"], linewidth=0.8, linestyle=":")
    ax2.text(4.05, 3.1015, "3.10", color=COLORS["orange"], fontsize=6.5, fontfamily="Times New Roman", ha="right", va="bottom")
    ax2.set_title("双模型等权融合性能上界", fontproperties=ZH_BOLD, fontsize=8.0, pad=5)
    ax2.set_xticks(x, [f"Fold{i}" for i in x], fontfamily="Times New Roman", fontsize=7)
    ax2.set_ylabel("MAE / 岁", fontproperties=ZH, fontsize=7.5)
    ax2.set_ylim(3.00, 3.12)
    ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.5, linestyle="--")
    ax2.tick_params(axis="y", labelsize=7)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(prop=font_manager.FontProperties(fname=str(FONT_PATH), size=6.2) if FONT_PATH else None, frameon=False, loc="upper right", borderaxespad=0.2, handlelength=2.2)

    for ax in axes:
        ax.set_xlabel("官方主体互斥划分", fontproperties=ZH, fontsize=7.2, labelpad=3)
    save_figure(fig, "fig3_fivefold_results")


def main() -> None:
    configure_matplotlib()
    architecture_figure()
    mechanism_figure()
    results_figure()
    for path in sorted(OUTPUT_DIR.glob("fig*")):
        print(path)


if __name__ == "__main__":
    main()
