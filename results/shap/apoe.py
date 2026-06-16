import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = [
    "Level 1\nDemographics",
    "Level 2\n+ RAVLT imm.",
    "Level 3\n+ MMSE",
    "Level 4\n+ EcogSPTotal",
]
without_apoe4 = [0.5242, 0.8554, 0.8923, 0.9123]
with_apoe4 = [0.6511, 0.8668, 0.9004, 0.9113]
x = range(len(labels))

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)

ax.plot(x, without_apoe4, color="#1b9e77", marker="o", linewidth=2.5,
        label="Without APOE4", zorder=3)
ax.plot(x, with_apoe4, color="#b8860b", marker="^", linewidth=2.5,
        linestyle="--", label="With APOE4", zorder=3)

ax.set_ylabel("AUC (OOF)", fontsize=11)
ax.set_ylim(0.45, 0.95)
ax.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)

ax.grid(axis="y", color="lightgray", linewidth=0.6)
ax.set_axisbelow(True)
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

ax.legend(loc="lower center", bbox_to_anchor=(0.2, -0.32), ncol=2,
          frameon=False, fontsize=10)

plt.subplots_adjust(bottom=0.32)
plt.savefig("Figure2_APOE4_ablation_line.tiff", dpi=300, bbox_inches="tight", format="tiff")
plt.savefig("Figure2_APOE4.png")
print("done")