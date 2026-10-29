"""Measure fixed static landmarks in context_render_1 using local NCC matching.

Coordinates are on original 320x176 images; inspect crops for match validity.
"""

from pathlib import Path
import cv2
import numpy as np
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(
        "outputs/context_render_1/scene-0103_3e8750f331d7499e9b5123e9eb70f2e2/batch_000000/target_000002"
    )
    out = Path("outputs/geometry_audit/image_alignment")
    out.mkdir(parents=True, exist_ok=True)
    ims = [
        cv2.imread(str(root / name))
        for name in ["ground_truth.png", "context_0.png", "context_1.png"]
    ]
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 for im in ims]
    # Stable architecture/street furniture, away from cars and pedestrians.
    points = [
        ("left window top", 84, 39),
        ("traffic light housing top", 156, 43),
        ("traffic light housing bottom", 155, 54),
    ]
    rows = []
    fig, axes = plt.subplots(len(points), 3, figsize=(12, 3 * len(points)))
    for j, (label, x, y) in enumerate(points):
        r = 9
        s = 18
        patch = gray[0][y - r : y + r + 1, x - r : x + r + 1]
        row = {"label": label, "target": [x, y], "contexts": []}
        for i in range(3):
            if i == 0:
                xx, yy, score = x, y, 1.0
            else:
                roi = gray[i][y - r - s : y + r + s + 1, x - r - s : x + r + s + 1]
                corr = cv2.matchTemplate(roi, patch, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(corr)
                xx = x - s + loc[0]
                yy = y - s + loc[1]
                # Fractional displacement from a quadratic fit around the peak.
                px, py = loc
                sub = []
                for a, b, c in [
                    (
                        corr[py, max(px - 1, 0)],
                        corr[py, px],
                        corr[py, min(px + 1, corr.shape[1] - 1)],
                    ),
                    (
                        corr[max(py - 1, 0), px],
                        corr[py, px],
                        corr[min(py + 1, corr.shape[0] - 1), px],
                    ),
                ]:
                    sub.append(
                        float(0.5 * (a - c) / (a - 2 * b + c))
                        if abs(a - 2 * b + c) > 1e-6
                        else 0.0
                    )
                row["contexts"].append(
                    {
                        "view": i - 1,
                        "point": [xx + sub[0], yy + sub[1]],
                        "delta": [xx + sub[0] - x, yy + sub[1] - y],
                        "correlation": score,
                    }
                )
            ax = axes[j, i]
            ax.imshow(cv2.cvtColor(ims[i], cv2.COLOR_BGR2RGB))
            ax.set_xlim(x - 30, x + 30)
            ax.set_ylim(y + 25, y - 25)
            ax.plot(x, y, "+", color="cyan", ms=12)
            ax.plot(xx, yy, "+", color="red", ms=12)
            ax.set_title(
                ["target", "context 0", "context 1"][i]
                + f" ({xx-x:+d}, {yy-y:+d}), NCC {score:.2f}"
            )
            if i == 0:
                ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        rows.append(row)
    fig.tight_layout()
    fig.savefig(out / "landmark_crops.png", dpi=130)
    plt.close(fig)
    (out / "landmarks.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    for i, ax in enumerate(axes):
        ax.imshow(cv2.cvtColor(ims[i], cv2.COLOR_BGR2RGB))
        for j, row in enumerate(rows):
            x, y = row["target"]
            xx, yy = (x, y) if i == 0 else row["contexts"][i - 1]["point"]
            ax.plot(x, y, "+", color="cyan", ms=12)
            ax.plot(xx, yy, "+", color="red", ms=12)
            ax.annotate(
                str(j + 1),
                (xx, yy),
                xytext=(8, 8),
                textcoords="offset points",
                color="yellow",
                weight="bold",
            )
            if i:
                ax.annotate(
                    "",
                    xy=(xx, yy),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="yellow"),
                )
        ax.set_title(
            ["target", "context 0 (t-1)", "context 1 (t+1)"][i]
            + " | cyan: target position; red: matched position"
        )
    fig.tight_layout()
    fig.savefig(out / "annotated_overview.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
