# AAA Dataset Toolkit (GND-based)

## About the AAA Dataset

**AAA** is an open-source dataset for vision-and-language research and accessibility. It provides four **Angles of View (AoV)** to enable controlled evaluation across camera geometries: **70°**, **110°**, **180° (fisheye)**, and **360° (equirectangular)**.

To support assistive applications for **visually impaired** users, AAA also includes a targeted **Q&A** benchmark organized into four categories:
- **Object** — identify, locate, or describe items.
- **Safety** — detect obstacles, or risky conditions.
- **Navigation** —  directions, and landmarks for wayfinding.
- **Surrounding** — scene context, and overall environment.

Together, these components enable fair comparison of models across AoVs while evaluating practical, assistive questions that matter in real-world settings.

We do not redistribute GND. Please download from the official site:
[**GND dataset (George Mason University)**](https://cs.gmu.edu/~xiao/Research/GND/).


## Quick Start

```bash
# Step A: build views (360 / 70 / 110 / 180)
python make_views_min.py <INPUT_ROOT> <OUTPUT_ROOT>

# Step B: run Mask2Former in place
python mask2former_build_masks.py <OUTPUT_ROOT>/frames --model facebook/mask2former-swin-large-coco-panoptic --batch 4
