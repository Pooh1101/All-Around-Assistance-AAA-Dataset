# All Around Assistance (AAA) Dataset

## About the AAA Dataset

**AAA** is an open-source dataset for vision-and-language research and accessibility. It provides four **Angles of View (AoV)** to enable controlled evaluation across camera geometries: **70°**, **110°**, **180° (fisheye)**, and **360° (equirectangular)**.

To support assistive applications for **visually impaired** users, AAA also includes a targeted **Q&A** benchmark organized into four categories:
- **Object** — identify, locate, or describe items.
- **Safety** — detect obstacles, or risky conditions.
- **Navigation** —  directions, and landmarks for wayfinding.
- **Surrounding** — scene context, and overall environment.

Together, these components enable fair comparison of models across AoVs while evaluating practical, assistive questions that matter in real-world settings. 

We do not redistribute GND. Please download from the official site:
[**GND dataset**](https://cs.gmu.edu/~xiao/Research/GND/).


## Quick Start

```bash
git clone https://github.com/Pooh1101/All-Around-Assistance-AAA-Dataset.git
cd All-Around-Assistance-AAA-Dataset
```

### Download [**GND Dataset**](https://github.com/jingGM/GND)

Ensure the structure is as follows:

```
frames_folder_root/
├── AU/
│   ├── S01/
│   │   ├── frame0000
│   │   ├── frame0001
│   │   └── ...
│   ├── S02/
│   │   └── ...
│   └── ...
│
└── CUA/
    └── ... 
```

```bash

# build views (70 / 110 / 180 / 360)
python make_views_min.py </path/to/frames_folder_root> </path/to/output_folder>

#run Mask2Former
python mask2former_build_masks.py </path/to/output_folder>/frames --model facebook/mask2former-swin-large-coco-panoptic
