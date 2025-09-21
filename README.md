# AAA Dataset Toolkit (GND-based)

Convert local **GND** equirect frames into multi-FoV views (**360 / 70 / 110 / 180**) and run **Mask2Former** to write per-frame bitmaps (`obj_*.npz`, `reg_*.npz`) in place.

Download GND yourself and point scripts to your local paths. 

## Quick Start

python make_views_min.py <INPUT_ROOT> <OUTPUT_ROOT>

python mask2former_build_masks.py <OUTPUT_ROOT>/frames \
  --model facebook/mask2former-swin-large-coco-panoptic --batch 4