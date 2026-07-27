# behavior-tools / behavior-lab Boundary

`behavior-tools` remains the utility layer for collection, splitting,
curation, annotation, and super-resolution. Reusable behavior-analysis APIs live
in `behavior-lab` to avoid duplicate feature and clustering implementations.

## Ownership

| Concern | Repository | Notes |
|---|---|---|
| Multi-view video splitting, frame extraction | `behavior-tools` | Feed extracted videos/frames into DLC/SLEAP or benchmark scripts. |
| Image/video curation and filtering | `behavior-tools` | CLIP/DINO filtering remains here. |
| Keypoint-guided SAM2 mask annotation | `sdannce-poc` | `segmentation/kp_sam2.py` + `viewers/mask_annotator.py`. 260727: the SAM1 module here was deleted — unused, `segment-anything` not installed anywhere, superseded. |
| Pose loaders and canonical `(T,K,D)` sequences | `behavior-lab` | Includes CalMS21, MABe22, SUBTLE, Shank3KO, Rat7M, SLEAP. **SSOT entry point**: `behavior-lab/docs/conventions.md` — kept in sync with the code by `tests/test_conventions_doc.py`. Everything else is a pointer. |
| Feature extraction | `behavior-lab` | raw, kinematic, dyadic, B-SOiD features, Morlet/SUBTLE style. |
| Unsupervised behavior discovery | `behavior-lab` | B-SOiD, SUBTLE, keypoint-MoSeq, hBehaveMAE, PCA/UMAP/KMeans. |
| Motif/syllable transition analysis | `behavior-lab` | Metrics and visualization modules. |

## Handoff Pattern

```python
# behavior-tools output
video_paths = ["cam0.mp4", "cam1.mp4"]

# pose estimation happens through DLC/SLEAP project tooling
# exported predictions are loaded in behavior-lab
from behavior_lab.pose import load_sleap_file

seq = load_sleap_file("predictions.analysis.h5").sequences[0]
```

Main manual: `../../behavior-lab/docs/behavior_analysis_workbench.md`.
Project PRD: `../../behavior-lab/docs/behavior_analysis_prd.md`.
