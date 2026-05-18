# Spatio-Semantic Feature Extraction from Cookie Theft Descriptions

Extracts graph-theoretic spatio-semantic features from picture description transcripts of the Cookie Theft stimulus (Boston Diagnostic Aphasia Examination). Used for dementia detection research.

## Overview

Each transcript is processed by matching spoken words against keyword lists for 23 predefined semantic units in the Cookie Theft image. Matched units are mapped to their pixel coordinates in the image, forming a directed multigraph of sequential unit transitions. Graph-theoretic features computed from this graph serve as the feature set.

This is the pipeline as described in the ICASSP 2023 paper.

## Semantic Units

| Category | Units |
|---|---|
| Subjects (1–3) | boy, girl, woman |
| Places (4–5) | kitchen, outside |
| Objects (6–16) | cookie, jar, stool, sink, plate, dishcloth, water, window, cupboard, dishes, curtains |
| Actions (17–23) | boy taking/stealing, boy or stool falling, woman drying/washing plates, water overflowing, action performed by girl, woman unconcerned by overflowing, woman indifferent to the children |

## Output Features

| Feature | Description |
|---|---|
| `CIU_seq` | Ordered list of matched unit indices |
| `total_path_distance` / `sum_of_edges` | Total Euclidean distance traversed through unit space |
| `unique_nodes` | Number of distinct semantic units mentioned |
| `nodes` | Total unit mentions (including repeats) |
| `self_cycles` | Transitions where the same unit appears consecutively |
| `sum_of_edges/unique_nodes` | Path efficiency normalized by unique units |
| `sum_of_edges/nodes` | Path efficiency normalized by total mentions |
| `cycles/unique nodes` | Redundancy normalized by unique units |
| `cycles/nodes` | Redundancy normalized by total mentions |
| `avg_x`, `avg_y` | Spatial centroid of mentioned units |
| `std_x`, `std_y` | Spatial spread of mentioned units |
| `self_cycles_quadrants` | Self-cycles in the 4-quadrant graph |
| `cross_ratio_quadrants` | Cross-quadrant transitions / self-cycles ratio |

For CHAT `.cha` files, participant metadata is also extracted: `mmse`, `sex`, `age`.

## Supported Transcript Formats

- **CHAT `.cha`** (DementiaBank format): `*PAR:` utterances are parsed sentence-by-sentence; participant metadata is read from the `@ID` header.
- **Plain-text `.rtf`**: RTF markup is stripped via `striprtf`; the entire document is treated as one block.

## Requirements

```
regex
numpy
pandas
scikit-learn
tqdm
networkx
matplotlib
scipy
spacy
striprtf
```

Install spaCy's English model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Place your transcript files in a directory (e.g., `./WRAP` for `.cha` files or `./transcripts` for `.rtf` files).
2. Edit `extract_feat.py` to set `datasets` and `results_dir`.
3. Run:

```bash
python extract_feat.py
```

Output is written to `spatio_semantic_features.csv`.

## Output Directory Structure

```
results/
  dataframes/     # per-transcript .pkl files with unit-level data
  plots/          # unit-level transition graph plots (23 nodes)
  quadrant_plots/ # quadrant-level transition graph plots (4 nodes)
```
