# Spatio-Semantic Feature Extraction from Cookie Theft Descriptions (Refined Pipeline)

Extracts graph-theoretic spatio-semantic features from picture description transcripts of the Cookie Theft stimulus (Boston Diagnostic Aphasia Examination). Used for dementia detection research.

This is an updated version of the ICASSP 2023 pipeline with a refined keyword list, post-hoc keyword corrections, and automatic punctuation restoration for RTF transcripts.

## Overview

Each transcript is processed by matching spoken words against keyword lists for 23 predefined semantic units in the Cookie Theft image. Matched units are mapped to their pixel coordinates in the image, forming a directed multigraph of sequential unit transitions. Graph-theoretic features computed from this graph serve as the feature set.

## Citation

If you use this pipeline, please cite:

```
Ambadi PS, Basche K, Koscik RL, Berisha V, Liss JM and Mueller KD (2021) Spatio-Semantic Graphs From Picture Description: Applications to Detection of Cognitive Impairment. Front. Neurol. 12:795374. doi: 10.3389/fneur.2021.795374

Ng, S. I., Ambadi, P. S., Mueller, K. D., Liss, J., & Berisha, V. (2025). Automated extraction of spatio-semantic graphs for identifying cognitive impairment. In Proc. ICASSP (pp. 1-5). IEEE.
```

The ICASSP pipeline this repository extends is frozen at the [`icassp2025`](https://github.com/pranavambadi/spatio-semantic-feat-extraction/releases/tag/icassp2025) tag, and also lives standalone at [spatio-semantic-feat-extraction-icassp](https://github.com/pranavambadi/spatio-semantic-feat-extraction-icassp).

## Differences from the ICASSP Pipeline

**Sentence segmentation for RTF files** uses a deep-learning punctuation restoration model (`deepmultilingualpunctuation`) rather than treating the whole document as one block. CHAT `.cha` files continue to use `*PAR:` utterance boundaries.

**Post-hoc keyword corrections** applied to every sentence:

| Correction | Rule |
|---|---|
| `cookie jar` | Counts jar (unit 7) only; drops the spurious cookie (unit 6) match |
| `fall` | Unit 18 (boy/stool falling) retained only when immediately preceded by boy or girl |
| `reach` | Reassigned from girl-gesture (unit 21) to boy-taking (unit 17) when boy is present |
| Units 22/23 | Unit 22 retained only with sink/water/overflow present; unit 23 only with boy/girl/cookie/stool/falling present |

**Refined keyword list** reduces cross-unit overlap:
- `"dish"` removed from unit 10 (plate) — overlaps unit 15 (dishes)
- `"floor"` removed from unit 12 (water) — causes spurious matches
- `"plate"` removed from unit 15 (dishes) — overlaps unit 10 (plate)
- `"spillage"` added to unit 20 (water overflowing)
- `"back"` / `"behind"` removed from unit 23 — too generic

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
- **Plain-text `.rtf`**: RTF markup is stripped via `striprtf`; sentence boundaries are restored by `deepmultilingualpunctuation` before keyword matching.

Both formats can be mixed in the same run.

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
deepmultilingualpunctuation  # only required when processing .rtf files
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
