# Author Handwriting Recognition

This project addresses **writer identification** from handwritten documents using deep metric learning. A Siamese network with triplet loss is trained to learn an embedding space where patches from the same author are close together and patches from different authors are pushed apart.

The system is evaluated under two conditions:

- **Closed-set**: authors present during training
- **Open-set**: completely new identities never seen during training

Identification is performed by extracting embeddings from multiple patches of a document and aggregating patch-level predictions into a single document-level decision.

---

## Results

All experiments use dataset variant 4 (lines removed, grayscale), patch size `(180, 360)`.

### Closed-set Evaluation — 204 seen authors (authors 0–203)

Models trained on authors 0–203 and evaluated on a spatially separated test split from the same set.

#### MobileNetV2 + Triplet Loss

| Strategy | Aggregation | Accuracy |
| -------- | ----------- | -------- |
| 1-NN     | vote        | 100.00%  |
| 1-NN     | score       | 100.00%  |
| Top-K    | vote        | 99.51%   |
| Centroid | vote        | 91.18%   |
| Mean     | vote        | 58.33%   |

Embedding quality (cosine distance): Intra **0.4172** · Inter **0.9766** · Ratio **2.34**

Triplet evaluation: AUC **0.930** · Triplet accuracy **93.24%** · EER **0.1432**

#### EfficientNetV2B2 + Triplet Loss

| Strategy | Aggregation | Accuracy |
| -------- | ----------- | -------- |
| 1-NN     | vote        | 100.00%  |
| 1-NN     | score       | 100.00%  |
| Top-K    | vote        | 99.51%   |
| Centroid | vote        | 96.57%   |
| Mean     | vote        | 69.12%   |

Embedding quality (cosine distance): Intra **0.357** · Inter **0.994** · Ratio **2.78**

Triplet evaluation: AUC **0.955** · Triplet accuracy **95.19%** · EER **0.1126**

---

### Model Comparison — 100 Authors

Direct comparison of MobileNetV2, EfficientNetV2B2, and ViT-B/16 trained on 100 authors. ViT uses input shape `(224, 224)` with `resize_with_pad`; the other two use `(120, 240)`.

#### Closed-set — 100 seen authors

| Model | Strategy | Vote acc | Score acc | Intra | Inter |
|-------|----------|----------|-----------|-------|-------|
| MobileNetV2 | 1-NN | 100.00% | 100.00% | 0.365 | 0.996 |
| MobileNetV2 | Top-K | 100.00% | 100.00% | | |
| MobileNetV2 | Centroid | 97.00% | 97.00% | | |
| MobileNetV2 | Mean | 72.00% | 70.00% | | |
| EfficientNetV2B2 | 1-NN | 100.00% | 100.00% | **0.276** | **1.002** |
| EfficientNetV2B2 | Top-K | 100.00% | 100.00% | | |
| EfficientNetV2B2 | Centroid | **99.00%** | **99.00%** | | |
| EfficientNetV2B2 | Mean | **78.00%** | **78.00%** | | |
| ViT-B/16 | 1-NN | 100.00% | 100.00% | 0.325 | 0.981 |
| ViT-B/16 | Top-K | 100.00% | 100.00% | | |
| ViT-B/16 | Centroid | 98.00% | 98.00% | | |
| ViT-B/16 | Mean | 72.00% | 70.00% | | |

#### Open-set — 100 unseen authors (authors 100–199)

| Model | Strategy | Vote acc | Score acc | Intra | Inter |
|-------|----------|----------|-----------|-------|-------|
| MobileNetV2 | 1-NN | 100.00% | 100.00% | 0.439 | 0.935 |
| MobileNetV2 | Top-K | 100.00% | 100.00% | | |
| MobileNetV2 | Centroid | 86.87% | 86.87% | | |
| MobileNetV2 | Mean | 47.47% | 44.44% | | |
| EfficientNetV2B2 | 1-NN | 100.00% | 100.00% | 0.351 | 0.935 |
| EfficientNetV2B2 | Top-K | 100.00% | 100.00% | | |
| EfficientNetV2B2 | Centroid | 90.91% | 90.91% | | |
| EfficientNetV2B2 | Mean | **61.62%** | **61.62%** | | |
| ViT-B/16 | 1-NN | 100.00% | 100.00% | **0.325** | 0.883 |
| ViT-B/16 | Top-K | 100.00% | 100.00% | | |
| ViT-B/16 | Centroid | **91.92%** | **91.92%** | | |
| ViT-B/16 | Mean | 56.57% | 55.56% | | |

#### Key observations

- **1-NN and Top-K** reach 100% vote and score on both seen and unseen authors across all three models — the primary identification strategies are fully reliable at this scale.
- **EfficientNetV2B2** forms the most compact embedding space on seen authors (intra 0.276), which explains its centroid and mean advantage on the closed-set.
- **ViT-B/16** achieves the lowest intra distance on **unseen** authors (0.325 vs 0.351 for EfficientNet), indicating stronger generalization of its embedding space to new identities.
- **Mean strategy** remains the weakest across all models, confirming that the embedding space is not unimodal per author — this is a structural property of handwriting, not a model limitation.
- ViT triplet evaluation: AUC **0.9372** · Triplet accuracy **93.66%** · EER **0.1386**

---

### Open-set Evaluation — 203 unseen authors (authors 204–406)

Same trained models evaluated on authors never seen during training. Validates that the learned embedding space generalizes to new identities.

#### MobileNetV2 + Triplet Loss

| Strategy | Aggregation | Accuracy |
| -------- | ----------- | -------- |
| 1-NN     | vote        | 100.00%  |
| 1-NN     | score       | 100.00%  |
| Top-K    | vote        | 99.51%   |
| Centroid | vote        | 87.68%   |
| Mean     | vote        | 55.17%   |

Embedding quality (cosine distance): Intra **0.364** · Inter **0.949** · Ratio **2.61**

#### EfficientNetV2B2 + Triplet Loss

| Strategy | Aggregation | Accuracy |
| -------- | ----------- | -------- |
| 1-NN     | vote        | 100.00%  |
| 1-NN     | score       | 100.00%  |
| Top-K    | score       | 99.01%   |
| Centroid | vote        | 91.13%   |
| Mean     | vote        | 59.11%   |

Embedding quality (cosine distance): Intra **0.295** · Inter **0.972** · Ratio **3.29**

---

## Analysis

### Aggregation strategies

Patch-level predictions are combined into a document-level decision using one of four strategies, each paired with a patch, vote, or score aggregator:

| Strategy     | Behavior                                                                                                                                 |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **1-NN**     | Best overall. Assigns each patch to its nearest neighbor in the embedding space. Robust because it requires no global structure.         |
| **Top-K**    | Very strong. Averages the K nearest neighbors, reducing noise from isolated patches.                                                     |
| **Centroid** | Good on seen authors, degrades on unseen. Depends on a representative centroid per author — sensitive to embedding distribution quality. |
| **Mean**     | Poor performance (~55–69% vote accuracy). See below.                                                                                     |

### Patch accuracy vs document accuracy

Patch-level accuracy is moderate (64–75%), but document-level accuracy reaches near-perfect values through aggregation over ~44 patches per author.

A binomial approximation illustrates the theoretical lower bound: with n=44 and p≈0.75, the probability of majority-vote failure is ~0.003%. However, this model assumes patch independence — which does not hold in practice, since patches from the same document share author, instrument, and writing session. Errors tend to be correlated, not independent.

The stronger explanation is the quality of the embedding space itself: inter/intra ratio of 2.3–2.8 means most patches from a document are already closer to the correct author than to any competitor. Aggregation resolves residual ambiguity; it does not compensate for a weak model.

### Why Mean fails

The mean strategy computes a single average embedding per author and predicts based on proximity to that mean. Its poor performance (55–69% vote accuracy across both models) indicates that **the embedding space is not unimodal per author** — patches from the same writer form multiple local clusters rather than a single compact region.

This is expected for handwriting: different letters, strokes, and local structures produce distinct patch appearances that the backbone encodes into distinct regions of the embedding space. The mean embedding falls between clusters and is not representative of any of them.

### Generalization to unseen authors

Both models maintain near-perfect vote accuracy on 203 authors never seen during training. The drop in patch accuracy (~5pp) is expected — the model has not been exposed to these identities.

This result indicates the model has learned **writer-specific features** (stroke patterns, letter proportions, spacing) rather than memorizing identities. The embedding space transfers to new identities without retraining.

One caveat: the two author groups (0–203 and 204–406) were not randomly assigned. Visual inspection suggests the second group may contain more stylistically distinct writers, which could partially explain the higher inter/intra ratio (3.29 vs 2.78) in the open-set evaluation. This remains an open question that would require controlled analysis to confirm.

### Why Centroid degrades on open-set

Centroid accuracy drops more than 1-NN across both models (closed → open: 96.6% → 91.1% for EfficientNet, 91.2% → 87.7% for MobileNet). Unlike 1-NN, centroid requires a global representative point per author. For unseen authors, the centroid is computed directly from test patches — fewer samples under potentially higher intra-author variability leads to a less stable representative point.

---

## Conclusion

Siamese networks with triplet loss learn embedding spaces that are both discriminative and generalizable. Key findings:

- Moderate patch accuracy is sufficient for near-perfect document classification when aggregation is applied
- Embedding geometry — not the ensemble effect — is the dominant factor behind aggregation performance
- The learned representation transfers to unseen identities, confirming metric learning over identity memorization
- EfficientNetV2B2 produces consistently better-separated embeddings than MobileNetV2 across all metrics

---

## Architecture

```
.
├── config/                        # YAML experiment configs (authors, model, training)
│   ├── test_5_4.yaml
│   ├── test_10_4.yaml
│   ├── test_40_4.yaml
│   ├── test_100_4.yaml
│   └── test_204_4.yaml
├── notebooks/                     # One notebook per experiment (backbone × author count)
│   ├── generate_dataset.ipynb
│   ├── mobile_test_5_4.ipynb
│   ├── mobile_test_10_4.ipynb
│   ├── mobile_test_20_4.ipynb
│   ├── mobile_test_40_4.ipynb
│   ├── mobile_test_100_4.ipynb
│   ├── mobile_test_204_4.ipynb
│   ├── efficientnet_test_5_4.ipynb
│   ├── efficientnet_test_10_4.ipynb
│   ├── efficientnet_test_40_4.ipynb
│   ├── efficientnet_test_100_4.ipynb
│   ├── efficientnet_test_204_4.ipynb
│   └── vit_test_100_4.ipynb
└── src/
    ├── datasets/
    │   ├── builders/              # AuthorDatasetBuilder, DatasetBuilder
    │   └── loaders/               # SelectiveDataLoader, ZarrLoader, AuthorMetadataLoader
    ├── generators/
    │   ├── base_generator.py
    │   ├── data_generator.py
    │   ├── pair_generator.py
    │   ├── triplet_generator.py
    │   └── augmentation.py
    ├── models/
    │   ├── backbones/             # MobileNetV2, EfficientNetV2, ViT-B/16, DeepCNN
    │   ├── losses/                # TripletLoss, ContrastiveLoss
    │   ├── layers.py              # CosineDistance, CosineSimilarity, L2Normalization
    │   ├── siamese/
    │   │   ├── base_siamese_builder.py   # abstract builder, owns embedding_model
    │   │   ├── siamese_pair_builder.py
    │   │   ├── siamese_triplet_builder.py
    │   │   └── siamese_factory.py        # registry: "pair" | "triplet"
    │   └── embedding_network.py
    ├── training/
    │   ├── base_metric_trainer.py
    │   ├── pair_trainer.py
    │   ├── triplet_trainer.py
    │   ├── trainer_factory.py
    │   └── training_callbacks.py
    ├── evaluator/
    │   ├── embedding_utils/
    │   │   ├── embedding_extractor.py
    │   │   ├── embedding_geometry.py
    │   │   └── umap_reducer.py
    │   ├── embedding_strategy/
    │   │   ├── base_strategy.py
    │   │   ├── one_nn_strategy.py
    │   │   ├── centroid_strategy.py
    │   │   ├── mean_strategy.py
    │   │   ├── topk_strategy.py
    │   │   ├── aggregators.py      # PatchAggregator, VoteAggregator, ScoreAggregator
    │   │   └── strategy_pipeline.py
    │   ├── embedding_evaluator.py
    │   ├── embedding_visualizer.py
    │   ├── triplet_evaluator.py
    │   └── pair_evaluator.py
    └── io/
        ├── file_system.py
        ├── image_ops.py
        ├── exceptions.py
        ├── logging/
        └── serializers/           # zarr, pickle, image, npy, mat, keras, json
```

### Design Principles

- **Clean Architecture** — each layer has a single direction of dependency
- **Strategy Pattern** — identification strategies (`1nn`, `centroid`, `mean`, `topk`) are interchangeable and compose with aggregators (`patch`, `vote`, `score`)
- **Template Method** — `BaseMetricTrainer` defines the training contract; `PairTrainer` and `TripletTrainer` implement loss-specific steps
- **Factory + Registry** — `ModelTrainerFactory` with `TrainerSpec` validates generator types at construction time

---

## Dataset Generation

The raw dataset consists of 407 handwritten document images (one per author), each with associated `.mat` metadata files containing line segmentation coordinates and a designated test area.

### Image variants

Each author image is processed in 4 preprocessing variants:

| Folder                   | Description              |
| ------------------------ | ------------------------ |
| `1_ImagesRotated`        | Deskewed original        |
| `2_ImagesMedianBW`       | Median-filtered binary   |
| `3_ImagesLinesRemovedBW` | Lines removed, binary    |
| `4_ImagesLinesRemoved`   | Lines removed, grayscale |

### Patch extraction

Each image is sliced into overlapping patches using a sliding window. The test area is spatially separated from training data — no patch crosses the boundary.

```
patch_size : (180, 360)  # height × width in pixels
strides    : (30, 90)
```

The `empty_threshold` parameter filters out patches that are predominantly white (background), per folder:

```python
thresholds = {
    "1_ImagesRotated":        0.95,
    "2_ImagesMedianBW":       0.95,
    "3_ImagesLinesRemovedBW": 0.97,
    "4_ImagesLinesRemoved":   0.97,
}
```

### Why Zarr

The full dataset for 407 authors is ~275,000 patches at `(180, 360)` pixels each:

| Representation            | Size     |
| ------------------------- | -------- |
| uint8 grayscale (on disk) | ~16.6 GB |
| float32 RGB (model input) | ~199 GB  |

Loading the entire dataset into RAM before training is not feasible, especially on Google Colab (~12 GB limit). Zarr solves this with **chunked, lazy storage**: data lives on disk and only the requested chunks are materialized in memory.

During training, each batch of 64 triplets reads 192 patches — roughly **142 MB** of float32 data. RAM usage stays proportional to batch size regardless of how large the full dataset grows.

#### How generators interact with zarr during `fit`

`SelectiveDataLoader` opens the zarr archive in read mode and loads only two things eagerly into RAM: the label arrays (integers, negligible size). The image arrays (`images_non_test`, `images_test`) remain as lazy `zarr.Array` references — no pixel data is read at this point.

The `tf.data` pipeline built by `TripletGenerator` has a deliberate stage ordering:

```
indices (in-memory tensors)
  → _make_triplet()       # selects anchor/positive/negative indices — no I/O
  → .batch(batch_size)    # groups 64 triplets = 192 indices
  → _load_triplet_images()  # HERE: zarr reads happen, 192 patches per call
  → augment()             # CPU augmentation on loaded float32 tensors
  → .prefetch(AUTOTUNE)   # next batch is read while GPU trains on current
```

The zarr read is deferred to `_load_triplet_images`, which calls `ZarrLoader.get_images()` via `tf.numpy_function`. This means the expensive I/O only happens after batching — 192 contiguous reads per step instead of one-by-one. Prefetch then overlaps the next zarr read with the current GPU forward/backward pass, hiding I/O latency.

Index arrays are also sorted before being handed to the loaders (`np.sort` in `SelectiveDataLoader.build`). Sorted access improves zarr chunk locality: sequential indices are more likely to fall within the same chunk, reducing the number of distinct chunk reads per batch.

This design also makes selective experiments straightforward: `SelectiveDataLoader` filters authors by index range without touching unneeded chunks.

### Output format

Each variant produces an independent `.zarr` archive with the following structure:

```
dataset_<variant>_<timestamp>.zarr
├── test/
│   ├── images    # patches from the designated test area
│   └── labels    # integer author ids
├── non_test/
│   ├── images    # patches from the remaining area (train + val)
│   └── labels
└── attrs         # metadata: num_authors, patch_shape, strides, seed, created_at, author_names
```

Author label assignment is deterministic: authors are sorted by `natural_key` (numeric-aware string sort) before enumeration, ensuring consistent ids across runs.

Generation runs on Google Colab and takes ~18 minutes per variant for 407 authors. Output files are archived to `.tar.gz` and saved to Google Drive.

### Running generation

See `notebooks/generate_dataset.ipynb`. Key configuration:

```python
config = Config(
    data_source=DataSourceConfig(
        base_path=Path("handwriting data/"),
        folders=["1_ImagesRotated", "2_ImagesMedianBW",
                 "3_ImagesLinesRemovedBW", "4_ImagesLinesRemoved"],
        mat_path=Path("handwriting data/5_DataDarkLines"),
        thresholds={...},
    ),
    patch=PatchConfig(
        patch_size=(180, 360),
        strides=(30, 90),
    ),
)
```

---

## Pipeline Overview

### 1. Load generated dataset

```python
from src.datasets import SelectiveDataLoader

loader = SelectiveDataLoader(zarr_path="dataset_2_ImagesMedianBW.zarr")
split = loader.load(author_ids=range(204))  # select subset of authors
```

### 2. Training

```python
from src.generators import TripletGenerator
from src.models import MobileNetV2Backbone, EmbeddingNetwork, SiameseFactory, TripletLoss
from src.training import ModelTrainerFactory, get_default_callbacks

train_gen = TripletGenerator(split=train, batch_size=64, seed=42, augment=augment_train)
val_gen   = TripletGenerator(split=val,   batch_size=32, seed=42, augment=augment)

embedding_network = EmbeddingNetwork(
    backbone=MobileNetV2Backbone(freeze_base=False, fine_tune_at=100),
    input_shape=(120, 240, 3),
    num_authors=204,
)

siamese_builder = SiameseFactory.create(
    model_type="triplet",
    embedding_network=embedding_network,
    input_shape=(120, 240, 3),
)

trainer = ModelTrainerFactory.create(
    siamese_builder=siamese_builder,
    train_generator=train_gen,
    val_generator=val_gen,
    model_type="triplet",
)

trainer.initialize_model()
trainer.compile(optimizer=Adam(1e-5), loss=TripletLoss(margin=0.4), metrics=[...])
trainer.train_model(epochs=50, callbacks=get_default_callbacks(...))
```

### 3. Evaluation

```python
from src.evaluator import EmbeddingEvaluator
from src.evaluator.embedding_utils import EmbeddingExtractor, EmbeddingGeometry
from src.evaluator.embedding_strategy import (
    StrategyPipeline, OneNNStrategy, CentroidStrategy, TopKStrategy,
    PatchAggregator, VoteAggregator, ScoreAggregator,
)

extractor = EmbeddingExtractor(model=trainer.embedding_model, logger=logger)
pipeline  = StrategyPipeline(
    strategies=[OneNNStrategy(), CentroidStrategy(), TopKStrategy(k=5)],
    aggregators=[PatchAggregator(), VoteAggregator(), ScoreAggregator()],
    logger=logger,
)

evaluator = EmbeddingEvaluator(extractor=extractor, pipeline=pipeline, logger=logger)
report = evaluator.evaluate(test_generator)
# report.strategy_results["1nn"]["vote"].metrics["accuracy"]
```

---

## Technologies

- **Deep Learning**: TensorFlow / Keras
- **Data Storage**: zarr
- **Data Processing**: NumPy, Pandas, OpenCV
- **Visualization**: UMAP, Matplotlib, Seaborn
- **ML Tools**: Scikit-learn

---

## Dataset

The handwriting images used in this project are the property of the **Israel Police Document Laboratory**.
They are not included in this repository and are not publicly distributed.
This project is for educational and research purposes only.

---

## License

Code released under the MIT License (see the LICENSE file for details).
The dataset remains subject to its own copyright and usage terms.
