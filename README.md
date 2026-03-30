# Author Handwriting Recognition

Deep learning pipeline for **writer identification** from handwritten documents, using Siamese Networks with Triplet Loss for metric learning.

The system learns an embedding space where patches from the same author cluster together and patches from different authors are pushed apart, enabling identification of both seen and unseen writers.

---

## Results

### MobileNetV2 + Triplet Loss — 204 Authors (seen writers)

| Strategy   | Aggregation | Accuracy |
|------------|-------------|----------|
| 1-NN       | vote        | ~100%    |
| Top-K      | vote        | ~99.5%   |
| Centroid   | vote        | ~90.2%   |
| Mean       | vote        | ~57.8%   |

Embedding quality (cosine distance):
- Intra-author distance: **0.354**
- Inter-author distance: **0.974**

---

## Architecture

```
src/
├── datasets/
│   ├── builders/          # AuthorDatasetBuilder, DatasetBuilder
│   └── loaders/           # SelectiveDataLoader, ZarrLoader, AuthorMetadataLoader
├── generators/
│   ├── base_generator.py
│   ├── data_generator.py
│   ├── pair_generator.py
│   ├── triplet_generator.py
│   └── augmentation.py
├── models/
│   ├── backbones/         # MobileNetV2, EfficientNet, DeepCNN
│   ├── losses/            # TripletLoss, ContrastiveLoss, BinaryDistance
│   ├── layers.py
│   ├── embedding_network.py
│   ├── siamese_pair_model.py
│   ├── siamese_triplet_model.py
│   └── siamese_factory.py
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
│   │   ├── agregators.py      # PatchAggregator, VoteAggregator, ScoreAggregator
│   │   └── strategy_pipeline.py
│   ├── embedding_evaluator.py
│   ├── embedding_visualizer.py
│   ├── triplet_evaluator.py
│   └── pair_evaluator.py
├── io/
│   ├── file_system.py
│   ├── image_ops.py
│   ├── image_cache.py
│   ├── logging/
│   └── serializers/       # zarr, pickle, image, npy, mat, keras, json
└── processors/
    ├── author_data_loader.py
    ├── author_processor.py
    └── line_segment_processor.py
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

| Folder                    | Description                        |
|---------------------------|------------------------------------|
| `1_ImagesRotated`         | Deskewed original                  |
| `2_ImagesMedianBW`        | Median-filtered binary             |
| `3_ImagesLinesRemovedBW`  | Lines removed, binary              |
| `4_ImagesLinesRemoved`    | Lines removed, grayscale           |

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
└── attrs         # metadata: num_authors, patch_shape, strides, seed, created_at
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
from src.models.backbones import MobileNetBackbone
from src.training import ModelTrainerFactory

train_gen = TripletGenerator(loader=split.train, batch_size=32)
val_gen   = TripletGenerator(loader=split.val,   batch_size=32)

trainer = ModelTrainerFactory.create(
    backbone=MobileNetBackbone(),
    input_shape=(128, 128, 1),
    num_authors=204,
    train_generator=train_gen,
    val_generator=val_gen,
    model_type="triplet",
)

trainer.initialize_model()
trainer.compile(optimizer=Adam(1e-4), loss=TripletLoss(margin=0.2), metrics=[...])
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
