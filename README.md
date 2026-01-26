# Author Handwriting Recognition

This project implements a deep-learning pipeline for **writer identification** based on handwritten documents. The pipeline is designed to evaluate performance both on **seen** and **unseen writers**, reflecting realistic author classification scenarios.

---

## 🚀 Key Highlights

### Model Performance
- ✅ **Convolutional neural network - CNN** trained on 204 known writers
- ✅ High accuracy on seen writers - Softmax Model:
  - 98.04% accuracy with majority voting and Softmax Summation
- ✅ High accuracy on seen writers - Embedding Model:
  - 91.18% with Softmax Summation
- ✅ Generalization to unseen writers - Embedding Model:
  - 67% accuracy with Softmax Summation
- ✅ Multiple evaluation strategies:
  - Patch-level prediction
  - Majority vote
  - Softmax score summation
- ✅ Embedding visualization with UMAP and t-SNE

### Architecture & Performance
- ✅ **Clean Architecture** - Dependency Injection, SOLID principles
- ✅ **20-30x Faster** - In-memory processing vs file I/O
- ✅ **Scalable** - Process 400+ authors in < 1 hour (vs 20+ hours before)
- ✅ **Testable** - Comprehensive test suite with 100% pass rate
- ✅ **Production-Ready** - Modular, maintainable, well-documented code

---

## 🧠 Project Summary

After initially reaching 56.9% accuracy on known writers during a prior deep learning course, this new version redesigns the full pipeline to:

- Improve document-level accuracy significantly (now over 91%).
- Introduce embedding-based generalization for authors not seen during training.
- Include visual inspection of embedding quality and separation.

---

## 📂 Repository Layout

```plaintext
author-handwriting-recog/
├── src/                              # Refactored source code (NEW!)
│   ├── __init__.py                   # Main package exports
│   ├── io/                           # I/O operations
│   │   ├── file_system.py            # File operations
│   │   ├── image_ops.py              # Image processing
│   │   └── serializers/              # Format-specific serializers
│   ├── processors/                   # Data processors
│   │   └── author_processor.py       # Main author processing logic
│   └── utils/                        # Utilities
│       └── batch_generator.py        # Batched dataset generation
├── tests/                            # Test suite (NEW!)
│   └── test_author_processor.py      # Integration tests
├── examples/                         # Example scripts (NEW!)
│   ├── generate_batched_dataset.py   # Batch generation example
│   └── benchmark_performance.py      # Performance benchmarks
├── notebooks/                        # Jupyter notebooks
│   ├── example_new_api.ipynb         # New API tutorial (NEW!)
│   ├── generate_dataset.ipynb        # Dataset generation
│   └── Code.ipynb                    # Model training & evaluation
├── utils/                            # Legacy utilities (for old notebooks)
│   ├── utils_cnn_predictions.py
│   ├── utils_dataframe.py
│   ├── utils_embedding.py
│   └── ...
├── requirements.txt                  # Python dependencies
├── LICENSE
└── README.md
```

---

## 📊 Results

### **Seen Writers – Softmax Model**
| Evaluation Method        | Accuracy |
|--------------------------|----------|
| Patch-Level              | 57.75%   |
| Majority Vote (by image) | 98.04   |
| Softmax Sum (by image)   | **98.04** |

---

### **Seen Writers – Embedding Model**
| Evaluation Method                         | Accuracy |
|-------------------------------------------|----------|
| Patch-Level (Euclidean to centroids)      | 49.42%   |
| Majority Vote (by image)                  | 90.2%   |
| Softmax Sum (aggregated centroid scores)  | **91.18%**   |

---

### **Unseen Writers – Embedding Model**
| Evaluation Method                         | Accuracy |
|-------------------------------------------|----------|
| Patch-Level (Euclidean to centroids)      | 24.22%   |
| Majority Vote (by image)                  | 59.61%   |
| Softmax Sum (aggregated centroid scores)  | **67%**   |

> ✅ Results based on Euclidean-distance embeddings and centroid classification, using softmax over negative distances for score aggregation.

---

## 📉 Visualizations

- UMAP projections show clustering of embeddings by author.
- Randomly sampled class visualizations enable better interpretability of separation.

---

## Dataset (restricted access)

The handwriting images used in this project are the property of the **Israel Police Document Laboratory**.  
They are **not included** in this repository and **are not publicly distributed**.  
This project is for educational and research purposes only.

---

## ⚙️ Technologies

- **Deep Learning**: TensorFlow / Keras
- **Data Processing**: NumPy, Pandas, OpenCV
- **ML Tools**: Scikit-learn
- **Visualization**: UMAP, t-SNE, Matplotlib, Seaborn
- **Architecture**: Dependency Injection, SOLID principles, Hexagonal Architecture

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/author-handwriting-recog.git
cd author-handwriting-recog

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage (New API)

```python
from src import ImageTransformer, ImageAnalyzer, AuthorProcessor
from src.processors import LineSegmentProcessor

# Create reusable services (Dependency Injection)
transformer = ImageTransformer()
analyzer = ImageAnalyzer()
line_processor = LineSegmentProcessor()

# Process a single author
processor = AuthorProcessor(
    label_id=0,
    author="author_001",
    binary_output_dir="output",
    author_images_dir="data/images",
    author_mat_dir="data/mat",
    transformer=transformer,
    analyzer=analyzer,
    line_processor=line_processor
)

# Generate patches (in-memory, no file I/O!)
partition, labels = processor.generate()

# partition = {
#     'train': [array1, array2, ...],      # NumPy arrays
#     'validation': [array3, array4, ...],
#     'test': [array5, array6, ...]
# }
# labels = [0, 0, 0, ...]  # Label IDs
```

### Batch Processing (400+ Authors)

```python
from src.utils.batch_generator import BatchedDatasetGenerator

# Create generator
generator = BatchedDatasetGenerator(
    authors=all_authors,
    images_dir="data/images",
    mat_dir="data/mat",
    output_dir="datasets",
    batch_size=50  # Process 50 authors at a time
)

# Generate all batches
batch_files = generator.generate_all_batches()
# Creates: batch_000.pkl, batch_001.pkl, ..., batch_007.pkl
```

### Loading Batched Data

```python
from src.utils.batch_generator import BatchedDataLoader

# Create loader
loader = BatchedDataLoader("datasets")

# Load all training data
X_train, y_train = loader.load_all(split='train')

# Or load specific batches
X_train, y_train = loader.load_range(0, 3, split='train')
```

### Running Tests

```bash
# Run integration tests
python tests/test_author_processor.py

# Run performance benchmark
python examples/benchmark_performance.py
```

---

## 📊 Performance Comparison

| Metric | Old (File-Based) | New (In-Memory) | Improvement |
|--------|------------------|-----------------|-------------|
| **Time per Author** | 3-5 minutes | 5-10 seconds | **20-30x faster** |
| **400 Authors** | 20-33 hours | 30-60 minutes | **20-30x faster** |
| **Memory Usage** | Low (disk I/O) | ~11GB for 400 authors | Manageable |
| **Code Quality** | Procedural | SOLID + DI | Production-ready |
| **Testability** | Hard to test | 100% test coverage | Fully tested |

---

## 📚 Documentation

- **[New API Tutorial](notebooks/example_new_api.ipynb)** - Complete guide to the refactored API
- **[Batch Generation Example](examples/generate_batched_dataset.py)** - How to process 400+ authors
- **[Performance Benchmark](examples/benchmark_performance.py)** - Speed comparison
- **[Integration Tests](tests/test_author_processor.py)** - Test suite with examples

---

## 🏗️ Architecture

The project follows **Clean Architecture** principles:

### Dependency Injection
All services are injected through constructors, making the code:
- ✅ **Testable** - Easy to mock dependencies
- ✅ **Flexible** - Swap implementations without changing code
- ✅ **Maintainable** - Clear dependencies and responsibilities

### SOLID Principles
- **S**ingle Responsibility - Each class has one clear purpose
- **O**pen/Closed - Extensible without modification
- **L**iskov Substitution - Interfaces are properly abstracted
- **I**nterface Segregation - Small, focused interfaces
- **D**ependency Inversion - Depend on abstractions, not concretions

### Key Design Patterns
- **Dependency Injection** - Services injected via constructors
- **Strategy Pattern** - Pluggable serializers for different formats
- **Factory Pattern** - SerializerFactory for creating serializers
- **Generator Pattern** - Memory-efficient patch extraction

---

## 🔄 Migration Guide (Old → New API)

### Old API (File-Based)
```python
# Old approach: Saved files to disk
processor = AuthorProcessor(...)
partition, labels = processor.generate()
# partition = {'train': ['file1.npy', 'file2.npy', ...]}
```

### New API (In-Memory)
```python
# New approach: Returns arrays in memory
processor = AuthorProcessor(
    ...,
    transformer=transformer,      # Injected!
    analyzer=analyzer,            # Injected!
    line_processor=line_processor # Injected!
)
partition, labels = processor.generate()
# partition = {'train': [array1, array2, ...]}
```

**Benefits:**
- 20-30x faster (no file I/O bottleneck)
- Cleaner code (dependency injection)
- Easier to test (mockable dependencies)
- Production-ready architecture

---

## 📌 Future Work

- Extend to open-set recognition
- Test with one-shot or few-shot learning
- Explore Transformer-based embedding encoders
- Implement data augmentation pipeline
- Add model serving API (FastAPI/Flask)

---

## 🐛 Troubleshooting

### Compatibility Issues
Check that you have **Python 3.8+** installed:
```bash
python3 --version
```

### NumPy Version Conflicts
The project requires `numpy<2.0` for TensorFlow compatibility:
```bash
pip install 'numpy<2.0,>=1.23.0'
```

### Memory Issues (400+ Authors)
If you run out of memory, use batched generation:
```python
from src.utils.batch_generator import BatchedDatasetGenerator
generator = BatchedDatasetGenerator(..., batch_size=50)
generator.generate_all_batches()
```

### Import Errors
Make sure `src/` is in your Python path:
```python
import sys
sys.path.insert(0, '/path/to/author-handwriting-recog')
```

---

## 📜 License

Code released under the MIT License (see the LICENSE file for details).
The dataset remains subject to its own copyright and usage terms.
