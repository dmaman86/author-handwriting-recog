# Author Handwriting Recognition

This project implements a deep-learning pipeline for **writer identification** based on handwritten documents. The pipeline is designed to evaluate performance both on **seen** and **unseen writers**, reflecting realistic author classification scenarios.

---

## 🚀 Key Highlights

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
- ✅ Modular, scalable pipeline for reproducibility

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
├── code.ipynb
├── datagenerator.py
├── generate_authors_dataset.ipynb
├── LICENSE
├── README.md
├── requirements.txt
└── utils
    ├── __init__.py
    ├── utils_cnn_predictions.py
    ├── utils_dataframe.py
    ├── utils_embedding.py
    ├── utils_files.py
    ├── utils_graphics.py
    ├── utils_model.py
    └── utils.py
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

- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn
- UMAP, t-SNE
- Matplotlib

---

## 📌 Future Work (optional)

- Extend to open-set recognition
- Test with one-shot or few-shot learning
- Explore Transformer-based embedding encoders

---

## Notes:
- If you encounter compatibily issues, check that you have **Python 3.8+** installed.
```bash
python3 --version
```

## 📜 License

Code released under the MIT License (see the LICENSE file for details).
The dataset remains subject to its own copyright and usage terms.
