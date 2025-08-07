# Author Handwriting Recognition

This project implements a deep-learning pipeline for **writer identification** based on handwritten documents. The pipeline is designed to evaluate performance both on **seen** and **unseen writers**, reflecting realistic author classification scenarios.

---

## 🚀 Key Highlights

- ✅ **Convolutional neural network - CNN** trained on 204 known writers
- ✅ High accuracy on seen writers - Softmax Model:
  - 98.04% accuracy with majority voting and Softmax Summation
- ✅ High accuracy on seen writers - Embedding Model:
  - 93.1% with majority voting
- ✅ Generalization to unseen writers - Embedding Model:
  - 60.1% accuracy with softmax summation
- ✅ Multiple evaluation strategies:
  - Patch-level prediction
  - Majority vote
  - Softmax score summation
- ✅ Embedding visualization with UMAP and t-SNE
- ✅ Modular, scalable pipeline for reproducibility

---

## 🧠 Project Summary

After initially reaching 56.9% accuracy on known writers during a prior deep learning course, this new version redesigns the full pipeline to:

- Improve document-level accuracy significantly (now over 93%).
- Introduce embedding-based generalization for authors not seen during training.
- Include visual inspection of embedding quality and separation.

---

## 📂 Repository Layout

```plaintext
author-handwriting-recog/
├── Code.ipynb                      # end-to-end Jupyter notebooks (outputs stripped)
├── generate_authors_dataset.ipynb  # notebook to generate the dataset
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Results

### Seen Writers (204 authors) - **Softmax Model**

| Method                  | Accuracy   |
| ----------------------- | ---------- |
| Softmax – Patch Level   | 57.75%     |
| Softmax – Majority Vote | **98.04%** |
| Softmax – Softmax Sum   | 98.04%     |

### Seen Writers (204 authors) - **Embedding Model**

| Method                  | Accuracy  |
| ----------------------- | --------- |
| Softmax – Patch Level   | 48.1%     |
| Softmax – Majority Vote | **93.1%** |
| Softmax – Softmax Sum   | 90.6%     |

### Unseen Writers (203 new authors) - **Embedding Model**

| Method                  | Accuracy  |
| ----------------------- | --------- |
| Softmax – Patch Level   | 22.9%     |
| Softmax – Majority Vote | 50.2%     |
| Softmax – Softmax Sum   | **60.1%** |

> ✅ Results based on Euclidean-distance embeddings and centroid classification.

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

## 📜 License

Code released under the MIT License (see the LICENSE file for details).
The dataset remains subject to its own copyright and usage terms.
