# Author Handwriting Recognition

Deep-learning pipeline for **writer identification** from handwritten documents.  
The current model—fine-tuned VGG-16—achieves **56.9 % document-level accuracy** on a private set of 203 writers.

## Repository layout

```plaintext
author-handwriting-recog/
├── Code.ipynb              # end-to-end Jupyter notebooks (outputs stripped)
├── final_project.pdf       # final report (PDF) and diagrams
├── .gitignore 
├── LICENSE             
└── README.md
```


## Dataset (restricted access)

The original handwriting images are the property of the **Israel Police Document Laboratory** and **are not redistributed** in this repository.

To reproduce the reported results you must first obtain written permission from the data owners and place the images under:

```python
data_path = 'path/to/author_images'
```

## Detailed Documentation

The complete project report (methodology, experiments, and results) is available at [final_project.pdf](./final_project.pdf).

## License

Code released under the MIT License (see the LICENSE file for details).
The dataset remains subject to its own copyright and usage terms.
