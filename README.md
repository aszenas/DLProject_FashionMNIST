
# Fashion-MNIST: Machine Learning vs. Deep Learning

## Project Overview

This project is developed as part of the Deep Learning
course at Óbuda University.

The objective is to compare classical machine learning
algorithms and deep learning architectures on the
Fashion-MNIST image classification dataset.

The project investigates differences in classification
performance, computational requirements, model complexity
and the impact of hyperparameter optimization.

## Dataset

Fashion-MNIST is a publicly available image classification
dataset developed by Zalando Research.

- 70,000 grayscale images
- Image resolution: 28 × 28 pixels
- 10 clothing categories
- 60,000 original training images
- 10,000 original test images

Dataset source:
https://github.com/zalandoresearch/fashion-mnist

### Dataset Split

All models will use the same dataset split:

| Dataset | Number of samples |
|---|---:|
| Training | 50,000 |
| Validation | 10,000 |
| Test | 10,000 |

Random seed: `42`

## Models and Team Responsibilities

| Team member | Machine Learning | Deep Learning |
|---|---|---|
| Anna | Logistic Regression | ResNet-18 |
| Máté | SVM (RBF kernel) | MLP |
| Beni | Random Forest | Simple CNN |

## Project Workflow

1. Exploratory Data Analysis
2. ML and DL preprocessing pipelines
3. Dataset split
4. Model research and experimental design
5. Baseline model development
6. Hyperparameter optimization
7. Validation-based model selection
8. Final test set evaluation
9. Comparative analysis
10. Documentation and presentation

## Evaluation Metrics

The models will be evaluated using:

- Accuracy
- Macro F1-score
- Confusion matrix
- Training time
- Inference time
- Model size

Hyperparameter optimization and model selection will
be performed using the validation set.

The test set is reserved for final evaluation.

## Technologies (planned)

- Python
- PyTorch
- Torchvision
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook
- Git and GitHub
- GitHub Actions
- Ruff

## Repository Structure (planned)

```text
.github/      Pull Request template and CI workflows
data/         Dataset storage and shared split indices
src/          Reusable preprocessing and model code
notebooks/    EDA, training and evaluation notebooks
configs/      Experiment configurations
results/      Experiment logs and figures
artifacts/    Locally stored trained models
docs/         Feature-level and final documentation
tests/        Shared pipeline tests
```

## Getting Started

Detailed installation and execution instructions
will be added as the project develops.

The planned development environment uses Python 3.11.

## Development Workflow

- `main`: Stable project version
- `dev`: Shared development branch
- `feature/*`: Individual development tasks
- `fix/*`: Bug fixes

All changes must be submitted through Pull Requests
and reviewed by at least one other team member.

For detailed instructions, see:

- [Branching Strategy](docs/branching.md)
- [Documentation Template](docs/documentation_template.md)

## Results

Results will be added after model training,
hyperparameter optimization and final evaluation.

## Project Status

Under development.
