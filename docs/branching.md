
# Git Workflow

## Branch Strategy

- `main`: Stable and reviewed project version
- `dev`: Shared development branch
- `feature/*`: Individual development and research tasks
- `fix/*`: Bug fixes

## Branch Naming Convention

Use the task ID and a descriptive branch name.

Examples:

- `feature/03-01-eda`
- `feature/03-02-ml-preprocessing`
- `feature/05-01-resnet18`
- `feature/05-02-svm`
- `fix/dataset-split`

## Development Process – Step by Step

### 1. Switch to the latest dev branch

Before starting a new task:

```bash
git checkout dev
git pull origin dev
```

### 2. Create a new feature branch

Always create a separate branch for your task.

```bash
git checkout -b feature/branch-name
```

Example:

```bash
git checkout -b feature/05-01-resnet18
```

### 3. Implement the task

Follow the agreed repository structure and experiment protocol.

For model development tasks:

- Use the shared dataset split.
- Use the shared preprocessing pipeline.
- Implement and evaluate a baseline model.
- Perform documented hyperparameter experiments.
- Select the best configuration using validation results.
- Save experiment results and relevant figures.
- Do not use the test set for model optimization.

### 4. Document the changes

Create a Markdown file under `docs/features/`
using `docs/documentation_template.md`.

Include implementation details, design decisions,
experiments, results, limitations and references.

Documentation is part of the feature and must be
included in the same Pull Request.

### 5. Make changes and commit

```bash
git status
git add .
git commit -m "feat: implement ResNet-18 baseline"
```

Review staged files before committing.
Do not commit datasets, checkpoints or temporary files.

### 6. Run local quality checks

```bash
ruff check .
ruff format --check .
```

To fix formatting and supported linting issues:

```bash
ruff check --fix .
ruff format .
```

### 7. Push changes to GitHub

For the first push:

```bash
git push -u origin feature/05-01-resnet18
```

For subsequent commits:

```bash
git add .
git commit -m "feat: add ResNet-18 tuning experiments"
git push
```

### 8. Create a Pull Request

1. Open GitHub.
2. Navigate to Pull requests → New pull request.
3. Set base: `dev`.
4. Set compare: your feature branch.
5. Complete the Pull Request template.
6. Assign at least one other team member as reviewer.
7. Wait for approval and successful quality checks.
8. Resolve review comments.
9. Merge using a merge commit.

### 9. Update your branch if necessary

If dev has changed during development:

```bash
git fetch origin
git merge origin/dev
```

Resolve conflicts, run the relevant checks and push.

### 10. Update dev and start the next task

After merging the Pull Request:

```bash
git checkout dev
git pull origin dev
```

Create a new feature branch for your next task.

## Release Process

The final project version is merged from `dev` into
`main` using a reviewed Pull Request.

The release must include:

- Source code and notebooks
- Experiment configurations
- Final evaluation results
- Documentation
- Presentation
- Reproduction instructions

## Conventions

- Never commit directly to main or dev after initialization.
- Every feature branch requires a Pull Request.
- At least one approving peer review is required.
- All required automated checks must pass.
- Each feature must include its documentation.
- Preserve feature history using merge commits.
- Use meaningful commit messages.
- Never commit credentials or large generated files.
- Follow the common experiment protocol.

## Commit Examples

- `feat: implement shared dataset split`
- `feat: add ML preprocessing pipeline`
- `feat: implement CNN baseline`
- `experiment: tune SVM C and gamma`
- `fix: correct validation data transformation`
- `refactor: extract common evaluation functions`
- `test: add dataset split validation`
- `docs: document ResNet-18 architecture`
- `chore: configure Ruff code quality checks`
