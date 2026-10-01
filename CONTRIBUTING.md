# 🐘 Contributing to Project Zogan

Thank you for contributing to Project Zogan! To ensure stability and protect the wildlife monitoring system, all contributions must pass our automated quality gates before merging into `main`.

---

## 🚀 Development Workflow

```
main (Protected)
  ▲
  │  (Pull Request — CI status checks must pass)
  │
feature/your-feature-name
```

### 1. Create a Feature Branch
Always branch off the latest `main` branch with a descriptive name:
```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

### 2. Set Up Local Environment
Ensure your Python environment is up to date with project and development dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

### 3. Make Changes and Verify Locally
Before committing or opening a pull request, run all local CI checks:
```bash
# 1. Check Python syntax
python -m compileall -q -x "venv|\.venv" .

# 2. Check code quality and style
ruff check .

# 3. Check code formatting
ruff format --check .

# 4. Verify dataset and configuration
python ai/validate_dataset.py

# 5. Run automated test suites
python test_phase2.py
python test_phase3_1.py
python test_phase3_2.py
python test_phase3_4.py
python test_phase4.py
python test_phase5.py
python ai/simulate_risk.py
```

To auto-format code locally:
```bash
ruff format .
ruff check --fix .
```

### 4. Push Branch & Open Pull Request
Commit your changes with clear messages and push your branch to GitHub:
```bash
git add .
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
```

### 5. Automated CI Verification
- Opening a PR against `main` automatically triggers GitHub Actions workflow (`.github/workflows/ci.yml`).
- If any check fails (syntax, lint, format, dataset, or test), inspect the CI failure logs and fix it locally.
- Pushing new commits to your branch automatically cancels outdated runs and re-runs CI on the latest commit.

### 6. Merge Requirements
Direct pushes to `main` are restricted. A PR can only be merged when:
- ✅ The **`ci`** workflow job ("Code Quality & Test Gate") has passed.
- ✅ The branch is up to date with `main`.
- ✅ No merge conflicts or failing tests exist.

---

## 🔒 Security & Privacy Guidelines
- **Never commit credentials, API keys, or private coordinates**: Keep all local credentials in `.env` (which is excluded by `.gitignore`).
- **Simulated Coordinates Only**: All unit tests and default configurations must use synthetic demo coordinates (e.g. `20.123456, 85.123456`), never real private locations.
- **Headless Testing**: Do not add tests that open GUI windows (`cv2.imshow`), require webcams, or download multi-gigabyte models in automated test scripts.
