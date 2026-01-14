# AI Guide

This guide helps you understand the SPFlow codebase to provide assistance.

## 🎯 Project Overview

SPFlow is a library for **Probabilistic Circuits (PCs)** built on **PyTorch**. This is the development branch for a major rewrite (v1.0.0).

* **Core Tech:** Python 3.10+, PyTorch, NumPy
* **Environment:** `.venv/`

---

## Programming Practices
* Prefer clarity, simplicity, and explicitness (Zen of Python).
* Write code that is correct, readable, maintainable, and efficient.
* Keep functions small and focused on one task (single responsibility).
* Keep modules cohesive; avoid unnecessary coupling.
* Prefer simple control flow; avoid deeply nested logic.
* Avoid repetition; follow DRY (Don’t Repeat Yourself).

---

## 🛠️ Common Commands

* **Run All Tests:** `uv run pytest -n 4`
* **Format Code:** `uv run black spflow tests`
* **Build Documentation:** `cd docs && make html` (automatically generates API docs from source code)
* **View Documentation:** Open `docs/build/index.html` in your browser
* **Clean Documentation:** `cd docs && make clean`

---

## 🔏 Versioning & Commits

* **Versioning:** Semantic Versioning. The version is in `spflow/__init__.py`.
* **Commits:** Use [Conventional Commits](https://www.conventionalcommits.org/). Keep the commit body brief. Don't mention which files changed in detail since we can see this in the git diff anyway.
* **NOTE:** Never `git add` or `git commit` unless I ask you to.
