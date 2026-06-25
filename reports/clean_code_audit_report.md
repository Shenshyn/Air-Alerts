# Clean Code Audit Report: Air Alerts Codebase

This report provides a thorough analysis of the codebase located in `/src/`, evaluated against the clean code principles outlined by Robert C. Martin ("Uncle Bob"). The goal is to identify areas for improvement in structure, readability, and maintainability.

## 1. Meaningful and Unambiguous Naming
**Status: Fair to Good**

Overall, the codebase uses relatively clear and descriptive names for variables and functions.
* **Strengths**: Variable names like `kmf_global`, `region_medians`, `split_date`, and `oblast_dummies` clearly indicate their purpose and the data they hold. Paths are also explicitly named (e.g., `cox_model_path`, `data_path`).
* **Areas for Improvement**: The generic variable `df` is used pervasively for pandas DataFrames. While this is standard convention in data science scripts, passing `df` through monolithic functions makes it harder to track state changes. Variables like `events_sweep` could simply be `events`, and temporary loop variables like `i` and `ends` are occasionally used where more descriptive names might clarify the complex sweep-line algorithm.

## 2. Single Responsibility Principle (SRP)
**Status: Severely Violated**

The Single Responsibility Principle states that a module, class, or function should have only one reason to change. 
* **Strengths**: `data_loader.py` partially adheres to SRP by separating concerns into `load_and_preprocess_data()` and `cache_processed_data()`.
* **Weaknesses**: The scripts `baseline_km.py`, `evaluate.py`, `model_cox.py`, and `model_xgb.py` package their entire execution logic into single, massive `main()` functions. For example, `evaluate.py` handles:
  1. File I/O (loading data, saving models, saving plots)
  2. Data preprocessing
  3. Complex algorithmic feature engineering (sweep-line algorithm)
  4. Train/test splitting
  5. Model evaluation (KM, Cox, XGBoost)
  6. Matplotlib visualization logic
  
This makes the functions incredibly difficult to test, maintain, or read. Any change to feature engineering, plotting, or model evaluation requires modifying the same massive function.

## 3. Absence of Logic Duplication (DRY Principle)
**Status: Severely Violated**

The "Don't Repeat Yourself" (DRY) principle is the most critically violated rule in this codebase. There is massive, unchecked copy-pasting across the machine learning scripts.
* **Feature Engineering**: The sweep-line algorithm used to calculate alert concurrency (~35 lines of dense logic) is duplicated verbatim across `evaluate.py`, `model_cox.py`, and `model_xgb.py`.
* **Temporal Encoding**: The logic for generating cyclic features (`sin_hour`, `cos_hour`, etc.) is identical in all three model scripts.
* **Spatial Encoding**: One-hot encoding for the `oblast` column is repeated identically.
* **Train/Test Splitting**: The chronological split logic (`df['started_at'] < pd.to_datetime('2026-01-01...')`) is duplicated.
* **Data Validation**: Ensuring `duration_minutes` exists or renaming it from `duration` is repeated in four out of five files.

**Impact**: If a bug is found in the concurrency algorithm, or if the train/test split date needs to change, the developer must remember to update the exact same code in three different files. This is a massive source of future technical debt and bugs.

## 4. Reasonable and Small Function and Module Sizes
**Status: Poor**

* **Strengths**: The individual module sizes (file lengths) are generally small to moderate, ranging from 116 lines (`data_loader.py`) to 330 lines (`evaluate.py`).
* **Weaknesses**: The function sizes are far too large. Clean Code advocates for functions to be very small, ideally doing just one thing.
  * `evaluate.py` has a single `main()` function spanning roughly 320 lines.
  * `model_xgb.py` and `model_cox.py` have `main()` functions spanning over 130 lines.
  * These large functions force the reader to keep an immense amount of context (variables, state, intermediate DataFrames) in their head at once.

## 5. Minimization of Side Effects and Implicit Dependencies
**Status: Fair**

* **Strengths**: The scripts run top-to-bottom and mostly isolate their side effects to the end of the script (e.g., saving a plot or a model), which is better than scattering side effects throughout.
* **Weaknesses**: Because all logic is inside `main()`, feature engineering and data processing implicitly depend on the specific state of the `df` variable at a given line. There are no pure functions for transformations (e.g., a function `def calculate_concurrency(df: pd.DataFrame) -> pd.DataFrame:` that is easily unit testable without side effects).

## Conclusion and Recommendations

The codebase currently functions more as a collection of rapid prototypes or Jupyter notebooks exported to `.py` files rather than a production-ready repository. 

To bring this codebase into alignment with Clean Code principles, the following refactoring steps are highly recommended:
1. **Extract Feature Engineering**: Create a `features.py` module. Move the concurrency calculation, temporal encoding, and spatial encoding into well-named, pure functions within this module.
2. **Centralize Utilities**: Move common logic like data loading, train/test splitting, and `duration_minutes` validation into a shared `utils.py` or within `data_loader.py`.
3. **Break Down Monoliths**: Refactor the `main()` functions in the modeling and evaluation scripts. The `main()` function should read like a high-level table of contents, orchestrating calls to smaller, focused functions.
4. **Isolate Plotting**: Extract the complex matplotlib configurations into dedicated plotting functions, perhaps in a `visualization.py` module.
