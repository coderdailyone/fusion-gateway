"""Learnability gate: can public features predict a model's per-task correctness?

For each model we ask whether a simple linear classifier trained on
`TaskFeaturizer` features (problem text + source + length — never the gold
answer) can predict that model's per-task correctness better than chance. If
no model clears the AUC threshold, per-task routing has no signal to exploit
and a full sampling run is not worth its cost.

Cross-validation is honest by construction: the featurizer is a fresh
instance re-fit on the TRAIN fold's tasks only, in every fold. Fitting the
featurizer once on the full task set before CV-ing would leak test-fold
vocabulary into the vectorizer and inflate the AUC.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from evaluator.suite.types import Task
from router.features import TaskFeaturizer
from router.matrix import ResultMatrix


def per_model_cv_auc(
    tasks: list[Task],
    matrix: ResultMatrix,
    featurizer_cls=TaskFeaturizer,
    models: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Out-of-fold CV AUC of predicting each model's correctness from public features.

    For each model: y[i] = matrix.correct[model][tasks[i].id]. Rows are
    grouped by task id (one row per task, so groups are unique) and split
    with GroupKFold. In each fold a fresh `featurizer_cls()` is fit on the
    train tasks only, train+test are transformed with it, a
    LogisticRegression(max_iter=1000) is fit on train, and predict_proba on
    test is collected into an out-of-fold prediction array. One
    roc_auc_score is computed over all out-of-fold predictions.

    If a model's y is all one class (no variance to learn or to score),
    its AUC is float('nan').
    """
    if models is None:
        models = matrix.models

    n_tasks = len(tasks)
    task_ids = [task.id for task in tasks]
    groups = np.array(task_ids)
    k = max(2, min(n_splits, n_tasks))

    aucs: dict[str, float] = {}
    for model in models:
        y = np.array(
            [bool(matrix.correct[model][task_id]) for task_id in task_ids], dtype=int
        )

        if len(np.unique(y)) < 2:
            aucs[model] = float("nan")
            continue

        oof_pred = np.full(n_tasks, np.nan)
        gkf = GroupKFold(n_splits=k)
        for train_idx, test_idx in gkf.split(np.zeros(n_tasks), y, groups=groups):
            y_train = y[train_idx]
            if len(np.unique(y_train)) < 2:
                # Degenerate fold (train side is single-class): nothing to
                # learn, but still fill in a neutral prediction so every
                # task contributes to the final OOF AUC.
                oof_pred[test_idx] = 0.5
                continue

            train_tasks = [tasks[i] for i in train_idx]
            test_tasks = [tasks[i] for i in test_idx]

            featurizer = featurizer_cls()
            featurizer.fit(train_tasks)
            X_train = featurizer.transform(train_tasks)
            X_test = featurizer.transform(test_tasks)

            clf = LogisticRegression(max_iter=1000, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                clf.fit(X_train, y_train)
            oof_pred[test_idx] = clf.predict_proba(X_test)[:, 1]

        aucs[model] = float(roc_auc_score(y, oof_pred))

    return aucs


def gate(aucs: dict[str, float], threshold: float = 0.55) -> dict:
    """GO/NO_GO verdict on whether per-task routing signal exists.

    GO iff at least one non-nan model AUC is >= threshold.
    """
    non_nan = {model: auc for model, auc in aucs.items() if not np.isnan(auc)}
    max_auc = max(non_nan.values()) if non_nan else float("nan")
    passing = sorted(model for model, auc in non_nan.items() if auc >= threshold)

    return {
        "verdict": "GO" if passing else "NO_GO",
        "max_auc": max_auc,
        "passing": passing,
    }
