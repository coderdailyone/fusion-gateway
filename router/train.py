"""Per-model success classifiers: out-of-fold predictions and a deployable fit.

`fit_oof` produces honest, held-out probabilities of success per (model,
task): the featurizer and the LogisticRegression are both fit on the TRAIN
fold only, in every fold, so no test-fold information (vocabulary or labels)
leaks into a prediction. These out-of-fold probabilities are what the Pareto
policy evaluation consumes, so the numbers it reports are held-out estimates
rather than in-sample overfits.

`fit_final` fits one classifier per model on all available tasks, for use as
a deployable router (not the subject of the held-out evaluation).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from evaluator.suite.types import Task
from router.features import TaskFeaturizer
from router.matrix import ResultMatrix


@dataclass
class OOFPredictions:
    """Out-of-fold predictions and per-model CV AUC.

    proba[model][task_id] = out-of-fold P(correct) for that model on that task.
    cv_auc[model] = roc_auc_score over all out-of-fold predictions for that model
    (nan if the model's labels are single-class).
    """

    proba: dict[str, dict[str, float]] = field(default_factory=dict)
    cv_auc: dict[str, float] = field(default_factory=dict)


def fit_oof(
    tasks: list[Task],
    matrix: ResultMatrix,
    featurizer_cls=TaskFeaturizer,
    models: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 0,
) -> OOFPredictions:
    """Group-by-task CV out-of-fold P(correct) per model, plus per-model AUC.

    For each model: y[i] = matrix.correct[model][tasks[i].id]. Rows are
    grouped by task id (one row per task, so groups are unique) and split
    with GroupKFold, k = min(n_splits, n_tasks) clamped to >= 2. In each
    fold a fresh `featurizer_cls()` is fit on the train tasks only,
    train+test are transformed with it, a LogisticRegression(max_iter=1000)
    is fit on train, and predict_proba on test is collected as that fold's
    OOF proba. Every task ends up with exactly one OOF proba per model.

    If a model's y is all one class, its AUC is float('nan'). A degenerate
    fold (train side single-class) contributes a neutral 0.5 OOF proba for
    its test tasks, mirroring router/learnability.py.
    """
    if models is None:
        models = matrix.models

    n_tasks = len(tasks)
    task_ids = [task.id for task in tasks]
    groups = np.array(task_ids)
    k = max(2, min(n_splits, n_tasks))

    proba: dict[str, dict[str, float]] = {}
    cv_auc: dict[str, float] = {}

    for model in models:
        y = np.array(
            [bool(matrix.correct[model][task_id]) for task_id in task_ids], dtype=int
        )

        if len(np.unique(y)) < 2:
            proba[model] = {task_id: 0.5 for task_id in task_ids}
            cv_auc[model] = float("nan")
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

        proba[model] = {
            task_id: float(oof_pred[i]) for i, task_id in enumerate(task_ids)
        }
        cv_auc[model] = float(roc_auc_score(y, oof_pred))

    return OOFPredictions(proba=proba, cv_auc=cv_auc)


def fit_final(
    tasks: list[Task],
    matrix: ResultMatrix,
    featurizer_cls=TaskFeaturizer,
    models: list[str] | None = None,
) -> dict[str, tuple]:
    """Fit one (featurizer, classifier) pair per model on ALL tasks.

    This is the deployable router: unlike `fit_oof`, there is no held-out
    fold here, so these classifiers should not be used to estimate
    generalization performance.
    """
    if models is None:
        models = matrix.models

    task_ids = [task.id for task in tasks]
    fitted: dict[str, tuple] = {}

    for model in models:
        y = np.array(
            [bool(matrix.correct[model][task_id]) for task_id in task_ids], dtype=int
        )

        featurizer = featurizer_cls()
        featurizer.fit(tasks)
        X = featurizer.transform(tasks)

        clf = LogisticRegression(max_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            clf.fit(X, y)

        fitted[model] = (featurizer, clf)

    return fitted
