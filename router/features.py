"""Public-only feature extraction for routing.

Features are derived exclusively from information available before a task is
solved: the problem text, its source dataset, and the problem's length.
`task.answer` and `task.tests` are NEVER read here — the learnability gate and
router training must not leak the gold answer into the feature space.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from evaluator.suite.types import Task


class TaskFeaturizer:
    """Turns tasks into a public-only sparse feature matrix.

    Features = TF-IDF(problem, word 1-2grams, max_features=2000) hstacked with
    a source one-hot (columns fixed at fit time) and a log1p(len(problem))
    column.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
        self.sources: list[str] = []

    def fit(self, tasks: list[Task]) -> "TaskFeaturizer":
        self.vectorizer.fit(task.problem for task in tasks)
        self.sources = sorted({task.source for task in tasks})
        return self

    def transform(self, tasks: list[Task]) -> sp.csr_matrix:
        tfidf = self.vectorizer.transform(task.problem for task in tasks)

        source_index = {source: i for i, source in enumerate(self.sources)}
        onehot = sp.lil_matrix((len(tasks), len(self.sources)))
        for row, task in enumerate(tasks):
            col = source_index.get(task.source)
            if col is not None:
                onehot[row, col] = 1.0

        length = np.array(
            [[np.log1p(len(task.problem))] for task in tasks], dtype=float
        )

        return sp.hstack([tfidf, onehot.tocsr(), sp.csr_matrix(length)]).tocsr()
