from router.features import TaskFeaturizer
from router.train import fit_oof

# reuse the predictable generator idea from test_learnability
from tests.router.test_learnability import make


def test_oof_predictions_cover_all_tasks_and_auc():
    tasks, m = make(True, n=60)
    oof = fit_oof(tasks, m, TaskFeaturizer)
    assert set(oof.proba["A"].keys()) == {t.id for t in tasks}  # every task has an OOF proba
    assert oof.cv_auc["A"] > 0.7
    assert all(0.0 <= p <= 1.0 for p in oof.proba["A"].values())
