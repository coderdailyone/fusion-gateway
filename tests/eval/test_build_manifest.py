from evaluator.build_manifest import stratified_sample


def test_stratified_sample_deterministic_and_proportional():
    ids = [f"t{i}" for i in range(100)]
    strata = {tid: ("A" if int(tid[1:]) < 60 else "B") for tid in ids}  # 60% A, 40% B
    a = stratified_sample(ids, strata, 20, seed=7)
    b = stratified_sample(ids, strata, 20, seed=7)
    assert a == b                       # deterministic for a seed
    assert len(a) <= 20
    assert len(set(a)) == len(a)        # no duplicates
    na = sum(1 for t in a if strata[t] == "A")
    nb = sum(1 for t in a if strata[t] == "B")
    assert na >= nb                     # the larger stratum contributes at least as much


def test_stratified_sample_edges():
    assert stratified_sample([], {}, 10, seed=1) == []
    assert stratified_sample(["t1"], {"t1": "A"}, 0, seed=1) == []
