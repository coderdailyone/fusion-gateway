from evaluator.build_manifest import HARD_SOURCES, stratified_sample


def test_hard_sources_declared():
    names = {s[0] for s in HARD_SOURCES}
    assert names == {"livecodebench", "gpqa_diamond", "aime", "math_l5"}


def test_stratified_sample_still_pure():
    ids = [f"t{i}" for i in range(20)]
    strata = {i: ("a" if int(i[1:]) % 2 else "b") for i in ids}
    picked = stratified_sample(ids, strata, 10, seed=1)
    assert len(picked) <= 10 and picked == sorted(picked)
