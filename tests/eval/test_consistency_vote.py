from evaluator.suite.types import Task
from evaluator.consistency.vote import Ballot, first_k, tally_keys, vote

MCQ = Task(id="m1", source="mmlu_pro", problem="?", answer="B", tests=(), meta={})
MATH = Task(id="q1", source="math", problem="?", answer="0.5", tests=(), meta={})
CODE = Task(id="c1", source="humaneval", problem="?", answer=None, tests=(), meta={})


def _b(model, key, text=""):
    return Ballot(model=model, key=key, text=text or f"text-{key}")


def test_plurality_winner_and_tally():
    res = vote(MCQ, [_b("a", "B"), _b("b", "B"), _b("c", "A")])
    assert res.winner_key == "B" and res.tally == {"B": 2, "A": 1}
    assert res.tied is False and res.spoiled == 0 and res.n == 3
    assert res.winner_text == "text-B"


def test_exact_tie_is_flagged_for_the_fuser():
    res = vote(MCQ, [_b("a", "A"), _b("b", "B")])
    assert res.tied is True


def test_spoiled_ballots_are_dropped_but_counted():
    res = vote(MCQ, [_b("a", "B"), _b("b", None), _b("c", None)])
    assert res.winner_key == "B" and res.spoiled == 2 and res.tied is False


def test_all_spoiled_yields_no_winner():
    res = vote(MCQ, [_b("a", None), _b("b", None)])
    assert res.winner_key is None and res.winner_text is None and res.tied is False


def test_math_equivalent_answers_merge_into_one_candidate():
    # "0.5" and "\\frac{1}{2}" are the same answer and must not split the vote
    t = tally_keys(MATH, [_b("a", "0.5"), _b("b", "\\frac{1}{2}"), _b("c", "3")])
    assert max(t.values()) == 2 and len(t) == 2


def test_code_passing_sample_beats_more_numerous_failing_ones():
    res = vote(CODE, [_b("a", "FAIL:aaa"), _b("b", "FAIL:aaa"), _b("c", "PASS:2")])
    assert res.winner_key == "PASS:2"       # execution beats popularity


def test_first_k_takes_k_ballots_per_model():
    ballots = [_b("a", "A"), _b("a", "A"), _b("a", "B"),
               _b("b", "A"), _b("b", "B")]
    got = first_k(ballots, 2)
    assert len(got) == 4                     # 2 from "a", 2 from "b"
    assert [x.model for x in got].count("a") == 2
