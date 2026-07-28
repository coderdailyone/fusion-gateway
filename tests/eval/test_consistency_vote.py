from evaluator.suite.types import Task
from evaluator.consistency.vote import Ballot, first_k, tally_keys, vote

MCQ = Task(id="m1", source="mmlu_pro", problem="?", answer="B", tests=(), meta={})
MATH = Task(id="q1", source="math", problem="?", answer="0.5", tests=(), meta={})
CODE = Task(id="c1", source="humaneval", problem="?", answer=None, tests=(), meta={})
LCB = Task(id="l1", source="livecodebench", problem="?", answer=None, tests=(), meta={})


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


def test_code_text_beginning_with_pass_is_not_mistaken_for_a_passing_run():
    # livecodebench problems carry no ">>>" doctests, so ballot_key falls back
    # to the raw source for essentially every LCB sample. "PASS_THRESHOLD = 3"
    # is code, not a "PASS:<n>" signature, and must not outrank a plurality.
    res = vote(LCB, [_b("a", "PASS_THRESHOLD = 3"),
                     _b("b", "def f(): ..."), _b("c", "def f(): ...")])
    assert res.winner_key == "def f(): ..."   # plurality, not a fake pass
    assert res.tied is False


def test_tie_winner_does_not_depend_on_ballot_order():
    """A tie must not be broken by "whichever model was listed first".

    Sorting by count alone is stable, so the old winners[0] returned the
    first-encountered top key. On the frozen HumanEval data 51.2% of tasks tie
    and permuting only the ballot order moved vote accuracy 149/164 -> 160/164.
    """
    ballots = [_b("a", "C"), _b("b", "A"), _b("c", "B")]
    first = vote(MCQ, ballots).winner_key
    for perm in ([ballots[1], ballots[2], ballots[0]],
                 [ballots[2], ballots[0], ballots[1]],
                 list(reversed(ballots))):
        res = vote(MCQ, perm)
        assert res.winner_key == first
        assert res.tied is True          # still routed to the fuser
    assert first == "A"                  # min(), not first-encountered


def test_code_tie_winner_is_order_independent_too():
    ballots = [_b("a", "PASS:3", "z-code"), _b("b", "PASS:2", "y-code")]
    assert vote(CODE, ballots).winner_key == "PASS:2"
    assert vote(CODE, list(reversed(ballots))).winner_key == "PASS:2"


def test_code_winner_text_does_not_depend_on_ballot_order():
    """`PASS:<n>` is many-to-one over source texts, so the KEY being stable is not
    enough: several distinct programs share `PASS:2` and Phase B submits the TEXT
    to the official grader. Two programs sharing a key can differ on the hidden
    tests, so an order-dependent text is an order-dependent benchmark number.
    """
    ballots = [_b("a", "PASS:2", "z_solution"),
               _b("b", "PASS:2", "a_solution"),
               _b("c", "FAIL:xyz", "c_solution")]
    fwd = vote(CODE, ballots)
    rev = vote(CODE, list(reversed(ballots)))
    assert fwd.winner_key == rev.winner_key == "PASS:2"
    assert fwd.tied is False and rev.tied is False   # outright plurality both ways
    assert fwd.winner_text == rev.winner_text        # the point of the test
    assert fwd.winner_text == "a_solution"           # min(), not first-encountered


def test_winner_count_and_effective_n_describe_the_deciding_tally():
    res = vote(MCQ, [_b("a", "B"), _b("b", "B"), _b("c", "A"), _b("d", None)])
    assert res.winner_count == 2         # "B" appeared twice
    assert res.effective_n == 3          # 4 ballots, 1 spoiled
    assert res.n == 4 and res.spoiled == 1


def test_winner_count_and_effective_n_are_taken_after_the_pass_filter():
    """`tally` is the PRE-filter dict, so the new fields must not be read off it."""
    res = vote(CODE, [_b("a", "FAIL:aaa"), _b("b", "FAIL:aaa"),
                      _b("c", "FAIL:bbb"), _b("d", "PASS:2")])
    assert res.winner_key == "PASS:2"
    assert res.winner_count == 1         # NOT 2 (the FAIL:aaa plurality)
    assert res.effective_n == 1          # only the passing pool decided it
    assert res.tally == {"FAIL:aaa": 2, "FAIL:bbb": 1, "PASS:2": 1}


def test_math_effective_n_counts_ballots_not_merged_keys():
    res = vote(MATH, [_b("a", "0.5"), _b("b", "\\frac{1}{2}"), _b("c", "3")])
    assert res.winner_key == "0.5" and res.winner_count == 2
    assert res.effective_n == 3


def test_no_winner_reports_zero_counts():
    res = vote(MCQ, [_b("a", None), _b("b", None)])
    assert res.winner_count == 0 and res.effective_n == 0


def test_first_k_takes_k_ballots_per_model():
    ballots = [_b("a", "A"), _b("a", "A"), _b("a", "B"),
               _b("b", "A"), _b("b", "B")]
    got = first_k(ballots, 2)
    assert len(got) == 4                     # 2 from "a", 2 from "b"
    assert [x.model for x in got].count("a") == 2
