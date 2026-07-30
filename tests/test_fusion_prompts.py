import pytest
from gateway.fusion_prompts import (
    Verdict, render_conversation, build_review_prompt, build_fusion_prompt,
    parse_review,
)

CANDS = {"deepseek-chat": "The answer is 4.", "glm-5.2": "It is 4.", "kimi-k3": "Four."}


def test_render_conversation_keeps_turns_and_roles():
    out = render_conversation([
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ])
    assert "Be terse." in out and "What is 2+2?" in out and "And 3+3?" in out
    # Order preserved: the latest turn is what the panel is answering.
    assert out.index("What is 2+2?") < out.index("And 3+3?")


def test_render_conversation_survives_hostile_message_shapes():
    # app.py does request.json() with no validation, so these are client-reachable.
    for messages in ([], [{}], [{"role": "user"}], [{"content": None}],
                     [{"role": 5, "content": ["a", "b"]}], [None], "notalist"):
        out = render_conversation(messages)
        assert isinstance(out, str)


def test_render_conversation_extracts_text_from_a_bare_dict_content():
    # Clients send content parts as a bare dict as well as a list; a Python
    # repr leaking into the prompt would be read by the reviewer and fuser
    # models as if it were the user's own words.
    out = render_conversation([{"role": "user", "content": {"type": "text", "text": "hi"}}])
    assert "hi" in out and "{" not in out


def test_render_conversation_drops_an_unrenderable_content_object():
    out = render_conversation([{"role": "user", "content": {"unexpected": "shape"}}])
    assert "unexpected" not in out and "{" not in out


def test_review_prompt_never_shows_the_reviewer_its_own_answer():
    p = build_review_prompt("Q", CANDS, reviewer="glm-5.2")
    assert "It is 4." not in p                      # its own candidate text
    assert "The answer is 4." in p and "Four." in p  # the others
    assert "--- Candidate glm-5.2 ---" not in p      # and no block header for it


def test_review_prompt_states_the_exact_verdict_format():
    p = build_review_prompt("Q", CANDS, reviewer="glm-5.2")
    assert "VERDICT <candidate-name> <correct|wrong|unsure> <one-sentence reason>" in p


def test_fusion_prompt_carries_the_majority_copy_rule():
    # This rule is the `break` countermeasure: M5 measured fusion talking
    # itself out of 26 correct answers. Losing it silently loses the fix.
    p = build_fusion_prompt("Q", CANDS, {})
    low = p.lower()
    assert "majority" in low and "copy" in low and "verbatim" in low
    assert "only depart from the majority" in low


def test_fusion_prompt_renders_reviews_and_tolerates_none():
    reviews = {"deepseek-chat": {"glm-5.2": Verdict("correct", "matches")}}
    assert "deepseek-chat says glm-5.2 is correct: matches" in build_fusion_prompt("Q", CANDS, reviews)
    assert "(no reviews available)" in build_fusion_prompt("Q", CANDS, {})


def test_no_benchmark_scaffolding_survives_the_port():
    # These strings exist only to satisfy official graders (MCQ letter
    # extraction, \boxed{} math extraction, per-benchmark format sentences).
    # A chat gateway has no grader; carrying them over would corrupt answers.
    prompts = [build_review_prompt("Q", CANDS, reviewer="glm-5.2"),
               build_fusion_prompt("Q", CANDS, {})]
    import gateway.fusion_prompts as fp
    src = open(fp.__file__).read()
    for banned in ("answer is (X)", "\\boxed", "mmlu_pro", "gpqa_diamond",
                   "humaneval", "livecodebench", "option letter"):
        for p in prompts:
            assert banned not in p, banned
        assert banned not in src, banned


def test_parse_review_extracts_valid_lines():
    text = ("VERDICT glm-5.2 correct matches the others\n"
            "VERDICT kimi-k3 wrong off by one\n")
    got = parse_review(text, {"glm-5.2", "kimi-k3"})
    assert got["glm-5.2"] == Verdict("correct", "matches the others")
    assert got["kimi-k3"].verdict == "wrong"


def test_parse_review_drops_junk_without_raising():
    text = ("hello\n"
            "VERDICT\n"
            "VERDICT glm-5.2\n"
            "VERDICT nobody correct not a target\n"      # unknown target
            "VERDICT glm-5.2 maybe not a valid verdict\n"  # invalid verdict
            "VERDICT kimi-k3 CORRECT case is normalised\n")
    got = parse_review(text, {"glm-5.2", "kimi-k3"})
    assert "nobody" not in got and got.get("glm-5.2") is None
    assert got["kimi-k3"].verdict == "correct"
    assert parse_review(None, {"a"}) == {}
    assert parse_review("", set()) == {}


def test_parse_review_allows_a_verdict_with_no_reason():
    assert parse_review("VERDICT a correct", {"a"}) == {"a": Verdict("correct", "")}


from gateway.fusion import Candidate

TOOL = ({"id": "c1", "type": "function",
         "function": {"name": "read", "arguments": '{"path":"a.py"}'}},)


def test_a_tool_call_candidate_renders_its_name_and_arguments():
    p = build_review_prompt("Q", {"m1": Candidate("", TOOL),
                                  "m2": Candidate("prose")}, reviewer="m2")
    assert "read" in p and '"path"' in p and "a.py" in p


def test_a_reviewer_still_never_sees_its_own_tool_call():
    p = build_review_prompt("Q", {"m1": Candidate("", TOOL),
                                  "m2": Candidate("", TOOL)}, reviewer="m1")
    assert "--- Candidate m1 ---" not in p


def test_the_fusion_prompt_tells_the_fuser_to_act_not_narrate():
    p = build_fusion_prompt("Q", {"m1": Candidate("", TOOL)}, {})
    low = p.lower()
    assert "tool" in low and ("call" in low or "action" in low)


def test_prose_prompts_are_byte_identical_to_the_string_era():
    # The prose path must not shift by a single character. These are the exact
    # strings the pre-Candidate implementation produced.
    cands = {"a": Candidate("first"), "b": Candidate("second")}
    review = build_review_prompt("CONV", cands, reviewer="b")
    assert "--- Candidate a ---\nfirst" in review
    assert "--- Candidate b ---" not in review
    fusion = build_fusion_prompt("CONV", cands, {})
    assert "--- Candidate a ---\nfirst\n\n--- Candidate b ---\nsecond" in fusion
    assert "(no reviews available)" in fusion
