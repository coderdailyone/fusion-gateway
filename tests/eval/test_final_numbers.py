from scripts.final_numbers import sota_verdict


def test_sota_verdict_picks_best_single():
    points = {"deepseek-chat": (0.85, 0.0001), "claude-sonnet-5": (0.87, 0.002),
              "gpt-5.6-sol": (0.83, 0.004)}
    v = sota_verdict(points)
    assert v["best_single"] == "claude-sonnet-5"
    assert abs(v["best_acc"] - 0.87) < 1e-9
