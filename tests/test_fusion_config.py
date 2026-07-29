import pytest
from pathlib import Path
from gateway.config import load_config, ConfigError

REAL = Path("configs/gateway.toml")

BASE = """
[budget]
active = "T"
[budgets.T]
[providers.p]
base_url = "https://example.invalid"
api_key_env = "P_KEY"
[models."a"]
provider = "p"
upstream_model = "a"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 1.0
fallback = []
[models."b"]
provider = "p"
upstream_model = "b"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 1.0
fallback = []
[models."c"]
provider = "p"
upstream_model = "c"
in_usd_per_mtok = 1.0
out_usd_per_mtok = 1.0
fallback = []
[fusion]
model = "fusion"
panel = ["a", "b", "c"]
quorum = ["a", "b"]
reviewers = ["a", "b"]
fuser = "b"
review_max_tokens = 512
stage_timeout_s = 120
[policy]
version = "static-v0"
default_model = "fusion"
"""


def write(tmp_path, text):
    p = tmp_path / "g.toml"
    p.write_text(text)
    return p


def test_fusion_section_loads(tmp_path):
    cfg = load_config(write(tmp_path, BASE))
    f = cfg.fusion
    assert f.model == "fusion"
    assert f.panel == ("a", "b", "c") and f.quorum == ("a", "b")
    assert f.reviewers == ("a", "b") and f.fuser == "b"
    assert f.review_max_tokens == 512 and f.stage_timeout_s == 120


def test_default_model_may_name_the_fusion_pseudo_model(tmp_path):
    # `fusion` is deliberately NOT in [models]; the pre-existing
    # "default_model not in models" check must not reject it.
    cfg = load_config(write(tmp_path, BASE))
    assert cfg.default_model == "fusion" and "fusion" not in cfg.models


def test_fusion_section_is_optional(tmp_path):
    text = BASE.split("[fusion]")[0] + '[policy]\nversion = "v"\ndefault_model = "a"\n'
    cfg = load_config(write(tmp_path, text))
    assert cfg.fusion is None


def test_default_model_still_rejected_when_it_names_nothing(tmp_path):
    text = BASE.replace('default_model = "fusion"', 'default_model = "nope"')
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize("old,new", [
    ('panel = ["a", "b", "c"]', 'panel = ["a", "nope"]'),      # panel member unknown
    ('quorum = ["a", "b"]', 'quorum = ["a", "nope"]'),          # quorum member unknown
    ('reviewers = ["a", "b"]', 'reviewers = ["nope"]'),         # reviewer unknown
    ('fuser = "b"', 'fuser = "nope"'),                          # fuser unknown
])
def test_unknown_names_are_rejected(tmp_path, old, new):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, BASE.replace(old, new)))


def test_quorum_must_be_a_subset_of_panel(tmp_path):
    text = BASE.replace('panel = ["a", "b", "c"]', 'panel = ["a", "c"]')
    with pytest.raises(ConfigError):    # quorum names "b", not in panel
        load_config(write(tmp_path, text))


def test_panel_smaller_than_two_is_rejected(tmp_path):
    text = (BASE.replace('panel = ["a", "b", "c"]', 'panel = ["a"]')
                .replace('quorum = ["a", "b"]', 'quorum = ["a"]')
                .replace('reviewers = ["a", "b"]', 'reviewers = ["a"]')
                .replace('fuser = "b"', 'fuser = "a"'))
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


def test_fusion_model_may_not_collide_with_a_real_model(tmp_path):
    text = BASE.replace('model = "fusion"', 'model = "a"')
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


def test_the_real_config_still_loads(tmp_path):
    # Task 6 adds [fusion] to it; until then fusion is None. Either way the
    # real file must load -- several other test modules depend on it.
    load_config(REAL)
