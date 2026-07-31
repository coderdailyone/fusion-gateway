from gateway.tool_vote import canonical_calls, plurality, all_readonly

READONLY = frozenset({"read", "ls", "grep", "find"})


def call(name, args):
    return {"id": "x", "type": "function",
            "function": {"name": name, "arguments": args}}


def test_key_order_and_whitespace_do_not_matter():
    a = canonical_calls([call("read", '{"path":"a.py","limit":10}')])
    b = canonical_calls([call("read", '{ "limit": 10 , "path" : "a.py" }')])
    assert a == b is not None


def test_comparison_is_exact_not_semantic():
    # "a.py" and "./a.py" name the same file to a human. Treating them as
    # equal would need an LLM, which is the cost this module exists to avoid.
    a = canonical_calls([call("read", '{"path":"a.py"}')])
    b = canonical_calls([call("read", '{"path":"./a.py"}')])
    assert a != b


def test_a_different_tool_name_never_matches():
    assert canonical_calls([call("read", "{}")]) != canonical_calls([call("write", "{}")])


def test_unparseable_arguments_are_unusable():
    # Unusable must be None, not some sentinel that could match another
    # unusable call -- two models failing differently is not agreement.
    assert canonical_calls([call("read", "{not json")]) is None
    assert canonical_calls([call("read", None)]) is None
    assert plurality({"a": None, "b": None, "c": None}) is None


def test_an_empty_call_list_is_unusable():
    # THE load-bearing case: a text-only candidate has tool_calls == [].
    # If that canonicalised to (), two prose candidates would "agree" and
    # prose would be routed through the tool path.
    assert canonical_calls([]) is None
    assert canonical_calls(None) is None
    assert plurality({"a": canonical_calls([]), "b": canonical_calls([])}) is None


def test_malformed_shapes_never_raise():
    for bad in ("notalist", [None], [{}], [{"function": None}],
                [{"function": {"name": 5, "arguments": "{}"}}],
                [{"function": {"arguments": "{}"}}],
                # Non-string arguments must not raise
                [call("read", 42)],
                [call("read", {"a": 1})],
                [call("read", [1, 2])],
                [call("read", True)],
                # Empty string tool name must be rejected
                [call("", "{}")]):
        assert canonical_calls(bad) is None


def test_parallel_calls_ignore_order_but_not_duplication():
    one = canonical_calls([call("read", '{"path":"a"}'), call("read", '{"path":"b"}')])
    two = canonical_calls([call("read", '{"path":"b"}'), call("read", '{"path":"a"}')])
    assert one == two is not None
    dup = canonical_calls([call("read", '{"path":"a"}'), call("read", '{"path":"a"}')])
    single = canonical_calls([call("read", '{"path":"a"}')])
    assert dup != single


def test_plurality_returns_a_two_of_three_winner():
    same = canonical_calls([call("read", '{"path":"a"}')])
    other = canonical_calls([call("write", '{"path":"a"}')])
    winner = plurality({"m1": same, "m2": other, "m3": same})
    assert winner in ("m1", "m3")


def test_plurality_is_none_on_a_three_way_split():
    got = plurality({"m1": canonical_calls([call("read", '{"path":"a"}')]),
                     "m2": canonical_calls([call("read", '{"path":"b"}')]),
                     "m3": canonical_calls([call("read", '{"path":"c"}')])})
    assert got is None


def test_plurality_is_deterministic():
    # Two models tie; the winner must not depend on dict iteration order.
    same = canonical_calls([call("read", '{"path":"a"}')])
    first = plurality({"b": same, "a": same})
    second = plurality({"a": same, "b": same})
    assert first == second

    # Hard case: 4 models with two pairs each holding 2 votes (a tie).
    # Deterministic: models are sorted, first tie group wins.
    call_a = canonical_calls([call("read", '{"x":"1"}')])
    call_b = canonical_calls([call("read", '{"x":"2"}')])
    # m1, m2 vote call_a; m3, m4 vote call_b. Across different insertion orders,
    # the same pair should win (m1 and m2 are alphabetically first).
    panel1 = plurality({"m4": call_b, "m3": call_b, "m2": call_a, "m1": call_a})
    panel2 = plurality({"m1": call_a, "m2": call_a, "m3": call_b, "m4": call_b})
    panel3 = plurality({"m3": call_b, "m1": call_a, "m4": call_b, "m2": call_a})
    assert panel1 == panel2 == panel3 == "m1"


def test_all_readonly_is_exact_and_default_deny():
    assert all_readonly(canonical_calls([call("read", "{}")]), READONLY)
    assert not all_readonly(canonical_calls([call("write", "{}")]), READONLY)
    # An unlisted tool -- a new Pi tool, or another client's -- is write-class.
    assert not all_readonly(canonical_calls([call("brand_new_tool", "{}")]), READONLY)
    # No prefix matching: "read" being listed must not admit "readwrite".
    assert not all_readonly(canonical_calls([call("readwrite", "{}")]), READONLY)
    # Case-sensitive.
    assert not all_readonly(canonical_calls([call("Read", "{}")]), READONLY)
    # A mixed batch is write-class: one unsafe call taints the whole step.
    mixed = canonical_calls([call("read", "{}"), call("write", "{}")])
    assert not all_readonly(mixed, READONLY)


def test_all_readonly_rejects_an_unusable_batch():
    assert not all_readonly(None, READONLY)


def test_missing_arguments_key_defaults_to_empty_json():
    # When "arguments" key is entirely absent, it defaults to "{}".
    call_no_args = {"id": "x", "type": "function",
                    "function": {"name": "read"}}
    result = canonical_calls([call_no_args])
    assert result == (("read", "{}"),)
    assert result is not None


def test_a_custom_type_call_never_classifies_by_its_function_name():
    # M9 final review, finding 4 (Minor): `_canonical_one` used to read only
    # `function.name`, so a call shaped like OpenAI's `type: "custom"` --
    # {"type": "custom", "custom": {"name": "bash", ...}, "function":
    # {"name": "read", ...}} -- classified (and canonicalised) as "read"
    # purely from the `function` block, even though a client dispatching on
    # `type` would execute the `custom` block's "bash" instead. Before the
    # fix this returned (("read", "{}"),) -- a usable, read-classified call
    # -- and would have sailed through `all_readonly` as if it were the safe
    # "read" tool.
    hostile = {"id": "x", "type": "custom",
              "custom": {"name": "bash", "input": "rm -rf /"},
              "function": {"name": "read", "arguments": "{}"}}
    assert canonical_calls([hostile]) is None
    assert not all_readonly(canonical_calls([hostile]), READONLY)


def test_a_function_type_or_an_omitted_type_still_classifies_normally():
    # The only two shapes this module has ever accepted must keep working:
    # `type: "function"` (every fixture above) and `type` omitted entirely
    # (OpenAI's wire format treats an absent `type` as "function").
    explicit = {"id": "x", "type": "function", "function": {"name": "read", "arguments": "{}"}}
    omitted = {"id": "x", "function": {"name": "read", "arguments": "{}"}}
    assert canonical_calls([explicit]) == (("read", "{}"),)
    assert canonical_calls([omitted]) == (("read", "{}"),)
