from app.chunker import SentenceChunker, clean_for_speech


def stream(text, chunker=None, step=3):
    c = chunker or SentenceChunker()
    out = []
    for i in range(0, len(text), step):
        out += c.feed(text[i : i + step])
    return out + c.flush()


def test_splits_sentences_incrementally():
    segs = stream("Battery is at forty percent. Return to home now! Is the wind calm? Yes.")
    assert segs == ["Battery is at forty percent.", "Return to home now!", "Is the wind calm?", "Yes."]


def test_first_segment_can_be_clause_for_low_latency():
    c = SentenceChunker(first_clause_min_chars=20)
    segs = stream("Given the gusty conditions today, I would land immediately and wait it out.", c)
    assert segs[0] == "Given the gusty conditions today,"
    assert segs[1].startswith("I would land")


def test_does_not_split_decimals_or_abbreviations():
    segs = stream("Keep e.g. 3.5 metres from Dr. Rao at all times. Then climb.")
    assert segs == ["Keep e.g. 3.5 metres from Dr. Rao at all times.", "Then climb."]


def test_max_chars_forces_cut():
    c = SentenceChunker(max_chars=40)
    segs = stream("word " * 30, c)
    assert all(len(s) <= 40 for s in segs)
    assert " ".join(segs).split() == ["word"] * 30


def test_tokens_arrive_one_at_a_time():
    c = SentenceChunker()
    got = []
    for tok in ["Hold", " position", ".", " Then", " land", "."]:
        got += c.feed(tok)
    # last sentence has no trailing whitespace yet -> only emitted on flush
    assert got == ["Hold position."]
    assert c.flush() == ["Then land."]


def test_markdown_stripped():
    assert clean_for_speech("**Land** `now`  please") == "Land now please"
    assert stream("") == []


def test_modes():
    text = "Given the gusty conditions today, I would land. Then wait."
    assert stream(text, SentenceChunker(mode="sentence"))[0] == "Given the gusty conditions today, I would land."
    assert stream(text, SentenceChunker(mode="full")) == [text]


def test_eager_caps_first_segment_words():
    text = "The safest option in these gusty conditions is to land now. Then wait."
    segs = stream(text, SentenceChunker(mode="eager", first_max_words=5))
    assert segs[0] == "The safest option in these"
    assert segs[1] == "gusty conditions is to land now."
    assert stream("Hold. Then land.", SentenceChunker(mode="eager")) == ["Hold. Then land."]
