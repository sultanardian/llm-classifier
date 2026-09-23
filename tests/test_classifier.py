from types import SimpleNamespace
from unittest import mock

import pytest

from llm_classifier import PromptLogprobClassifier, ask
from llm_classifier.scoring import sigmoid

# Token ID fiktif yang dipakai bersama seluruh fake.
YES_ID = 123
NO_ID = 456
SINGLE_TOKEN_LABELS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class _FakeLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


def _two_choice(yes_val, no_val):
    """Bangun response completions dengan dua choice.

    choice 0 = prompt `Yes` -> classifier membaca token YES_ID.
    choice 1 = prompt `No`  -> classifier membaca token NO_ID.
    score yang dihasilkan = yes_val - no_val.
    """
    return mock.Mock(
        choices=[
            mock.Mock(index=0, prompt_logprobs=[{str(YES_ID): _FakeLogprob(yes_val)}]),
            mock.Mock(index=1, prompt_logprobs=[{str(NO_ID): _FakeLogprob(no_val)}]),
        ]
    )


def _one_label(scores, selected):
    return mock.Mock(
        choices=[
            SimpleNamespace(
                index=0,
                text=selected,
                logprobs=SimpleNamespace(top_logprobs=[scores]),
            )
        ]
    )


def _fake_post(*args):
    payload = args[-1]
    if "messages" in payload:
        return {"tokens": [10, 20, 30]}
    label = payload.get("prompt")
    if label == "No":
        return {"tokens": [NO_ID], "token_strs": ["No"]}
    if isinstance(label, str) and len(label) == 1 and (
        label.isdigit() or label.isalpha()
    ):
        return {
            "tokens": [100 + SINGLE_TOKEN_LABELS.index(label)],
            "token_strs": [label],
        }
    return {"tokens": [YES_ID], "token_strs": ["Yes"]}


def _make_client():
    client = mock.Mock()
    client.base_url = "http://example.test/v1"
    return client


def _make_classifier(client=None, **overrides):
    kwargs = {
        "client": client or _make_client(),
        "model": "test-model",
        "decision_temperature": 3.0,
        "seed": 42,
        "prompt_logprobs": 20,
    }
    kwargs.update(overrides)
    return PromptLogprobClassifier(**kwargs)


def test_tokenize_url_for_handles_trailing_slash():
    from llm_classifier.classifier import _tokenize_url_for

    assert _tokenize_url_for("http://x.test/v1") == "http://x.test/tokenize"
    # openai client menormalisasi base_url dengan slash akhir
    assert _tokenize_url_for("http://x.test/v1/") == "http://x.test/tokenize"
    assert _tokenize_url_for("http://x.test") == "http://x.test/tokenize"
    assert _tokenize_url_for("http://x.test/") == "http://x.test/tokenize"


def test_mayor_settings_are_required():
    # Nilai mayor (pengaruhi hasil scoring) tidak boleh punya default.
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        with pytest.raises(TypeError):
            PromptLogprobClassifier(_make_client(), "test-model")
        with pytest.raises(TypeError):
            PromptLogprobClassifier(
                _make_client(), "test-model", decision_temperature=3.0
            )
        # minor punya default: cukup mayor lengkap
        classifier = PromptLogprobClassifier(
            _make_client(),
            "test-model",
            decision_temperature=3.0,
            seed=42,
            prompt_logprobs=20,
        )
    assert classifier.image_detail == "low"
    assert classifier.timeout == 120
    assert classifier.sampling_temperature == 0.0


def test_from_openai_signature():
    import inspect

    signature = inspect.signature(PromptLogprobClassifier.from_openai)
    required = [
        name
        for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
        and name != "cls"
    ]
    assert required == [
        "base_url",
        "model",
        "api_key",
        "decision_temperature",
        "seed",
        "prompt_logprobs",
    ]


def test_build_prompt_no_candidate():
    prompt = PromptLogprobClassifier.build_prompt("ctx", "q?")
    assert prompt.startswith("Context:\nctx\n\nQuestion: q?\n")
    assert "yes" in prompt


def test_build_prompt_with_candidate():
    prompt = PromptLogprobClassifier.build_prompt("ctx", "q?", "billing: x")
    assert "Proposed answer: billing: x" in prompt
    assert "correct?" in prompt


def test_init_resolves_token_ids_and_tokenize_url():
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        classifier = _make_classifier()

    assert classifier.tokenize_url == "http://example.test/tokenize"
    assert classifier.yes_token_id == YES_ID
    assert classifier.no_token_id == NO_ID
    calls = [call.args for call in mock_post.call_args_list]
    assert all(c[0] == "http://example.test/tokenize" for c in calls)


def test_yes_no_score_text_only():
    client = _make_client()
    client.completions.create.side_effect = lambda model, prompt, **k: _two_choice(-1.0, -4.0)
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        classifier = _make_classifier(client)
        score = classifier.yes_no_score("some prompt")

    # yes - no = -1.0 - (-4.0) = 3.0
    assert score == pytest.approx(3.0)


def test_yes_no_score_uses_cache():
    client = _make_client()
    client.completions.create.side_effect = lambda model, prompt, **k: _two_choice(-1.0, -4.0)
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        classifier = _make_classifier(client)
        first = classifier.yes_no_score("some prompt")
        second = classifier.yes_no_score("some prompt")
        third = classifier.yes_no_score("some prompt", use_cache=False)

    assert first == second
    assert third == pytest.approx(first)
    # dua panggilan unik (pertama + use_cache=False); yang kedua di-cache
    assert client.completions.create.call_count == 2


def test_clear_cache_forces_rerun():
    client = _make_client()
    client.completions.create.side_effect = lambda model, prompt, **k: _two_choice(-1.0, -4.0)
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        classifier = _make_classifier(client)
        classifier.yes_no_score("some prompt")
        classifier.clear_cache()
        classifier.yes_no_score("some prompt")

    assert client.completions.create.call_count == 2


def test_noul_text_only():
    client = _make_client()
    client.completions.create.side_effect = lambda model, prompt, **k: _two_choice(-1.0, -4.0)
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        classifier = _make_classifier(client)
        result = classifier.noul("ctx", "q?")

    expected = sigmoid(3.0 / 3.0)
    assert result["type"] == "noul"
    assert result["noul"] == pytest.approx(expected)


def test_noul_with_criteria_appended_to_question():
    client = _make_client()
    client.completions.create.side_effect = lambda model, prompt, **k: _two_choice(-1.0, -4.0)
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        classifier = _make_classifier(client)
        result = classifier.noul(
            "ctx", "q?", criteria={"true": "meaning T", "false": "meaning F"}
        )

    assert result["type"] == "noul"
    prompt_arg = client.completions.create.call_args.kwargs["prompt"][0]
    assert isinstance(prompt_arg, list)


def test_choice_text_only():
    client = _make_client()
    client.completions.create.return_value = _one_label(
        {"0": 6.0, "1": 2.0, "2": 0.0},
        "0",
    )
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        classifier = _make_classifier(client)
        result = classifier.choice("ctx", "q?", {"a": "def-a", "b": "def-b", "c": None})

    assert result["type"] == "choice"
    assert result["choice"] == "a"
    assert set(result["probabilities"]) == {"a", "b", "c"}
    assert result["probabilities"]["a"] > result["probabilities"]["b"]
    assert result["probabilities"]["b"] > result["probabilities"]["c"]
    assert 0.0 <= result["confidence"] <= 1.0


def test_score_text_only():
    client = _make_client()
    client.completions.create.return_value = _one_label(
        {"0": 0.0, "1": 3.0, "2": 9.0},
        "2",
    )
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        classifier = _make_classifier(client)
        result = classifier.score("ctx", "q?", ["low: x", "mid: y", "high: z"])

    assert result["type"] == "score"
    assert set(result["probabilities"]) == {"0", "1", "2"}
    # kandidat index 2 paling tinggi -> expected_score mendekati 2
    assert result["score"] > 1.5
    assert result["legend"] == {"0": "low: x", "1": "mid: y", "2": "high: z"}


def test_direct_label_missing_scores_become_zero_and_request_is_single():
    client = _make_client()
    client.completions.create.return_value = _one_label(
        {"0": -0.1, "1": -1.1},
        "0",
    )
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        classifier = _make_classifier(client)
        result = classifier.choice(
            "ctx",
            "q?",
            {"a": "def-a", "b": "def-b", "c": "def-c"},
        )

    assert result["choice"] == "a"
    assert result["probabilities"]["c"] == 0.0
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert client.completions.create.call_count == 1
    assert client.completions.create.call_args.kwargs["temperature"] == 0.0
    assert client.completions.create.call_args.kwargs["logprobs"] == 20


def test_direct_label_above_top_twenty_is_used_if_returned():
    client = _make_client()
    client.completions.create.return_value = _one_label(
        {"0": -4.0, "L": -0.1},
        "L",
    )
    criteria = {f"key-{index}": f"definition-{index}" for index in range(22)}
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        classifier = _make_classifier(client)
        result = classifier.choice("ctx", "q?", criteria)

    assert result["choice"] == "key-21"
    assert result["probabilities"]["key-21"] > 0.5
    assert client.completions.create.call_count == 1


def test_direct_label_multimodal_uses_one_chat_request(tmp_path):
    import base64

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    image_file = tmp_path / "tiny.png"
    image_file.write_bytes(png_bytes)

    top_logprobs = [
        SimpleNamespace(token="0", logprob=-0.1),
        SimpleNamespace(token="1", logprob=-2.0),
    ]
    client = _make_client()
    client.chat.completions.create.return_value = mock.Mock(
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(content="0"),
                logprobs=SimpleNamespace(
                    content=[SimpleNamespace(top_logprobs=top_logprobs)]
                ),
            )
        ]
    )
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        classifier = _make_classifier(client)
        result = classifier.choice(
            "ctx",
            "q?",
            {"a": "def-a", "b": "def-b"},
            image_paths=[str(image_file)],
        )

    assert result["choice"] == "a"
    assert client.chat.completions.create.call_count == 1
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.0
    assert kwargs["top_logprobs"] == 20
    assert kwargs["messages"][0]["content"][1]["type"] == "image_url"


def test_multimodal_yes_no_score_uses_chat_completions(tmp_path):
    import base64

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    image_file = tmp_path / "tiny.png"
    image_file.write_bytes(png_bytes)

    class _ChatResponse:
        def __init__(self, token_id, logprob):
            self.prompt_token_ids = [1, 2, 3, token_id]
            self.prompt_logprobs = [{}, {}, {}, {str(token_id): _FakeLogprob(logprob)}]

    client = _make_client()

    def create_chat(**kwargs):
        label = kwargs["messages"][1]["content"]
        token_id = YES_ID if label == "Yes" else NO_ID
        return _ChatResponse(token_id, -1.0 if label == "Yes" else -4.0)

    client.chat.completions.create.side_effect = create_chat
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        classifier = _make_classifier(client)
        score = classifier.yes_no_score("some prompt", image_paths=[str(image_file)])

    assert score == pytest.approx(3.0)
    first_call = client.chat.completions.create.call_args.kwargs
    content = first_call["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["image_url"]["detail"] == "low"
    assert content[1]["uuid"].startswith("sha256:")


def test_image_paths_empty_list_raises():
    client = _make_client()
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        classifier = _make_classifier(client)
        with pytest.raises(ValueError, match="image_paths"):
            classifier.yes_no_score("prompt", image_paths=[])


def test_image_paths_missing_file_raises():
    client = _make_client()
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        classifier = _make_classifier(client)
        with pytest.raises(FileNotFoundError):
            classifier.yes_no_score("prompt", image_paths=["/nonexistent/nope.png"])


def test_image_paths_non_image_file_raises(tmp_path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not an image")
    client = _make_client()
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        classifier = _make_classifier(client)
        with pytest.raises(ValueError, match="bukan image"):
            classifier.yes_no_score("prompt", image_paths=[str(text_file)])


def test_image_cache_reused_and_cleared(tmp_path):
    import base64

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    image_file = tmp_path / "tiny.png"
    image_file.write_bytes(png_bytes)

    class _ChatResponse:
        def __init__(self, token_id, logprob):
            self.prompt_token_ids = [1, token_id]
            self.prompt_logprobs = [{}, {str(token_id): _FakeLogprob(logprob)}]

    client = _make_client()

    def create_chat(**kwargs):
        label = kwargs["messages"][1]["content"]
        token_id = YES_ID if label == "Yes" else NO_ID
        return _ChatResponse(token_id, -2.0)

    client.chat.completions.create.side_effect = create_chat
    with mock.patch.object(PromptLogprobClassifier, "_post_json") as mock_post:
        mock_post.side_effect = _fake_post
        classifier = _make_classifier(client)
        classifier.yes_no_score("prompt", image_paths=[str(image_file)])
        assert len(classifier._image_cache) == 1

        classifier.clear_cache()
        assert classifier._image_cache == {}


def _stub_classifier():
    classifier = mock.Mock()
    classifier.model = "test-model"
    classifier.decision_temperature = 3.0
    classifier.noul = mock.Mock(return_value={"type": "noul", "noul": 0.7})
    classifier.choice = mock.Mock(
        return_value={
            "type": "choice",
            "choice": "a",
            "probabilities": {"a": 0.8, "b": 0.2},
            "confidence": 0.5,
        }
    )
    classifier.score = mock.Mock(
        return_value={
            "type": "score",
            "score": 1.0,
            "probabilities": {"0": 0.4, "1": 0.6},
            "confidence": 0.3,
            "legend": {"0": "x", "1": "y"},
        }
    )
    return classifier


def test_ask_rejects_bad_max_workers():
    classifier = _stub_classifier()
    with pytest.raises(ValueError):
        ask(classifier, "state", {"q": {"type": "noul"}}, max_workers=0)
    with pytest.raises(ValueError):
        ask(classifier, "state", {"q": {"type": "noul"}}, max_workers=1.5)


def test_ask_dispatches_all_kinds_in_order():
    classifier = _stub_classifier()
    questions = {
        "is_dup": {"type": "noul", "instructions": "dup?"},
        "route": {
            "type": "choice",
            "instructions": "route?",
            "criteria": {"a": "def-a", "b": "def-b"},
        },
        "urgency": {
            "type": "score",
            "instructions": "urgency?",
            "criteria": ["low", "high"],
        },
    }
    result = ask(classifier, "state", questions, max_workers=3)

    assert list(result["answers"]) == ["is_dup", "route", "urgency"]
    assert result["usage"]["candidate_sequences"] == 1 + 2 + 2
    assert result["metadata"]["method"] == "vllm_yes_no_logprob"
    assert result["model"] == "test-model"
    classifier.noul.assert_called_once()
    classifier.choice.assert_called_once()
    classifier.score.assert_called_once()


def test_ask_passes_image_paths_and_marks_method():
    classifier = _stub_classifier()
    questions = {"is_dup": {"type": "noul", "instructions": "dup?"}}
    result = ask(
        classifier,
        "state",
        questions,
        image_paths=["/a/b.png"],
        max_workers=1,
    )
    assert result["metadata"]["method"] == "vllm_multimodal_prompt_logprob"
    kwargs = classifier.noul.call_args.kwargs
    assert kwargs["image_paths"] == ["/a/b.png"]


def test_ask_rejects_unknown_kind():
    classifier = _stub_classifier()
    with pytest.raises(ValueError, match="jenis pertanyaan"):
        ask(classifier, "state", {"q": {"type": "wat"}})


def test_ask_choice_requires_mapping():
    classifier = _stub_classifier()
    with pytest.raises(ValueError, match="choice criteria"):
        ask(classifier, "state", {"q": {"type": "choice", "criteria": []}})


def test_ask_score_requires_two_items():
    classifier = _stub_classifier()
    with pytest.raises(ValueError, match="score criteria"):
        ask(classifier, "state", {"q": {"type": "score", "criteria": ["only"]}})


def test_ask_empty_questions():
    classifier = _stub_classifier()
    result = ask(classifier, "state", {})
    assert result["answers"] == {}
    assert result["usage"]["candidate_sequences"] == 0
