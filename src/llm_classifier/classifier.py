import json
import math
from collections.abc import Mapping
from urllib.request import Request, urlopen

from openai import OpenAI

from llm_classifier.images import load_image_parts
from llm_classifier.scoring import (
    choice_confidence,
    render,
    sigmoid,
    score_confidence,
)


def _tokenize_url_for(base_url):
    url = base_url.strip()
    while url.endswith("/"):
        url = url[:-1]
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url + "/tokenize"


class PromptLogprobClassifier:
    def __init__(
        self,
        client,
        model,
        decision_temperature,
        seed,
        prompt_logprobs,
        sampling_temperature=0.0,
        image_detail="low",
        timeout=120,
    ):
        self.client = client
        self.model = model
        self.decision_temperature = decision_temperature
        self.seed = seed
        self.prompt_logprobs = prompt_logprobs
        self.sampling_temperature = sampling_temperature
        self.image_detail = image_detail
        self.timeout = timeout
        self.tokenize_url = _tokenize_url_for(str(client.base_url))
        self._score_cache = {}
        self._image_cache = {}
        self._label_token_ids = {}
        self._init_label_ids()

    @classmethod
    def from_openai(
        cls,
        base_url,
        model,
        api_key,
        decision_temperature,
        seed,
        prompt_logprobs,
        sampling_temperature=0.0,
        image_detail="low",
        timeout=120,
    ):
        client = OpenAI(api_key=api_key, base_url=base_url)
        return cls(
            client,
            model,
            decision_temperature=decision_temperature,
            seed=seed,
            prompt_logprobs=prompt_logprobs,
            sampling_temperature=sampling_temperature,
            image_detail=image_detail,
            timeout=timeout,
        )

    def clear_cache(self):
        self._score_cache.clear()
        self._image_cache.clear()

    @staticmethod
    def build_prompt(state, question, candidate=None):
        prompt = (
            f"Context:\n{render(state)}\n\n"
            f"Question: {render(question)}\n"
        )

        if candidate is None:
            return prompt + (
                "Is the answer to this question yes? Answer Yes or No."
            )

        return prompt + (
            f"Proposed answer: {render(candidate)}\n"
            "Is this proposed answer correct? Answer Yes or No."
        )

    @staticmethod
    def build_options_prompt(state, question, labels, candidates):
        options = "\n".join(
            f"{label}: {render(candidate)}"
            for label, candidate in zip(labels, candidates)
        )
        return (
            f"Context:\n{render(state)}\n\n"
            f"Question: {render(question)}\n"
            f"Options:\n{options}\n"
            "Answer with exactly one option code."
        )

    @staticmethod
    def _field(value, name):
        if isinstance(value, Mapping):
            return value[name]
        return getattr(value, name)

    @staticmethod
    def _optional_field(value, name, default=None):
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    def _user_content(self, prompt, image_paths=None):
        if image_paths is None:
            return prompt, None
        image_parts, image_key = load_image_parts(
            image_paths, self.image_detail, cache=self._image_cache
        )
        return [
            {"type": "text", "text": prompt},
            *image_parts,
        ], image_key

    def _post_json(self, url, payload):
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def _tokenize_label(self, label):
        data = self._post_json(self.tokenize_url, {
            "prompt": label,
            "add_special_tokens": False,
            "return_token_strs": True,
        })
        tokens = data.get("tokens", [])
        if len(tokens) != 1:
            raise ValueError(
                f"Label {label!r} harus satu token, dapat {data.get('token_strs')}"
            )
        return tokens[0]

    @staticmethod
    def _candidate_labels(count):
        if count < 1:
            raise ValueError("jumlah candidate harus positif")
        labels = []
        for index in range(count):
            if index < 10:
                labels.append(str(index))
            elif index < 36:
                labels.append(chr(ord("A") + index - 10))
            elif index < 62:
                labels.append(chr(ord("a") + index - 36))
            else:
                raise ValueError("maksimal 62 candidate didukung")
        return labels

    def _ensure_label_tokens(self, labels):
        token_to_label = {}
        for label in labels:
            if label not in self._label_token_ids:
                self._label_token_ids[label] = self._tokenize_label(label)
            token_id = self._label_token_ids[label]
            previous = token_to_label.get(token_id)
            if previous is not None and previous != label:
                raise ValueError(
                    f"Label {previous!r} dan {label!r} memakai token ID yang sama"
                )
            token_to_label[token_id] = label

    @staticmethod
    def _top_logprobs(choice):
        logprobs = PromptLogprobClassifier._optional_field(choice, "logprobs")
        if not logprobs:
            return {}

        content = PromptLogprobClassifier._optional_field(logprobs, "content")
        if content:
            row = content[0]
            entries = PromptLogprobClassifier._optional_field(row, "top_logprobs")
            result = {}
            for entry in entries or []:
                token = PromptLogprobClassifier._optional_field(entry, "token")
                value = PromptLogprobClassifier._optional_field(entry, "logprob")
                if token is not None and value is not None:
                    result[str(token)] = float(value)
            if result:
                return result

        rows = PromptLogprobClassifier._optional_field(logprobs, "top_logprobs")
        if not rows:
            return {}
        row = rows[0]
        if isinstance(row, Mapping):
            return {str(token): float(value) for token, value in row.items()}

        result = {}
        for entry in row:
            token = PromptLogprobClassifier._optional_field(entry, "token")
            value = PromptLogprobClassifier._optional_field(entry, "logprob")
            if token is not None and value is not None:
                result[str(token)] = float(value)
        return result

    def _direct_label_scores(self, prompt, labels, image_paths=None, use_cache=True):
        self._ensure_label_tokens(labels)
        user_content, image_key = self._user_content(prompt, image_paths)
        cache_key = (
            "direct_label",
            self.model,
            prompt,
            image_key,
            self.seed,
            self.sampling_temperature,
            self.prompt_logprobs,
        )
        if use_cache and cache_key in self._score_cache:
            return self._score_cache[cache_key]

        if image_paths is None:
            prefix_tokens = self._tokenize_chat(prompt)
            response = self.client.completions.create(
                model=self.model,
                prompt=prefix_tokens,
                max_tokens=1,
                temperature=self.sampling_temperature,
                seed=self.seed,
                logprobs=self.prompt_logprobs,
                extra_body={"return_token_ids": True},
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=1,
                temperature=self.sampling_temperature,
                seed=self.seed,
                logprobs=True,
                top_logprobs=self.prompt_logprobs,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "return_token_ids": True,
                },
            )

        choices = sorted(
            response.choices,
            key=lambda choice: getattr(choice, "index", 0),
        )
        if len(choices) != 1:
            raise RuntimeError("vLLM tidak mengembalikan satu label pilihan")

        top_logprobs = self._top_logprobs(choices[0])
        scores = {label: None for label in labels}
        for token, value in top_logprobs.items():
            label = token.strip()
            if label in scores:
                previous = scores[label]
                scores[label] = value if previous is None else max(previous, value)

        if not any(value is not None and math.isfinite(value) for value in scores.values()):
            raise RuntimeError("vLLM tidak mengembalikan logprob candidate label")

        if use_cache:
            self._score_cache[cache_key] = scores
        return scores

    def _candidate_probabilities(self, scores):
        finite_scores = [
            score for score in scores
            if score is not None and math.isfinite(score)
        ]
        if not finite_scores:
            raise RuntimeError("Tidak ada candidate label dengan logprob")
        if not math.isfinite(self.decision_temperature) or self.decision_temperature <= 0:
            raise ValueError("temperature harus positif dan finite")

        maximum = max(finite_scores) / self.decision_temperature
        weights = [
            0.0 if score is None or not math.isfinite(score)
            else math.exp(score / self.decision_temperature - maximum)
            for score in scores
        ]
        total = sum(weights)
        return [weight / total for weight in weights]

    def _tokenize_chat(self, prompt):
        data = self._post_json(self.tokenize_url, {
            "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        tokens = data.get("tokens", [])
        if not tokens:
            raise RuntimeError("vLLM tidak mengembalikan prompt token IDs")
        return tokens

    @staticmethod
    def _prompt_label_logprob(choice, token_id):
        rows = PromptLogprobClassifier._field(choice, "prompt_logprobs")
        if not rows:
            raise RuntimeError("vLLM tidak mengembalikan prompt_logprobs")
        row = rows[-1]
        entry = row.get(str(token_id), row.get(token_id))
        if entry is None:
            raise RuntimeError(
                f"Token ID {token_id} tidak ditemukan pada prompt_logprobs"
            )
        if isinstance(entry, Mapping):
            return float(entry["logprob"])
        return float(entry.logprob)

    @staticmethod
    def _chat_prompt_label_logprob(response, token_id):
        token_ids = response.prompt_token_ids
        rows = response.prompt_logprobs
        if not token_ids or not rows or len(token_ids) != len(rows):
            raise RuntimeError("vLLM tidak mengembalikan prompt token logprobs")

        positions = [
            index for index, value in enumerate(token_ids)
            if value == token_id
        ]
        if not positions:
            raise RuntimeError(
                f"Token ID {token_id} tidak ditemukan pada image prompt"
            )
        row = rows[max(positions)]
        entry = row.get(str(token_id), row.get(token_id))
        if entry is None:
            raise RuntimeError(
                f"Logprob token ID {token_id} tidak tersedia"
            )
        return float(entry["logprob"] if isinstance(entry, Mapping) else entry.logprob)

    def _multimodal_yes_no_score(self, prompt, image_paths, use_cache=True):
        user_content, image_key = self._user_content(prompt, image_paths)
        cache_key = (
            "yes_no",
            self.model,
            prompt,
            image_key,
            self.seed,
            self.prompt_logprobs,
            self.sampling_temperature,
        )
        if use_cache and cache_key in self._score_cache:
            return self._score_cache[cache_key]

        def score_label(label):
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": label},
                ],
                max_tokens=1,
                temperature=self.sampling_temperature,
                seed=self.seed,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "prompt_logprobs": self.prompt_logprobs,
                    "return_token_ids": True,
                },
            )

        yes_response = score_label("Yes")
        no_response = score_label("No")
        yes_logprob = self._chat_prompt_label_logprob(
            yes_response, self.yes_token_id
        )
        no_logprob = self._chat_prompt_label_logprob(
            no_response, self.no_token_id
        )
        score = yes_logprob - no_logprob

        if use_cache:
            self._score_cache[cache_key] = score
        return score

    def _init_label_ids(self):
        self.yes_token_id = self._tokenize_label("Yes")
        self.no_token_id = self._tokenize_label("No")
        self._label_token_ids.update({
            "Yes": self.yes_token_id,
            "No": self.no_token_id,
        })

    def yes_no_score(self, prompt, image_paths=None, use_cache=True):
        if image_paths is not None:
            return self._multimodal_yes_no_score(
                prompt, image_paths, use_cache=use_cache
            )

        cache_key = (
            "yes_no",
            self.model,
            prompt,
            None,
            self.seed,
            self.prompt_logprobs,
            self.sampling_temperature,
        )
        if use_cache and cache_key in self._score_cache:
            return self._score_cache[cache_key]

        prefix_tokens = self._tokenize_chat(prompt)
        response = self.client.completions.create(
            model=self.model,
            prompt=[
                prefix_tokens + [self.yes_token_id],
                prefix_tokens + [self.no_token_id],
            ],
            max_tokens=1,
            temperature=self.sampling_temperature,
            seed=self.seed,
            extra_body={
                "prompt_logprobs": self.prompt_logprobs,
                "return_token_ids": True,
            },
        )
        choices = sorted(response.choices, key=lambda choice: choice.index)
        if len(choices) != 2:
            raise RuntimeError("vLLM tidak mengembalikan dua prompt score")

        yes_logprob = self._prompt_label_logprob(
            choices[0], self.yes_token_id
        )
        no_logprob = self._prompt_label_logprob(
            choices[1], self.no_token_id
        )
        score = yes_logprob - no_logprob

        if use_cache:
            self._score_cache[cache_key] = score
        return score

    def noul(self, state, question, criteria=None, use_cache=True, image_paths=None):
        if criteria is not None:
            question = (
                f"{question}\n"
                f"Yes means: {render(criteria['true'])}\n"
                f"No means: {render(criteria['false'])}"
            )

        score = self.yes_no_score(
            self.build_prompt(state, question),
            image_paths=image_paths,
            use_cache=use_cache,
        )
        probability = sigmoid(score / self.decision_temperature)
        return {"type": "noul", "noul": probability}

    def choice(self, state, question, criteria, use_cache=True, image_paths=None):
        keys = list(criteria)
        candidates = [
            key if criteria[key] is None
            else f"{key}: {render(criteria[key])}"
            for key in keys
        ]
        labels = self._candidate_labels(len(candidates))
        prompt = self.build_options_prompt(state, question, labels, candidates)
        scores = self._direct_label_scores(
            prompt,
            labels,
            image_paths=image_paths,
            use_cache=use_cache,
        )
        probabilities = self._candidate_probabilities(
            [scores[label] for label in labels]
        )
        selected = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": keys[selected],
            "probabilities": dict(zip(keys, probabilities)),
            "confidence": choice_confidence(probabilities),
        }

    def score(self, state, question, criteria, use_cache=True, image_paths=None):
        labels = self._candidate_labels(len(criteria))
        prompt = self.build_options_prompt(state, question, labels, criteria)
        scores = self._direct_label_scores(
            prompt,
            labels,
            image_paths=image_paths,
            use_cache=use_cache,
        )
        probabilities = self._candidate_probabilities(
            [scores[label] for label in labels]
        )
        expected_score = sum(
            index * probability
            for index, probability in enumerate(probabilities)
        )
        return {
            "type": "score",
            "score": expected_score,
            "probabilities": {
                str(index): probability
                for index, probability in enumerate(probabilities)
            },
            "confidence": score_confidence(probabilities),
            "legend": {
                str(index): render(value)
                for index, value in enumerate(criteria)
            },
        }
