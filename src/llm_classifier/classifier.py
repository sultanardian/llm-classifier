from collections.abc import Mapping
from urllib.request import Request, urlopen

import json

from openai import OpenAI

from llm_classifier.images import load_image_parts
from llm_classifier.scoring import (
    choice_confidence,
    render,
    sigmoid,
    softmax,
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
        image_detail="low",
        timeout=120,
    ):
        self.client = client
        self.model = model
        self.decision_temperature = decision_temperature
        self.seed = seed
        self.prompt_logprobs = prompt_logprobs
        self.image_detail = image_detail
        self.timeout = timeout
        self.tokenize_url = _tokenize_url_for(str(client.base_url))
        self._score_cache = {}
        self._image_cache = {}
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
    def _field(value, name):
        if isinstance(value, Mapping):
            return value[name]
        return getattr(value, name)

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
            self.model, prompt, image_key, self.seed, self.prompt_logprobs
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
                temperature=1.0,
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

    def yes_no_score(self, prompt, image_paths=None, use_cache=True):
        if image_paths is not None:
            return self._multimodal_yes_no_score(
                prompt, image_paths, use_cache=use_cache
            )

        cache_key = (self.model, prompt, None, self.seed, self.prompt_logprobs)
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
            temperature=1.0,
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
        scores = [
            self.yes_no_score(
                self.build_prompt(state, question, candidate),
                image_paths=image_paths,
                use_cache=use_cache,
            )
            for candidate in candidates
        ]
        probabilities = softmax(scores, self.decision_temperature)
        selected = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": keys[selected],
            "probabilities": dict(zip(keys, probabilities)),
            "confidence": choice_confidence(probabilities),
        }

    def score(self, state, question, criteria, use_cache=True, image_paths=None):
        scores = [
            self.yes_no_score(
                self.build_prompt(state, question, candidate),
                image_paths=image_paths,
                use_cache=use_cache,
            )
            for candidate in criteria
        ]
        probabilities = softmax(scores, self.decision_temperature)
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
