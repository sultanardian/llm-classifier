import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor


def ask(
    classifier,
    state,
    questions,
    use_cache=True,
    image_paths=None,
    max_workers=None,
):
    started = time.perf_counter()
    question_items = list(questions.items())
    if max_workers is None:
        max_workers = min(8, len(question_items)) if question_items else 1
    if not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers harus berupa integer positif")

    def answer_question(definition):
        kind = definition["type"]
        instructions = definition.get("instructions")
        criteria = definition.get("criteria")

        if kind == "noul":
            answer = classifier.noul(
                state,
                instructions,
                criteria,
                image_paths=image_paths,
                use_cache=use_cache,
            )
            candidate_count = 1
        elif kind == "choice":
            if not isinstance(criteria, Mapping) or not criteria:
                raise ValueError("choice criteria harus berupa mapping nonempty")
            answer = classifier.choice(
                state,
                instructions,
                criteria,
                image_paths=image_paths,
                use_cache=use_cache,
            )
            candidate_count = len(criteria)
        elif kind == "score":
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise ValueError("score criteria harus berupa list minimal 2 item")
            answer = classifier.score(
                state,
                instructions,
                criteria,
                image_paths=image_paths,
                use_cache=use_cache,
            )
            candidate_count = len(criteria)
        else:
            raise ValueError(f"jenis pertanyaan tidak dikenal: {kind}")
        return answer, candidate_count

    if question_items:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(answer_question, definition)
                for _, definition in question_items
            ]
            answers = {}
            candidate_sequences = 0
            for (question_id, _), future in zip(question_items, futures):
                answer, candidate_count = future.result()
                answers[question_id] = answer
                candidate_sequences += candidate_count
    else:
        answers = {}
        candidate_sequences = 0

    return {
        "answers": answers,
        "model": classifier.model,
        "usage": {
            "candidate_sequences": candidate_sequences,
        },
        "metadata": {
            "method": (
                "vllm_multimodal_prompt_logprob"
                if image_paths is not None
                else "vllm_yes_no_logprob"
            ),
            "decision_temperature": classifier.decision_temperature,
            "inference_seconds": time.perf_counter() - started,
        },
    }
