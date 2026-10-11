# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Map Decisions questions onto the /v1/systemone question types, and their
answers back."""

import json

from .protocol import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionInputMessage,
    DecisionQuestion,
    PredicateAnswer,
    PredicateQuestion,
    ScoreAnswer,
)
from .question_types import (
    LABELS,
    Option,
    Question,
    get_question_type,
    make_question,
)

QUESTION_TYPE_NAMES = {"predicate": "noul", "choice": "choice", "score": "score"}


def make_read_question(
    index: int, question: DecisionQuestion, max_options: int = len(LABELS)
) -> Question:
    qid = str(index)
    qtype = get_question_type(QUESTION_TYPE_NAMES[question.type])
    if isinstance(question, PredicateQuestion):
        options = qtype.parse_options(qid, None)
    elif isinstance(question, ChoiceQuestion):
        options = [
            Option(json.dumps(c.value, ensure_ascii=False), c.description)
            for c in question.choices
        ]
    else:
        options = [Option(level.label, level.description) for level in question.levels]
    return make_question(qid, qtype, question.instructions, options, max_options)


def make_answer(
    question: DecisionQuestion,
    read_question: Question,
    probs: list[float],
    label_mass: float,
) -> DecisionAnswer:
    answer = read_question.type.answer(read_question, probs, label_mass)
    if isinstance(question, PredicateQuestion):
        return PredicateAnswer(name=question.name, probability=answer["noul"])
    if isinstance(question, ChoiceQuestion):
        values = [choice.value for choice in question.choices]
        names = [option.name for option in read_question.options]
        return ChoiceAnswer(
            name=question.name,
            choice=values[names.index(answer["choice"])],
            probabilities=[
                {"value": value, "probability": probability}
                for value, probability in zip(values, answer["probabilities"].values())
            ],
            confidence=answer["confidence"],
        )
    return ScoreAnswer(
        name=question.name,
        score=answer["score"],
        probabilities=[
            {"value": int(i), "label": level.label, "probability": probability}
            for (i, probability), level in zip(
                answer["probabilities"].items(), question.levels
            )
        ],
        confidence=answer["confidence"],
    )


def input_text(input: str | list[DecisionInputMessage]) -> str:
    if isinstance(input, str):
        return input
    return "\n\n".join(
        message.content
        if isinstance(message.content, str)
        else "\n".join(part.text for part in message.content)
        for message in input
    )
