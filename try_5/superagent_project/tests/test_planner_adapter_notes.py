"""What the planner is told about the embedding adapter.

Two properties matter and pull against each other:

1. The planner must know the adapter exists and behaves differently from
   TF-IDF, or it will never choose it. In try_2 a newly registered adapter was
   tested and never selected once.
2. The note must describe a MECHANISM, not a task. Choosing an adapter because
   of a task or dataset name is prohibited by the assignment, and the planner is
   the one component that could do it.

So the tests below check both: the note is present and informative, AND it
contains no task identity and no "use this for X" phrase.
"""
import json
import unittest

from superagent.llm.experiment_planner import LLMExperimentPlanner


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"content": '{"action":"stop","reason":"test"}',
                "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0},
                "cost": {"total_cost_rmb": "0.00000000"}}


def build_planner():
    return LLMExperimentPlanner(
        llm_client=FakeClient(),
        temperature=0.1,
        max_tokens=2048,
        max_response_attempts=1,
    )


TEXT_SPECS = {
    "text_word_tfidf_logistic": {"modality": "text", "parameters": {}},
    "text_embedding_logistic": {"modality": "text", "parameters": {}},
}

PAIR_SPECS = {
    "text_pair_word_tfidf_logistic": {"modality": "text_pair", "parameters": {}},
    "text_pair_embedding_logistic": {"modality": "text_pair", "parameters": {}},
}


def system_message(specs):
    planner = build_planner()
    messages = planner._build_messages(
        data_profile={"modality": "text"},
        experiment_history=[],
        existing_plans=[],
        remaining_budget_rmb="20.00000000",
        next_experiment_id="exp_001",
        adapter_specs=specs,
    )
    return messages[0]["content"]


class MechanismNoteTest(unittest.TestCase):
    def test_note_appears_for_the_single_text_adapter(self):
        text = system_message(TEXT_SPECS)
        self.assertIn("MECHANISM NOTES", text)
        self.assertIn("text_embedding_logistic", text)
        self.assertIn("768", text)

    def test_note_appears_for_the_pair_adapter(self):
        text = system_message(PAIR_SPECS)
        self.assertIn("text_pair_embedding_logistic", text)
        self.assertIn("A*B", text)

    def test_note_is_absent_when_the_adapter_is_not_offered(self):
        text = system_message(
            {"text_word_tfidf_logistic": {"modality": "text", "parameters": {}}})
        self.assertNotIn("MECHANISM NOTES", text)

    def test_note_warns_about_the_one_off_encoding_cost(self):
        """The planner must know the first run is slow, or it will stop early."""
        text = system_message(TEXT_SPECS)
        self.assertIn("one-off", text)
        self.assertIn("cache", text)

    def test_guidance_prefers_a_different_representation(self):
        text = system_message(TEXT_SPECS)
        self.assertIn("FEATURE REPRESENTATION", text)


class NoTaskIdentityTest(unittest.TestCase):
    """The note must not name any task, dataset, or "use for X" rule."""

    FORBIDDEN = [
        "task1", "task2", "task 1", "task 2",
        "coupon", "headline", "covertype", "kdd", "newsgroup", "fashion",
        "bank marketing", "breast cancer", "wine",
        # "use this when the task is..." style rules are equally disallowed.
        "for text tasks", "for this task", "if the task is",
    ]

    def test_note_contains_no_task_or_dataset_name(self):
        text = system_message({**TEXT_SPECS, **PAIR_SPECS})
        lowered = text.lower()
        for needle in self.FORBIDDEN:
            self.assertNotIn(needle, lowered,
                             f"planner prompt leaks task identity: {needle!r}")

    def test_measurements_are_described_as_generic_effect_sizes(self):
        """Effect sizes may be quoted, but not attributed to a named task."""
        text = system_message(TEXT_SPECS)
        self.assertNotIn("+3.83pp", text)
        self.assertNotIn("0.7213", text)


if __name__ == "__main__":
    unittest.main()
