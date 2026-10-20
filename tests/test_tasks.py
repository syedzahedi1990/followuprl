import random
import unittest

from paper2.environments.belief import Episode, Move, State, Task, simulate, transition
from paper2.environments.rendering import render_prompt
from paper2.evaluation.independent_oracle import gold_answer, gold_trace
from paper2.evaluation.suites import validate_cpu


def fixture():
    return Episode(2, (
        Move(4, True, False), Move(1, False, True),
        Move(3, False, False), Move(0, True, False), Move(5, False, True),
    ))


class TaskTests(unittest.TestCase):
    def test_fixture(self):
        episode = fixture()
        self.assertEqual(simulate(episode), State(5, 0, 5))
        self.assertEqual(gold_trace(Task(episode)), (2, 4, 4, 4, 0, 0))
        self.assertEqual(gold_answer(Task(episode, query="B")), 5)
        self.assertEqual(gold_answer(Task(episode, query="world", family="world")), 5)

    def test_exhaustive_and_random_independent_oracles(self):
        result = validate_cpu()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["oracle_episodes_checked"], 4606)
        self.assertEqual(result["reward_table_cells"], 48)

    def test_event_changes_world_even_unobserved(self):
        self.assertEqual(transition(State(1, 2, 3), Move(5, False, False)), State(5, 2, 3))

    def test_schema_rejects_boolean_locations_and_bad_flags(self):
        for value in (True, 6, -1, "2", 2.0):
            with self.assertRaises(ValueError):
                Episode(value, ())
        with self.assertRaises(ValueError):
            Move(2, 1, False)

    def test_serialisation_and_immutable_identity(self):
        task = Task(fixture())
        self.assertEqual(Task.from_dict(task.to_dict()), task)
        variant = Task(task.episode, variant=1)
        other_query = Task(task.episode, query="B")
        self.assertEqual(task.episode_id, variant.episode_id)
        self.assertEqual(task.group_id, other_query.group_id)
        self.assertNotEqual(task.task_id, variant.task_id)
        self.assertNotEqual(task.task_id, other_query.task_id)
        self.assertEqual(Task(family="arithmetic", operands=(2, 7)).episode_id,
                         Task(family="arithmetic", operands=(7, 2)).episode_id)

    def test_rendering_contains_only_facts(self):
        task = Task(fixture(), sibling_group="SECRET_ANCESTOR_DO_NOT_RENDER")
        prompt = render_prompt(task)
        self.assertIn("Where does A believe", prompt)
        for forbidden in (task.task_id, task.episode_id, task.sibling_group,
                          "gold", "seed", "split", "correct answer"):
            self.assertNotIn(forbidden, prompt)
        self.assertEqual(prompt, render_prompt(Task(fixture())))
        trace_prompt = render_prompt(task, "trace")
        self.assertIn("exactly 6 integers", trace_prompt)
        self.assertNotIn("[2, 4, 4, 4, 0, 0]", trace_prompt)

    def test_arithmetic_all_answers_valid(self):
        rng = random.Random(17)
        for _ in range(100):
            operands = tuple(rng.randrange(96) for _ in range(2))
            task = Task(family="arithmetic", operands=operands)
            self.assertIn(gold_answer(task), range(6))
            self.assertIn("modulo 6", render_prompt(task))
            with self.assertRaises(ValueError):
                render_prompt(task, "trace")


if __name__ == "__main__":
    unittest.main()
