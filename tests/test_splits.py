import copy
import json
from pathlib import Path
import tempfile
import unittest

from paper2.environments.belief import Episode, Move, Task
from paper2.environments.rendering import render_prompt
from paper2.experiments.data import (
    freeze_final_protocol, is_structural_holdout, load_tasks,
    prepare_dataset, validate_splits,
)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "dataset"
        self.config = {
            "task": {"moves": 3, "protected_fraction": 0.2},
            "data": {"path": str(self.path), "seed": 19,
                     "sizes": {"train": 144, "development": 72,
                               "final_test": 72, "structural_test": 72}},
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_deterministic_immutable_data_and_balances(self):
        first = prepare_dataset(self.config)
        second = prepare_dataset(self.config)
        self.assertEqual(first, second)
        stats = first["splits"]["train"]["statistics"]
        self.assertEqual(stats["distinct_belief_episodes"], 144)
        self.assertEqual(stats["family_counts"], {"belief": 144, "world": 18, "arithmetic": 18})
        self.assertEqual(stats["queried_observers"], {"A": 72, "B": 72})
        self.assertEqual(set(stats["by_family"]["belief"]["gold_counts"].values()), {24})
        self.assertEqual(stats["belief_equals_world_count"], 24)
        self.assertEqual(set(stats["belief_label_world_counts"].values()), {4})
        elsewhere = copy.deepcopy(self.config)
        elsewhere["data"]["path"] = str(Path(self.temporary.name) / "replicate")
        self.assertEqual(prepare_dataset(elsewhere), first)
        changed = copy.deepcopy(self.config)
        changed["data"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "settings changed"):
            prepare_dataset(changed)

    def test_metadata_never_reaches_model(self):
        prepare_dataset(self.config)
        for task in load_tasks(self.path, "train"):
            prompt = render_prompt(task)
            self.assertNotIn(task.task_id, prompt)
            self.assertNotIn(task.episode_id, prompt)
            self.assertNotIn("train", prompt.lower())
            self.assertNotIn(str(self.path), prompt)

    def test_counts_not_divisible_by_72_still_balance_labels_and_observers(self):
        self.config["data"]["sizes"]["train"] = 600
        manifest = prepare_dataset(self.config)
        stats = manifest["splits"]["train"]["statistics"]
        self.assertEqual(stats["queried_observers"], {"A": 300, "B": 300})
        self.assertEqual(set(stats["by_family"]["belief"]["gold_counts"].values()), {100})

    def test_final_access_requires_frozen_plan_and_is_logged(self):
        prepare_dataset(self.config)
        with self.assertRaises(PermissionError):
            load_tasks(self.path, "final_test")
        definition = {
            "evaluation_checkpoints": ["dummy-unittest-checkpoint"],
            "metric_definitions": ["true utility"],
            "practical_effect_threshold": 0.01,
        }
        freeze_final_protocol(self.path, "unit-only-protocol", definition)
        freeze_final_protocol(self.path, "unit-only-protocol", definition)
        tasks = load_tasks(
            self.path, "final_test", protocol_id="unit-only-protocol",
            access_reason="Unit test fixture only; not research final data.",
            evaluation_checkpoint="dummy-unittest-checkpoint",
        )
        self.assertGreater(len(tasks), 0)
        records = [json.loads(x) for x in (self.path / "final_test_access.jsonl").read_text().splitlines()]
        self.assertEqual([record["status"] for record in records], ["denied", "opened"])
        with self.assertRaises(PermissionError):
            load_tasks(self.path, "structural_test", protocol_id="wrong", access_reason="unit test")
        with self.assertRaises(ValueError):
            freeze_final_protocol(self.path, "changed", definition)
        with self.assertRaises(PermissionError):
            load_tasks(self.path, "final_test", protocol_id="unit-only-protocol",
                       access_reason="unit test", evaluation_checkpoint="unfrozen-checkpoint")

    def test_incomplete_freeze_rejected(self):
        prepare_dataset(self.config)
        with self.assertRaises(ValueError):
            freeze_final_protocol(self.path, "x", {})

    def test_split_and_structural_integrity(self):
        prepare_dataset(self.config)
        # Raw deserialisation here is only of tiny disposable unittest fixtures.
        splits = {
            split: [Task.from_dict(json.loads(line)) for line in
                    (self.path / f"{split}.jsonl").read_text().splitlines()]
            for split in ("train", "development", "final_test", "structural_test")
        }
        self.assertGreater(validate_splits(splits)["unique_episode_ids"], 0)
        for split, tasks in splits.items():
            for task in tasks:
                if task.episode is not None:
                    self.assertEqual(is_structural_holdout(task.episode.skeleton),
                                     split == "structural_test")

    def test_variants_queries_and_counterfactual_siblings_cannot_cross_splits(self):
        prepare_dataset(self.config)
        task = next(task for task in load_tasks(self.path, "train") if task.family == "belief")
        variant = Task(task.episode, query=task.query, variant=1)
        query = Task(task.episode, query="B" if task.query == "A" else "A")
        altered = Episode((task.episode.initial + 1) % 6, task.episode.events)
        sibling = Task(altered, query=task.query, sibling_group=task.group_id)
        for counterpart in (variant, query, sibling):
            with self.assertRaises(ValueError):
                validate_splits({"train": [task], "development": [counterpart]})

    def test_corruption_is_not_reused(self):
        prepare_dataset(self.config)
        with (self.path / "train.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaises(ValueError):
            load_tasks(self.path, "train")
        with self.assertRaises(ValueError):
            prepare_dataset(self.config)

    def test_impossible_horizon_pool_is_rejected(self):
        config = copy.deepcopy(self.config)
        config["task"]["moves"] = 1
        config["data"]["sizes"]["train"] = 6000
        with self.assertRaisesRegex(ValueError, "more distinct"):
            prepare_dataset(config)


if __name__ == "__main__":
    unittest.main()
