"""Fast F005 preregistration and model/retention boundary tests."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import unittest

from speaker_id.training import f005_contract as contract
from speaker_id.training import f005_worker as worker


ROOT = Path(__file__).resolve().parents[1]


class F005ContractTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "configs/train/campp_f005_consistency.json").read_text(encoding="utf-8"))

    def test_exact_config_pins_advanced192_and_forbids_weight_tracking(self):
        config = self.config()
        contract.validate_f005_config(config)
        self.assertEqual(config["advanced_model_config"], "configs/model/campp_advanced.json")
        self.assertEqual(config["mlflow"]["experiment_id"], "1")
        self.assertTrue({"model_weights", "optimizer_state", "embeddings"}.issubset(config["mlflow"]["forbidden"]))
        self.assertEqual(config["retention"]["local_transfer"], "promotion_only")
        self.assertEqual(config["fit"]["shared_head_stage"],
                         "run_once_per_outer_then_fork_byte_identical_checkpoint_to_all_four_tails")

    def test_any_protocol_mutation_or_extra_key_is_a_new_experiment(self):
        changes = [
            lambda value: value.update(extra=True),
            lambda value: value["arms"][1].update(cosine_gamma=0.4),
            lambda value: value["fit"].update(mixed_precision=True),
            lambda value: value["fit"].update(consistency_ramp_tail_steps=99),
            lambda value: value["fit"].update(waveform_cache_max_bytes=0),
            lambda value: value["execution"].update(cublas_workspace_config=":16:8"),
            lambda value: value["scoring"].update(protocol="crossfit_scores"),
            lambda value: value["arm_selection"].update(control_is_selectable=False),
            lambda value: value["bootstrap"].update(true_class_purity_assumed=True),
            lambda value: value["promotion"].update(minimum_treatment_delta_vs_fresh_control=.002),
            lambda value: value["mlflow"]["forbidden"].remove("model_weights"),
        ]
        for mutate in changes:
            with self.subTest(mutate=mutate):
                value = deepcopy(self.config()); mutate(value)
                with self.assertRaises(ValueError):
                    contract.validate_f005_config(value)

    def test_training_loader_is_authoritative_advanced_not_public_loader(self):
        source = inspect.getsource(worker._load_advanced_trainable)
        self.assertIn("load_advanced", source)
        self.assertNotIn("load_campp", source)
        self.assertIn("advanced CAM++ 192D", source)
        all_source = inspect.getsource(worker)
        self.assertNotIn("tracker.add_artifact(checkpoint", all_source)
        self.assertNotIn("warn_only=True", all_source)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", all_source)
        self.assertIn("audio_hashes_checked", all_source)
        self.assertIn("VAST_INSTANCE_ID", all_source)


if __name__ == "__main__":
    unittest.main()
