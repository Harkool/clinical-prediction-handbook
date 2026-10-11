"""Known-case lookup and binary positive-probability contracts."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from cph.predictions import FrozenPredictions, metadata_path, positive_class_probability


ROOT = Path(__file__).absolute().parents[1]
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class PredictionPersistenceContracts(unittest.TestCase):
    def setUp(self):
        parent=ROOT / "outputs" / "test_predictions"
        parent.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix="cph-frozen-",dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.path=Path(self.temporary.name) / "known_cases.csv"

    def test_round_trip_preserves_string_ids_source_and_identity(self):
        original=FrozenPredictions(["001","病人,甲"],[0.,1.],source="explicit held-out neural network probabilities",model_identity="model-sha256-example")
        original.save(self.path)
        loaded=FrozenPredictions.load(self.path)
        self.assertEqual(loaded,original)
        self.assertEqual(loaded.prediction_identity,original.prediction_identity)
        self.assertEqual(loaded.ids[0],"001")
        self.assertTrue(metadata_path(self.path).is_file())

    def test_ids_and_probabilities_reject_ambiguous_inputs(self):
        for identifiers,probabilities in (
                ([],[]),(["a","a"],[.1,.2]),([""],[.1]),([" a"],[.1]),
                ([1],[.1]),("case1",[.1]),(["a"],[float("nan")]),
                (["a"],[float("inf")]),(["a"],[-.1]),(["a"],[1.1]),
                (["a"],[[.2]]),(["a"],[True]),(["a"],["0.2"]),
                (["a","b"],[.2])):
            with self.subTest(ids=identifiers,probabilities=probabilities),self.assertRaises(ValueError):
                FrozenPredictions(identifiers,probabilities)

    def test_old_manifest_cannot_accept_rewritten_predictions(self):
        old=FrozenPredictions(["a","b"],[.2,.8],source="first model")
        old.save(self.path)
        old_metadata=metadata_path(self.path).read_bytes()
        new=FrozenPredictions(["a","b"],[.3,.7],source="second model")
        new.save(self.path)
        self.assertEqual(FrozenPredictions.load(self.path),new)
        metadata_path(self.path).write_bytes(old_metadata)
        with self.assertRaises(ValueError):
            FrozenPredictions.load(self.path)

    def test_changes_to_probability_order_source_or_identity_are_detected(self):
        stored=FrozenPredictions(["a","b"],[.2,.8],source="saved output")
        for change in ("probability","order","source","identity","count","missing_metadata_field"):
            with self.subTest(change=change):
                stored.save(self.path)
                if change=="probability":
                    self.path.write_text("id,probability\na,0.3\nb,0.8\n",encoding="utf-8")
                elif change=="order":
                    self.path.write_text("id,probability\nb,0.8\na,0.2\n",encoding="utf-8")
                else:
                    manifest=json.loads(metadata_path(self.path).read_text(encoding="utf-8"))
                    if change=="source": manifest['source']="another output"
                    elif change=="identity": manifest['prediction_identity']="SHA256:bad"
                    elif change=="count": manifest['n']=3
                    else: del manifest['model_identity']
                    metadata_path(self.path).write_text(json.dumps(manifest),encoding="utf-8")
                with self.assertRaises(ValueError):
                    FrozenPredictions.load(self.path)

    def test_even_consistent_file_hash_does_not_allow_duplicate_ids(self):
        FrozenPredictions(["a","b"],[.2,.8]).save(self.path)
        payload=b"id,probability\na,0.2\na,0.8\n"
        self.path.write_bytes(payload)
        manifest=json.loads(metadata_path(self.path).read_text(encoding="utf-8"))
        manifest['csv_sha256']=hashlib.sha256(payload).hexdigest()
        metadata_path(self.path).write_text(json.dumps(manifest),encoding="utf-8")
        with self.assertRaises(ValueError):
            FrozenPredictions.load(self.path)


@unittest.skipUnless(HAS_NUMPY,"Install requirements.txt for NumPy probability-array contracts")
class PositiveProbabilityContracts(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np=np

    def test_lookup_uses_requested_ids_and_rejects_unseen_cases(self):
        frozen=FrozenPredictions(["001","002","003"],[.1,.2,.3])
        self.np.testing.assert_array_equal(frozen.predict(["003","001"]),[.3,.1])
        for requested in (["004"],["001","001"],[],"001"):
            with self.subTest(requested=requested),self.assertRaises(ValueError):
                frozen.predict(requested)

    def test_returned_array_cannot_change_frozen_probabilities(self):
        frozen=FrozenPredictions(["a"],[.2])
        result=frozen.predict(["a"])
        result[0]=.9
        self.assertEqual(frozen.probabilities,(.2,))

    def model(self,classes,probabilities):
        class Model:
            classes_=self.np.array(classes)
            def predict_proba(instance,X):
                return self.np.array(probabilities)
        return Model()

    def test_positive_column_is_found_by_label_even_if_classes_are_reversed(self):
        rows=[[.9,.1],[.2,.8]]
        self.np.testing.assert_array_equal(positive_class_probability(self.model([0,1],rows),[[1],[2]]),[.1,.8])
        self.np.testing.assert_array_equal(positive_class_probability(self.model([1,0],rows),[[1],[2]]),[.9,.2])

    def test_invalid_class_mapping_shape_nan_and_row_sum_fail(self):
        for classes,probabilities in (
                ([1,2],[[.9,.1]]),(["0","1"],[[.9,.1]]),([0,1,2],[[.4,.3,.3]]),
                ([0,0],[[.9,.1]]),([0,1],[.9,.1]),([0,1],[[.9,float('nan')]]),
                ([0,1],[[.9,.9]]),([0,1],[[-.1,1.1]]),([0,1],[[.9,.1],[.1,.9]])):
            with self.subTest(classes=classes,probabilities=probabilities),self.assertRaises(ValueError):
                positive_class_probability(self.model(classes,probabilities),[[1]])

    def test_explicit_neural_network_probability_vector_needs_no_model_class_or_keras(self):
        probabilities=self.np.array([.2,.8])
        stored=FrozenPredictions(["left","right"],probabilities,source="explicit held-out Keras probabilities")
        self.np.testing.assert_array_equal(stored.predict(["right","left"]),[.8,.2])
        with self.assertRaises(ValueError):
            positive_class_probability(object(),[[1]])


if __name__=="__main__":
    unittest.main()
