import unittest
from pipeline_uasz_ai import prepare_student_orientation_dataset, train_and_evaluate


class TestUaszAiMlPipeline(unittest.TestCase):
    def test_preprocessing_and_evaluation(self) -> None:
        df = prepare_student_orientation_dataset(n_samples=400, random_state=42)
        self.assertEqual(len(df), 400)
        results = train_and_evaluate(df, random_state=42)
        self.assertGreaterEqual(results["random_forest"]["accuracy"], 0.80)
        self.assertGreaterEqual(results["random_forest"]["f1_score"], 0.75)


if __name__ == "__main__":
    unittest.main()
