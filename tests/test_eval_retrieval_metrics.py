from __future__ import annotations

import unittest

from src.eval import (
    evaluate_retrieval,
    hit_rate_at_k,
    mean_reciprocal_rank_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


class RetrievalMetricsTests(unittest.TestCase):
    def test_recall_precision_hit_rate_and_rr_at_k(self) -> None:
        relevant_ids = ["c2", "c4"]
        retrieved_ids = ["c1", "c2", "c3", "c4", "c5"]

        self.assertEqual(recall_at_k(relevant_ids, retrieved_ids, k=3), 0.5)
        self.assertEqual(recall_at_k(relevant_ids, retrieved_ids, k=5), 1.0)
        self.assertAlmostEqual(
            precision_at_k(relevant_ids, retrieved_ids, k=3),
            1 / 3,
        )
        self.assertEqual(hit_rate_at_k(relevant_ids, retrieved_ids, k=1), 0.0)
        self.assertEqual(hit_rate_at_k(relevant_ids, retrieved_ids, k=2), 1.0)
        self.assertEqual(
            reciprocal_rank_at_k(relevant_ids, retrieved_ids, k=5),
            0.5,
        )

    def test_duplicate_retrieved_ids_do_not_inflate_scores(self) -> None:
        relevant_ids = ["c2"]
        retrieved_ids = ["c2", "c2", "c2"]

        self.assertEqual(recall_at_k(relevant_ids, retrieved_ids, k=3), 1.0)
        self.assertAlmostEqual(
            precision_at_k(relevant_ids, retrieved_ids, k=3),
            1 / 3,
        )
        self.assertEqual(
            reciprocal_rank_at_k(relevant_ids, retrieved_ids, k=3),
            1.0,
        )

    def test_empty_relevant_or_retrieved_ids_score_zero(self) -> None:
        self.assertEqual(recall_at_k([], ["c1"], k=5), 0.0)
        self.assertEqual(precision_at_k(["c1"], [], k=5), 0.0)
        self.assertEqual(hit_rate_at_k([], ["c1"], k=5), 0.0)
        self.assertEqual(reciprocal_rank_at_k(["c1"], [], k=5), 0.0)

    def test_mean_reciprocal_rank_at_k(self) -> None:
        rows = [
            (["c2"], ["c1", "c2"]),
            (["c1"], ["c1", "c2"]),
            (["c9"], ["c1", "c2"]),
        ]

        self.assertAlmostEqual(
            mean_reciprocal_rank_at_k(rows, k=2),
            (0.5 + 1.0 + 0.0) / 3,
        )

    def test_evaluate_retrieval_returns_macro_averages_and_per_query_metrics(
        self,
    ) -> None:
        summary = evaluate_retrieval(
            {
                "q1": (["c2", "c4"], ["c1", "c2", "c3", "c4"]),
                "q2": (["c1"], ["c1", "c2"]),
            },
            k_values=[3, 1, 3],
        )

        self.assertEqual(summary.query_count, 2)
        self.assertEqual(summary.k_values, (1, 3))
        self.assertEqual(summary.per_query["q1"].recall_at_k[1], 0.0)
        self.assertEqual(summary.per_query["q2"].recall_at_k[1], 1.0)
        self.assertEqual(summary.recall_at_k[1], 0.5)
        self.assertAlmostEqual(summary.recall_at_k[3], 0.75)
        self.assertAlmostEqual(summary.precision_at_k[3], ((1 / 3) + (1 / 3)) / 2)
        self.assertEqual(summary.hit_rate_at_k[1], 0.5)
        self.assertEqual(summary.mean_reciprocal_rank_at_k[1], 0.5)

    def test_invalid_k_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            recall_at_k(["c1"], ["c1"], k=0)

        with self.assertRaises(ValueError):
            evaluate_retrieval({"q1": (["c1"], ["c1"])}, k_values=[])


if __name__ == "__main__":
    unittest.main()
