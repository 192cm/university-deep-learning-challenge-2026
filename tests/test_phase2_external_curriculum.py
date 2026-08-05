from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from phase2_common import jaccard, token_shingles  # noqa: E402
from prepare_external_curriculum import LocalContaminationIndex  # noqa: E402


class ExternalCurriculumTests(unittest.TestCase):
    def test_rare_shingle_prefilter_matches_brute_force_jaccard(self) -> None:
        rng = random.Random(20260804)
        vocabulary = [f"token{index}" for index in range(1000)]
        rows: list[dict[str, str]] = []
        token_rows: list[list[str]] = []
        for index in range(200):
            tokens = rng.sample(vocabulary, rng.randint(10, 50))
            token_rows.append(tokens)
            rows.append({"id": f"row-{index}", "question": " ".join(tokens)})

        threshold = 0.86
        index = LocalContaminationIndex(rows, threshold)
        queries: list[str] = []
        for tokens in token_rows[:100]:
            changed = list(tokens)
            for position in rng.sample(range(len(changed)), rng.randint(0, min(5, len(changed)))):
                changed[position] = rng.choice(vocabulary)
            queries.append(" ".join(changed))
        queries.extend(" ".join(rng.sample(vocabulary, 30)) for _ in range(50))

        protected = [token_shingles(row["question"]) for row in rows]
        for query in queries:
            query_shingles = token_shingles(query)
            brute_score = max(jaccard(query_shingles, candidate) for candidate in protected)
            match_type, _, score = index.match(query)
            self.assertEqual(match_type is not None, brute_score >= threshold)
            if match_type is not None:
                self.assertAlmostEqual(score, brute_score)


if __name__ == "__main__":
    unittest.main()
