import importlib.util
import math
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intercoder import (
    TASK_COLUMNS,
    add_item_agreement_columns,
    align_overlap,
    calculate_reliability,
    cohen_kappa,
    sha256_file,
)


def load_numbered_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def coder_frame(sample_ids: list[str], *, sentiment: list[str] | None = None) -> pd.DataFrame:
    sentiment = sentiment or ["negative"] * len(sample_ids)
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "analysis_unit_id": [f"AU{index:04d}" for index in range(len(sample_ids))],
            "case_id": ["01"] * len(sample_ids),
            "case_name": ["case"] * len(sample_ids),
            "comment_type": ["top_level"] * len(sample_ids),
            "parent_text": [""] * len(sample_ids),
            "analysis_text": [f"comment {index}" for index in range(len(sample_ids))],
            "sentiment": sentiment,
            "target": ["league"] * len(sample_ids),
            "stance": ["blame"] * len(sample_ids),
            "is_sarcasm_mockery": ["false"] * len(sample_ids),
        }
    )


class ReliabilityTests(unittest.TestCase):
    def test_perfect_agreement_is_one(self):
        rows = []
        labels = {
            "sentiment": ["positive", "neutral", "negative"],
            "target": ["referee", "league", "player"],
            "stance": ["blame", "support", "neutral_fact"],
            "is_sarcasm_mockery": ["true", "false", "true"],
        }
        for index in range(3):
            row = {}
            for task, values in labels.items():
                for coder in ("coder1", "coder2", "coder3"):
                    row[f"{coder}_{task}"] = values[index]
            rows.append(row)
        summary, pairwise = calculate_reliability(
            pd.DataFrame(rows), ["coder1", "coder2", "coder3"]
        )
        self.assertTrue((summary["unanimous_agreement"] == 1).all())
        self.assertTrue((summary["fleiss_kappa"].round(8) == 1).all())
        self.assertTrue((summary["krippendorff_alpha_nominal"].round(8) == 1).all())
        self.assertTrue((pairwise["cohen_kappa"].round(8) == 1).all())

    def test_cohen_kappa_known_example(self):
        value = cohen_kappa(["A", "A", "B", "B"], ["A", "B", "B", "B"])
        self.assertAlmostEqual(value, 0.5)

    def test_constant_labels_report_undefined_kappa(self):
        row = {
            f"{coder}_{task}": (
                "false"
                if task == "is_sarcasm_mockery"
                else {"sentiment": "negative", "target": "league", "stance": "blame"}[task]
            )
            for task in TASK_COLUMNS
            for coder in ("coder1", "coder2", "coder3")
        }
        summary, pairwise = calculate_reliability(
            pd.DataFrame([row, row]), ["coder1", "coder2", "coder3"]
        )
        self.assertTrue(summary["fleiss_kappa"].apply(math.isnan).all())
        self.assertTrue(pairwise["cohen_kappa"].apply(math.isnan).all())

    def test_overlap_requires_exact_same_manifest_ids(self):
        coder1 = coder_frame(["S1", "S2", "S3"])
        coder2 = coder_frame(["S1", "S2"])
        coder3 = coder_frame(["S1", "S2", "S3"])
        with self.assertRaisesRegex(ValueError, "매니페스트와 다릅니다"):
            align_overlap(
                {"coder1": coder1, "coder2": coder2, "coder3": coder3},
                ["S1", "S2"],
            )


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_numbered_module("04_create_coder_workbooks.py", "coder_workbooks")

    def test_overlap_is_exact_deterministic_and_stratified(self):
        rows = []
        sample_number = 1
        for case_id, comment_type, count in [
            ("01", "top_level", 60),
            ("01", "reply", 40),
            ("02", "top_level", 70),
            ("02", "reply", 50),
            ("03", "top_level", 45),
            ("03", "reply", 35),
        ]:
            for _ in range(count):
                rows.append(
                    {
                        "sample_id": f"S{sample_number:04d}",
                        "case_id": case_id,
                        "comment_type": comment_type,
                    }
                )
                sample_number += 1
        frame = pd.DataFrame(rows)
        first, counts = self.module.select_overlap_sample(frame, 100, 42)
        second, _ = self.module.select_overlap_sample(frame, 100, 42)
        self.assertEqual(len(first), 100)
        self.assertEqual(first["sample_id"].tolist(), second["sample_id"].tolist())
        self.assertEqual(first["sample_id"].nunique(), 100)
        self.assertTrue((counts["sample_n"] >= 1).all())
        self.assertEqual(int(counts["sample_n"].sum()), 100)


class AdjudicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_numbered_module("05_finalize_gold_labels.py", "finalize_gold")

    def build_inputs(self):
        coder1 = coder_frame(["S1", "S2", "S3"], sentiment=["negative", "neutral", "positive"])
        overlap = pd.DataFrame(
            {
                "sample_id": ["S1", "S2"],
                "analysis_unit_id": ["AU0000", "AU0001"],
                "case_id": ["01", "01"],
                "case_name": ["case", "case"],
                "comment_type": ["top_level", "top_level"],
                "parent_text": ["", ""],
                "analysis_text": ["comment 0", "comment 1"],
            }
        )
        for task in TASK_COLUMNS:
            base_values = {
                "sentiment": ["negative", "neutral"],
                "target": ["league", "league"],
                "stance": ["blame", "blame"],
                "is_sarcasm_mockery": ["false", "false"],
            }[task]
            overlap[f"coder1_{task}"] = base_values
            overlap[f"coder2_{task}"] = base_values
            overlap[f"coder3_{task}"] = base_values
        overlap.loc[0, "coder2_sentiment"] = "neutral"
        items = add_item_agreement_columns(overlap, ["coder1", "coder2", "coder3"])
        decisions = pd.DataFrame(
            {
                "sample_id": ["S1"],
                "final_sentiment": ["neutral"],
                "final_target": ["league"],
                "final_stance": ["blame"],
                "final_is_sarcasm_mockery": ["false"],
            }
        )
        return coder1, items, decisions

    def test_only_disagreement_changes_and_sources_are_recorded(self):
        coder1, items, decisions = self.build_inputs()
        decisions = self.module.normalize_and_validate_decisions(decisions, items)
        output = self.module.build_gold_labels(coder1, items, decisions).set_index("sample_id")
        self.assertEqual(output.loc["S1", "gold_sentiment"], "neutral")
        self.assertEqual(output.loc["S1", "gold_sentiment_source"], "adjudicated_3_coders")
        self.assertEqual(output.loc["S2", "gold_sentiment_source"], "unanimous_3_coders")
        self.assertEqual(output.loc["S3", "gold_sentiment_source"], "coder1_only")

    def test_unanimous_value_cannot_be_changed_during_adjudication(self):
        _, items, decisions = self.build_inputs()
        decisions.loc[0, "final_target"] = "player"
        with self.assertRaisesRegex(ValueError, "이미 일치한 target"):
            self.module.normalize_and_validate_decisions(decisions, items)


class TimeDistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_numbered_module(
            "12_analyze_comment_time_distribution.py", "time_distribution"
        )

    def test_event_day_bin_boundaries(self):
        offsets = [0, 7, 8, 30, 31, 90, 91, 365, 366]
        event = pd.Timestamp("2022-07-13T00:00:00Z")
        frame = pd.DataFrame(
            {
                "analysis_unit_id": [f"A{index}" for index in range(len(offsets))],
                "case_id": ["01"] * len(offsets),
                "case_name": ["case"] * len(offsets),
                "comment_type": ["top_level"] * len(offsets),
                "published_at": [
                    (event + pd.Timedelta(days=offset)).isoformat() for offset in offsets
                ],
                "collected_at_utc": ["2026-07-13T00:00:00Z"] * len(offsets),
            }
        )
        result = self.module.prepare_time_data(frame)
        self.assertEqual(
            result["time_period"].astype(str).tolist(),
            [
                "0-7일",
                "0-7일",
                "8-30일",
                "8-30일",
                "31-90일",
                "31-90일",
                "91-365일",
                "91-365일",
                "366일 이상",
            ],
        )

    def test_period_percent_sums_to_100(self):
        frame = pd.DataFrame(
            {
                "analysis_unit_id": ["A1", "A2"],
                "case_id": ["01", "01"],
                "case_name": ["case", "case"],
                "comment_type": ["top_level", "reply"],
                "published_at": ["2022-07-13T00:00:00Z", "2023-07-14T00:00:00Z"],
                "collected_at_utc": ["2026-07-13T00:00:00Z"] * 2,
            }
        )
        outputs = self.module.analyze_comment_time_distribution(frame)
        self.assertAlmostEqual(float(outputs["period"]["percent"].sum()), 100.0)
        self.assertEqual(int(outputs["period"]["n"].sum()), 2)


class WorkflowIntegrationTests(unittest.TestCase):
    def test_coder1_is_preserved_through_generation_reliability_and_adjudication(self):
        create_module = load_numbered_module("04_create_coder_workbooks.py", "create_integration")
        reliability_module = load_numbered_module(
            "05_analyze_intercoder_reliability.py", "reliability_integration"
        )
        finalize_module = load_numbered_module(
            "05_finalize_gold_labels.py", "finalize_integration"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            paths = {
                "coder1": directory / "coder1_coding.xlsx",
                "coder2": directory / "coder2_coding.xlsx",
                "coder3": directory / "coder3_coding.xlsx",
                "overlap": directory / "intercoder_sample.csv",
                "manifest": directory / "intercoder_sample_manifest.json",
                "summary": directory / "summary.csv",
                "pairwise": directory / "pairwise.csv",
                "report": directory / "reliability.xlsx",
                "items": directory / "items.csv",
                "adjudication": directory / "adjudication.xlsx",
                "gold_csv": directory / "gold.csv",
                "gold_xlsx": directory / "gold.xlsx",
            }
            coder1 = coder_frame(
                [f"S{index:04d}" for index in range(1, 301)],
                sentiment=["negative", "neutral", "positive"] * 100,
            )
            coder1.to_excel(paths["coder1"], index=False, sheet_name="Coding")
            original_hash = sha256_file(paths["coder1"])

            create_patches = {
                "CODER1_PATH": paths["coder1"],
                "CODER2_PATH": paths["coder2"],
                "CODER3_PATH": paths["coder3"],
                "OVERLAP_PATH": paths["overlap"],
                "MANIFEST_PATH": paths["manifest"],
            }
            with ExitStack() as stack:
                for name, value in create_patches.items():
                    stack.enter_context(patch.object(create_module, name, value))
                overlap = create_module.generate_coder_workbooks()
            self.assertEqual(len(overlap), 100)
            self.assertEqual(sha256_file(paths["coder1"]), original_hash)

            coder1_lookup = coder1.set_index("sample_id")
            for coder_name in ["coder2", "coder3"]:
                coding = pd.read_excel(paths[coder_name], sheet_name="Coding", dtype=str).fillna("")
                for task in TASK_COLUMNS:
                    coding[task] = coder1_lookup.loc[coding["sample_id"], task].to_numpy()
                coding.to_excel(paths[coder_name], index=False, sheet_name="Coding")
            coder2 = pd.read_excel(paths["coder2"], sheet_name="Coding", dtype=str).fillna("")
            coder2.loc[0, "sentiment"] = (
                "positive" if coder2.loc[0, "sentiment"] != "positive" else "negative"
            )
            coder2.to_excel(paths["coder2"], index=False, sheet_name="Coding")

            reliability_patches = {
                "CODER_PATHS": {name: paths[name] for name in ["coder1", "coder2", "coder3"]},
                "MANIFEST_PATH": paths["manifest"],
                "SUMMARY_CSV": paths["summary"],
                "PAIRWISE_CSV": paths["pairwise"],
                "REPORT_XLSX": paths["report"],
                "ITEMS_CSV": paths["items"],
                "ADJUDICATION_XLSX": paths["adjudication"],
            }
            with ExitStack() as stack:
                for name, value in reliability_patches.items():
                    stack.enter_context(patch.object(reliability_module, name, value))
                summary, _, adjudication = reliability_module.analyze_intercoder_reliability()
            self.assertEqual(len(summary), 4)
            self.assertEqual(len(adjudication), 1)

            decisions = pd.read_excel(
                paths["adjudication"], sheet_name="Adjudication", dtype=str
            ).fillna("")
            disputed_id = decisions.loc[0, "sample_id"]
            decisions.loc[0, "final_sentiment"] = coder1_lookup.loc[disputed_id, "sentiment"]
            decisions.to_excel(paths["adjudication"], index=False, sheet_name="Adjudication")

            finalize_patches = {
                "CODER_PATHS": {name: paths[name] for name in ["coder1", "coder2", "coder3"]},
                "MANIFEST_PATH": paths["manifest"],
                "ADJUDICATION_PATH": paths["adjudication"],
                "OUTPUT_CSV_PATH": paths["gold_csv"],
                "OUTPUT_XLSX_PATH": paths["gold_xlsx"],
            }
            with ExitStack() as stack:
                for name, value in finalize_patches.items():
                    stack.enter_context(patch.object(finalize_module, name, value))
                gold = finalize_module.finalize_gold_labels()
            self.assertEqual(len(gold), 300)
            self.assertEqual(sha256_file(paths["coder1"]), original_hash)
            self.assertEqual(
                gold.set_index("sample_id").loc[disputed_id, "gold_sentiment_source"],
                "adjudicated_3_coders",
            )


if __name__ == "__main__":
    unittest.main()
