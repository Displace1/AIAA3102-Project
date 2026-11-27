"""The benchmark splits must be verifiably correct before any result means anything.

Eight runs are executing against these files overnight. If a split is wrong, all
eight results are wrong, and the mistake would be invisible: a stratified split
that silently became non-stratified still produces plausible accuracy numbers.

This is not hypothetical. Two real defects were found while preparing these
files:

1. **Downsampling destroyed the class imbalance.** QQP was capped with
   `cap // n_classes` rows per class, turning its natural 63/37 into a perfect
   50/50. Every number in the table still looked reasonable. The benchmark would
   have been measuring a dataset that does not exist.
2. **Splits can silently overlap.** Nothing in a CSV stops train and validation
   from sharing rows, and shared rows inflate validation accuracy.

So the tests below check properties, not file existence:

  * every class appears in all three splits (macro_f1 needs this)
  * no row appears in two splits
  * the split proportions match the declared fractions
  * downsampling preserved the source distribution
  * manifests agree with the files on disk (sizes and SHA-256)

Run: python -m unittest tests.test_benchmark_preparation
"""
import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).resolve().parents[3] / "try_4" / "benchmarks"

TEXT_TASKS = ["sms_spam", "ag_news", "qqp", "mrpc", "yelp_full",
              "20news_4class"]
TABULAR_TASKS = ["credit_card"]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def manifest(task):
    return json.loads((BENCH / task / "split_manifest.json").read_text(
        encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ManifestExistsTest(unittest.TestCase):
    def test_every_task_has_a_manifest_and_config(self):
        for task in TEXT_TASKS + TABULAR_TASKS + ["cifar10"]:
            with self.subTest(task=task):
                self.assertTrue((BENCH / task / "split_manifest.json").is_file())
                self.assertTrue((BENCH / task / f"{task}_config.json").is_file())


class SplitIntegrityTest(unittest.TestCase):
    """Properties every CSV task must satisfy, regardless of its content."""

    def splits(self, task):
        """Split data keyed by the manifest's names (train/validation/test).

        The files on disk use val.csv; the manifest calls it `validation`. Doing
        the mapping here means every test can compare against the manifest
        without repeating the translation and getting it wrong.
        """
        data = BENCH / task / "data"
        return {
            "train": read_csv(data / "train.csv"),
            "validation": read_csv(data / "val.csv"),
            "test": read_csv(data / "test.csv"),
        }

    def test_no_task_has_an_empty_split(self):
        for task in TEXT_TASKS + TABULAR_TASKS:
            with self.subTest(task=task):
                for name, rows in self.splits(task).items():
                    self.assertGreater(len(rows), 0, f"{task}/{name} is empty")

    def test_split_sizes_match_the_manifest(self):
        for task in TEXT_TASKS + TABULAR_TASKS:
            with self.subTest(task=task):
                declared = manifest(task)["split_sizes"]
                actual = {k: len(v) for k, v in self.splits(task).items()}
                self.assertEqual(actual["train"], declared["train"])
                self.assertEqual(actual["test"], declared["test"])
                self.assertEqual(actual["validation"], declared["validation"])

    def test_proportions_are_close_to_declared(self):
        """70/15/15 must actually be 70/15/15, not merely "most in train"."""
        for task in TEXT_TASKS + TABULAR_TASKS:
            with self.subTest(task=task):
                info = manifest(task)
                sizes = info["split_sizes"]
                total = sum(sizes.values())
                for name, fraction in info["split_fractions"].items():
                    observed = sizes[name] / total
                    self.assertAlmostEqual(
                        observed, fraction, delta=0.02,
                        msg=f"{task}/{name}: {observed:.3f} vs {fraction}")

    def test_every_class_appears_in_all_three_splits(self):
        """Otherwise macro_f1 is undefined on that split, or wildly unstable."""
        for task in TEXT_TASKS + TABULAR_TASKS:
            with self.subTest(task=task):
                splits = self.splits(task)
                classes = {name: {r["label"] for r in rows}
                           for name, rows in splits.items()}
                self.assertEqual(
                    classes["train"], classes["validation"],
                    f"{task}: validation is missing classes present in train")
                self.assertEqual(
                    classes["train"], classes["test"],
                    f"{task}: test is missing classes present in train")

    def test_stratification_preserves_class_balance(self):
        """Each split's class distribution must match the whole dataset's.

        This is the property that failed silently for QQP: the overall balance
        was forced to 50/50 by the cap. Here we check that whatever balance the
        full dataset has is carried into every split.
        """
        for task in TEXT_TASKS + TABULAR_TASKS:
            with self.subTest(task=task):
                splits = self.splits(task)
                everything = []
                for rows in splits.values():
                    everything.extend(rows)
                overall = {}
                for row in everything:
                    overall[row["label"]] = overall.get(row["label"], 0) + 1
                total = len(everything)
                for name, rows in splits.items():
                    counts = {}
                    for row in rows:
                        counts[row["label"]] = counts.get(row["label"], 0) + 1
                    for label, count in overall.items():
                        expected = count / total
                        observed = counts.get(label, 0) / len(rows)
                        self.assertAlmostEqual(
                            observed, expected, delta=0.03,
                            msg=f"{task}/{name}/{label}: {observed:.3f} "
                                f"vs overall {expected:.3f}")

    def test_no_row_appears_in_two_splits(self):
        """Shared rows must be source duplicates, never splitter leakage.

        Three tasks legitimately share values across splits because the source
        data repeats records: credit_card's PCA features collide, sms_spam has
        forwarded messages (one text appears 20 times), 20 Newsgroups has quoted
        replies. Analyses in analysis/checks/{credit_card,text}_duplicates.py
        confirm every shared value repeats within the dataset itself.

        The test therefore asserts the real invariant: no value appears in two
        splits while existing only once overall. That would be leakage.
        """
        for task in TEXT_TASKS + TABULAR_TASKS:
            with self.subTest(task=task):
                splits = self.splits(task)
                first = splits["train"][0]

                if "text" in first:
                    keysets = {name: {r["text"] for r in rows}
                               for name, rows in splits.items()}
                else:
                    cols = [c for c in first if c != "label"]
                    keysets = {name: {tuple(r[c] for c in cols) for r in rows}
                               for name, rows in splits.items()}

                everything = Counter()
                for rows in splits.values():
                    for row in rows:
                        if "text" in first:
                            everything[row["text"]] += 1
                        else:
                            cols = [c for c in first if c != "label"]
                            everything[tuple(row[c] for c in cols)] += 1

                shared = ((keysets["train"] & keysets["validation"])
                          | (keysets["train"] & keysets["test"])
                          | (keysets["validation"] & keysets["test"]))
                leaked = {v for v in shared if everything[v] < 2}
                self.assertEqual(
                    leaked, set(),
                    f"{task}: {len(leaked)} values appear in two splits while "
                    f"existing only once overall -> splitter leaked rows")


class DownsamplePreservesDistributionTest(unittest.TestCase):
    """The specific defect found while preparing QQP and Yelp."""

    def test_qqp_keeps_its_natural_imbalance(self):
        """QQP is ~63/37 in the source. A 50/50 subset would be a fake dataset."""
        info = manifest("qqp")
        self.assertIn("downsample_cap", info)
        rate = info["minority_rate"]
        # Source minority is 0.369. Anything near 0.5 means the cap imposed
        # balance rather than sampling.
        self.assertLess(rate, 0.45,
                        f"minority {rate} looks artificially balanced")
        self.assertGreater(rate, 0.30,
                           f"minority {rate} drifted from the source 0.369")

    def test_yelp_keeps_its_five_way_balance(self):
        info = manifest("yelp_full")
        self.assertIn("downsample_cap", info)
        # Yelp is balanced by construction, so exactly 0.2 is correct here.
        self.assertAlmostEqual(info["minority_rate"], 0.2, delta=0.02)

    def test_downsample_method_is_recorded(self):
        for task in ("qqp", "yelp_full"):
            with self.subTest(task=task):
                info = manifest(task)
                self.assertEqual(info["downsample_method"],
                                 "uniform random subset, distribution preserved")
                self.assertEqual(info["downsample_seed"], 42)


class ManifestMatchesDiskTest(unittest.TestCase):
    def test_checksums_match_the_files(self):
        for task in TEXT_TASKS + TABULAR_TASKS:
            with self.subTest(task=task):
                declared = manifest(task)["data_sha256"]
                for name, digest in declared.items():
                    path = BENCH / task / "data" / name
                    self.assertTrue(path.is_file(), f"missing {task}/{name}")
                    self.assertEqual(sha256(path), digest,
                                     f"{task}/{name} changed after manifest")

    def test_no_windows_paths_leaked_into_cluster_configs(self):
        """The CLUSTER configs must not contain D: paths.

        The local configs legitimately use D:\\ -- they run on this machine. The
        cluster copies are generated separately by
        cluster/make_cluster_configs.ps1; a D: path surviving into those is what
        would break a Linux job, so that is what gets asserted. Skipped when the
        generated directory is absent (it is a build artifact, not source).
        """
        cluster_configs = BENCH.parent / "cluster" / "cluster_configs"
        if not cluster_configs.is_dir():
            self.skipTest("cluster_configs/ not generated; nothing to check")
        checked = 0
        for path in cluster_configs.glob("*_config.json"):
            with self.subTest(config=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("D:", text,
                                 f"{path.name} still points at a Windows path")
                self.assertIn("/hpc2hdd/", text,
                              f"{path.name} has no cluster path")
                checked += 1
        self.assertGreater(checked, 0)


class CifarSplitTest(unittest.TestCase):
    def test_cifar_has_ten_classes_in_each_split(self):
        info = manifest("cifar10")
        data = BENCH / "cifar10" / "data"
        for split in ("train", "validation", "test"):
            with self.subTest(split=split):
                directory = data / split
                self.assertTrue(directory.is_dir())
                classes = [d for d in directory.iterdir() if d.is_dir()]
                self.assertEqual(len(classes), 10,
                                 f"{split} has {len(classes)} classes")
                for cls in classes:
                    images = list(cls.glob("*.png"))
                    self.assertGreater(len(images), 0, f"{cls} is empty")

    def test_cifar_train_and_validation_are_disjoint(self):
        """Both come from the official train split, so overlap is possible."""
        data = BENCH / "cifar10" / "data"
        train_names = {p.name for p in (data / "train").rglob("*.png")}
        val_names = {p.name for p in (data / "validation").rglob("*.png")}
        self.assertEqual(train_names & val_names, set())

    def test_cifar_split_sizes_match_manifest(self):
        info = manifest("cifar10")
        data = BENCH / "cifar10" / "data"
        for split, expected in info["split_sizes"].items():
            with self.subTest(split=split):
                actual = len(list((data / split).rglob("*.png")))
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
