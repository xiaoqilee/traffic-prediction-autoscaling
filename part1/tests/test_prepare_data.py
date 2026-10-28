"""Checks for count conservation, missing observations, and training-only selection."""

import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.prepare_data import (
    FUNCTION_COLUMNS,
    MINUTE_COLUMNS,
    MINUTES_PER_DAY,
    daily_path,
    main,
    prepare,
    read_day,
)


def write_rows(path, records):
    rows = []
    for owner, app, function, trigger, counts in records:
        row = dict(zip(FUNCTION_COLUMNS + ["Trigger"], [owner, app, function, trigger]))
        row.update(zip(MINUTE_COLUMNS, counts))
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


class ReadDayTests(unittest.TestCase):
    def test_cached_archive_extraction_is_temporary_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            archive_path = raw / "azurefunctions2019.tar.xz"
            with tarfile.open(archive_path, "w:xz") as archive:
                for day in range(1, 15):
                    content = f"source day {day}\n".encode()
                    member = tarfile.TarInfo(daily_path(raw, day).name)
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
            observed = []

            def check_source(source_dir, output_dir, figure_dir, chunksize):
                observed.append(source_dir)
                self.assertNotEqual(source_dir, raw)
                for day in range(1, 15):
                    self.assertEqual(daily_path(source_dir, day).read_text(), f"source day {day}\n")
                self.assertEqual(json.loads((source_dir / "source.json").read_text())["archive_bytes"],
                                 archive_path.stat().st_size)

            with patch("sys.argv", ["prepare_data.py", "--raw-dir", str(raw)]), \
                    patch("src.prepare_data.prepare", side_effect=check_source), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(len(observed), 1)
            self.assertFalse(observed[0].exists())
            self.assertEqual(len(list(raw.glob('*.csv'))), 0)
            self.assertTrue(archive_path.exists())
            self.assertTrue((raw / "source.json").exists())

    def test_http_functions_sum_across_chunks_and_ignore_other_triggers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "day.csv"
            one = np.ones(MINUTES_PER_DAY, dtype=int)
            write_rows(path, [
                ("owner", "app", "f1", "http", one * 2),
                ("owner", "other", "f1", "http", one * 5),
                ("owner", "app", "timer", "timer", one * 100),
                ("owner", "app", "f2", "http", one * 3),
            ])
            totals, functions, audit = read_day(path, chunksize=1)
            np.testing.assert_array_equal(totals[("owner", "app")], one * 5)
            self.assertEqual(functions[("owner", "app")], {"f1", "f2"})
            self.assertEqual(audit["aggregated_http_total"], 10 * MINUTES_PER_DAY)
            selected, _, _ = read_day(path, chunksize=3, selected_apps={("owner", "app")})
            self.assertEqual(set(selected), {("owner", "app")})
            np.testing.assert_array_equal(selected[("owner", "app")], one * 5)

    def test_duplicate_function_in_another_chunk_excludes_ambiguous_app(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "day.csv"
            one = np.ones(MINUTES_PER_DAY, dtype=int)
            write_rows(path, [
                ("owner", "ambiguous", "f1", "http", one * 2),
                ("owner", "valid", "f1", "http", one * 5),
                ("owner", "ambiguous", "f1", "http", one * 3),
            ])
            totals, _, audit = read_day(path, chunksize=1)
            self.assertEqual(set(totals), {("owner", "valid")})
            self.assertEqual(audit["excluded_ambiguous_total"], 5 * MINUTES_PER_DAY)
            self.assertEqual(audit["aggregated_http_total"], 5 * MINUTES_PER_DAY)
            self.assertTrue(audit["conservation_passed"])

    def test_negative_counts_fail_instead_of_being_clipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "day.csv"
            counts = np.zeros(MINUTES_PER_DAY, dtype=int)
            counts[17] = -1
            write_rows(path, [("owner", "app", "f1", "http", counts)])
            with self.assertRaisesRegex(ValueError, "Negative HTTP"):
                read_day(path, chunksize=1)


class PipelineTests(unittest.TestCase):
    def test_selection_ignores_held_out_values_and_missing_days_stay_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            minute = np.arange(MINUTES_PER_DAY)
            templates = []
            for number in range(8):
                if number < 4:
                    counts = 20 + (minute % (number + 1))
                else:
                    counts = np.full(MINUTES_PER_DAY, 20)
                    counts[::30] += 50 * (number - 3)
                templates.append(("owner", f"app{number}", "fn", "http", counts))
            for day in range(1, 15):
                write_rows(daily_path(raw, day), templates)

            def run(output):
                with patch("src.prepare_data.plot_workloads"), contextlib.redirect_stdout(io.StringIO()):
                    return prepare(raw, output, root / "figures", chunksize=3)

            first, quality = run(root / "first")
            self.assertEqual(len(first), 4)
            self.assertEqual(first.groupby("pattern").size().to_dict(), {"bursty": 2, "steady": 2})
            self.assertEqual(quality["missing_output_minutes"], 0)
            self.assertTrue(quality["all_daily_conservation_checks_passed"])
            self.assertTrue(quality["export_conservation_passed"])
            self.assertEqual(quality["selected_source_total"], quality["exported_count_total"])
            self.assertTrue(quality["continuous_minute_index"])
            traffic = pd.read_csv(root / "first/traffic.csv")
            self.assertEqual(len(traffic), 4 * 14 * MINUTES_PER_DAY)
            self.assertFalse(traffic.duplicated(["workload_id", "minute"]).any())
            for boundary, expected in [(0, "train"), (11519, "train"),
                                       (11520, "validation"), (15839, "validation"),
                                       (15840, "test"), (20159, "test")]:
                self.assertTrue(traffic.loc[traffic.minute.eq(boundary), "split"].eq(expected).all())

            # Drastically change held-out counts and remove one selected app-day.
            missing_app = first.iloc[0]["HashApp"]
            for day in range(9, 15):
                changed = [(*record[:4], record[4] * 10000) for record in templates
                           if day != 14 or record[1] != missing_app]
                write_rows(daily_path(raw, day), changed)
            second, quality = run(root / "second")
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(quality["missing_output_minutes"], MINUTES_PER_DAY)
            changed_traffic = pd.read_csv(root / "second/traffic.csv")
            missing_id = first.iloc[0]["workload_id"]
            missing = changed_traffic.loc[
                changed_traffic.workload_id.eq(missing_id)
                & changed_traffic.minute.ge(13 * MINUTES_PER_DAY), "request_count"]
            self.assertTrue(missing.isna().all())
            contract = json.loads((root / "second/data_contract.json").read_text())
            self.assertTrue(contract["selection_uses_training_only"])


if __name__ == "__main__":
    unittest.main()
