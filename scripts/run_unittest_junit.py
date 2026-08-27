#!/usr/bin/env python3
"""Run unittest-compatible test files and emit a compact JUnit XML report."""

from __future__ import annotations

import argparse
import importlib.util
import io
import sys
import time
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence


sys.path.insert(0, str(Path.cwd()))


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.started_at: dict[unittest.case.TestCase, float] = {}
        self.elapsed: dict[unittest.case.TestCase, float] = {}

    def startTest(self, test: unittest.case.TestCase) -> None:
        self.started_at[test] = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test: unittest.case.TestCase) -> None:
        self.elapsed[test] = time.perf_counter() - self.started_at[test]
        super().stopTest(test)


def load_test_file(path: Path, index: int, *, module_prefix: str) -> unittest.TestSuite:
    module_name = f"{module_prefix}_{index}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load test file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return unittest.defaultTestLoader.loadTestsFromModule(module)


def test_status(
    result: RecordingResult, test: unittest.case.TestCase
) -> tuple[str, str | None]:
    for failed_test, traceback in result.failures:
        if failed_test is test:
            return "failure", traceback
    for failed_test, traceback in result.errors:
        if failed_test is test:
            return "error", traceback
    for skipped_test, reason in result.skipped:
        if skipped_test is test:
            return "skipped", reason
    return "success", None


def flatten_tests(suite: unittest.TestSuite) -> list[unittest.case.TestCase]:
    result: list[unittest.case.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            result.extend(flatten_tests(item))
        else:
            result.append(item)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite-name", default="T8-3 focused tests")
    parser.add_argument("--module-prefix", default="t8_3_test")
    parser.add_argument("tests", nargs="+", type=Path)
    args = parser.parse_args(argv)

    suite = unittest.TestSuite(
        load_test_file(path, index, module_prefix=args.module_prefix)
        for index, path in enumerate(args.tests)
    )
    tests = flatten_tests(suite)
    stream = io.StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        resultclass=RecordingResult,
    )
    started = time.perf_counter()
    result = runner.run(suite)
    elapsed = time.perf_counter() - started
    assert isinstance(result, RecordingResult)

    root = ET.Element("testsuites")
    testsuite = ET.SubElement(
        root,
        "testsuite",
        {
            "name": args.suite_name,
            "tests": str(result.testsRun),
            "failures": str(len(result.failures)),
            "errors": str(len(result.errors)),
            "skipped": str(len(result.skipped)),
            "time": f"{elapsed:.6f}",
        },
    )
    for test in tests:
        test_id = test.id()
        classname, _, name = test_id.rpartition(".")
        testcase = ET.SubElement(
            testsuite,
            "testcase",
            {
                "classname": classname,
                "name": name,
                "time": f"{result.elapsed.get(test, 0.0):.6f}",
            },
        )
        status, detail = test_status(result, test)
        if status != "success":
            child = ET.SubElement(testcase, status)
            child.text = detail
    output = ET.SubElement(testsuite, "system-out")
    output.text = stream.getvalue()
    ET.indent(root, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(args.output, encoding="utf-8", xml_declaration=True)
    print(stream.getvalue(), end="")
    print(f"JUnit XML: {args.output}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
