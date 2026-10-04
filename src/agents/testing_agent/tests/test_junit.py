from src.agents.testing_agent.junit import parse_junit_report, parse_junit_xml

REPORT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
  <testcase classname="test_users" name="test_ok" time="0.001"/>
  <testcase classname="test_users" name="test_bad"><failure message="assert 1 == 2"/></testcase>
  <testcase classname="test_users" name="test_boom"><error message="fixture failed"/></testcase>
  <testcase classname="test_users.TestLogin" name="test_skip"><skipped message="later"/></testcase>
  <testcase classname="" name="test_broken_module"><error message="collection failure"/></testcase>
</testsuite></testsuites>
"""


def test_every_outcome_is_read(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(REPORT, encoding="utf-8")

    assert parse_junit_xml(report) == {
        "test_users::test_ok": "passed",
        "test_users::test_bad": "failed",
        "test_users::test_boom": "error",
        "test_users.TestLogin::test_skip": "skipped",
        "test_broken_module": "error",
    }


def test_messages_are_kept_for_failed_and_errored_tests_only(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(REPORT, encoding="utf-8")

    _, messages = parse_junit_report(report)

    assert messages == {
        "test_users::test_bad": "assert 1 == 2",
        "test_users::test_boom": "fixture failed",
        "test_broken_module": "collection failure",
    }


def test_missing_or_malformed_report_yields_nothing(tmp_path):
    malformed = tmp_path / "bad.xml"
    malformed.write_text("<testsuites><testcase", encoding="utf-8")

    assert parse_junit_xml(tmp_path / "absent.xml") == {}
    assert parse_junit_xml(malformed) == {}
