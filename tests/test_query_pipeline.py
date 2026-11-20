import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import query_pipeline


def test_main_prompts_for_query_when_argument_is_missing(monkeypatch):
    received_queries = []
    monkeypatch.setattr(sys, "argv", ["query_pipeline.py"])
    monkeypatch.setattr("builtins.input", lambda _: "  What is in my documents?  ")
    monkeypatch.setattr(
        query_pipeline, "run_query_pipeline", received_queries.append
    )

    query_pipeline.main()

    assert received_queries == ["What is in my documents?"]


def test_main_accepts_query_argument_without_prompting(monkeypatch):
    received_queries = []

    def fail_if_prompted(_):
        raise AssertionError("unexpected prompt")

    monkeypatch.setattr(
        sys, "argv", ["query_pipeline.py", "--query", "  Search this  "]
    )
    monkeypatch.setattr("builtins.input", fail_if_prompted)
    monkeypatch.setattr(
        query_pipeline, "run_query_pipeline", received_queries.append
    )

    query_pipeline.main()

    assert received_queries == ["Search this"]
