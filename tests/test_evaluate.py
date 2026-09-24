import json

from app.evaluate import evaluate


def test_evaluate_scores_each_case(tmp_path, provider):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "leave.md").write_text("Employees get 24 days of annual leave.", encoding="utf-8")
    (docs / "it.txt").write_text("Laptops are replaced every three years.", encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "name": "Test set",
        "documents": "docs",
        "cases": [
            {"question": "How many days of annual leave?", "expected_source": "leave.md",
             "expected_answer": ["first source"]},
            {"question": "Capital of France?", "answerable": False},
        ],
    }), encoding="utf-8")

    reports, summary = evaluate([spec], provider)

    checks = reports[0]["cases"][0]["checks"]
    assert checks == {"retrieval": True, "citation": True, "answer": True, "grounded": True}
    # The fake model always cites [1], so it fails to decline.
    assert reports[0]["cases"][1]["checks"] == {"declined": False}
    assert summary["retrieval"] == (1, 1) and summary["declined"] == (0, 1)
