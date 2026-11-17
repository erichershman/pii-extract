import pytest
from bench.gretel import BASE_LABELS, gold_spans, label_table, parse_entities, prompts
from bench.metrics import Counts, micro_average, percentile, score_exact, score_overlap


def test_exact_scoring_requires_boundaries_and_label():
    gold = [(0, 8, "person"), (20, 40, "email"), (44, 56, "phone number")]
    predicted = [(0, 8, "person"), (20, 39, "email"), (44, 56, "ssn")]
    counts = score_exact(gold, predicted)
    assert vars(counts["person"]) == {"tp": 1, "fp": 0, "fn": 0}
    assert vars(counts["email"]) == {"tp": 0, "fp": 1, "fn": 1}
    assert vars(counts["phone number"]) == {"tp": 0, "fp": 0, "fn": 1}
    assert vars(counts["ssn"]) == {"tp": 0, "fp": 1, "fn": 0}


def test_overlap_scoring_forgives_boundaries_but_not_labels():
    gold = [(20, 40, "email"), (44, 56, "phone number")]
    predicted = [(20, 39, "email"), (44, 56, "ssn"), (90, 95, "email")]
    counts = score_overlap(gold, predicted)
    assert vars(counts["email"]) == {"tp": 1, "fp": 1, "fn": 0}
    assert vars(counts["phone number"]) == {"tp": 0, "fp": 0, "fn": 1}
    assert vars(counts["ssn"]) == {"tp": 0, "fp": 1, "fn": 0}


def test_precision_recall_f1_and_micro_average():
    counts = Counts(tp=6, fp=2, fn=4)
    assert counts.precision == 0.75 and counts.recall == 0.6
    assert counts.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert counts.support == 10
    assert Counts().f1 == 0.0
    total = micro_average({"a": Counts(1, 2, 3), "b": Counts(4, 5, 6)})
    assert vars(total) == {"tp": 5, "fp": 7, "fn": 9}


def test_percentile_nearest_rank():
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == 5.0
    assert percentile(values, 0) == 1.0
    assert percentile([], 50) == 0.0


def test_parse_entities_accepts_python_repr_and_json():
    assert parse_entities("[{'entity': 'Jane', 'types': ['first_name']}]")[0]["entity"] == "Jane"
    assert parse_entities('[{"entity": "Jane", "types": ["first_name"]}]')[0]["types"] == [
        "first_name"
    ]


def test_gold_spans_locate_every_whole_token_occurrence():
    text = "Al wrote to al@example.com. Also, Al called 555-867-5309."
    gold, missing = gold_spans(
        text,
        [
            {"entity": "Al", "types": ["first_name"]},
            {"entity": "al@example.com", "types": ["email"]},
            {"entity": "555-867-5309", "types": ["phone_number"]},
            {"entity": "not in the text", "types": ["email"]},
            {"entity": "Burkeberg", "types": ["city"]},  # out-of-scope type: ignored
        ],
    )
    assert missing == 1
    assert gold == [
        (0, 2, "person"),
        (12, 26, "email"),
        (34, 36, "person"),  # not the "Al" inside "Also"
        (44, 56, "phone number"),
    ]


def test_gold_spans_merge_first_and_last_name_and_drop_nested():
    text = "Signed by Jane Doe and Doe again."
    gold, _ = gold_spans(
        text,
        [
            {"entity": "Jane", "types": ["first_name"]},
            {"entity": "Doe", "types": ["last_name"]},
            {"entity": "Jane Doe", "types": ["name"]},
        ],
    )
    assert gold == [(10, 18, "person"), (23, 26, "person")]


def test_label_table_is_configurable():
    assert list(label_table()) == list(BASE_LABELS) and len(BASE_LABELS) == 7
    extended = label_table(["medical record number"])
    assert list(extended)[-1] == "medical record number"
    assert prompts(extended)[-1] == "medical record number"
    with pytest.raises(ValueError, match="unknown extra label"):
        label_table(["shoe size"])


def test_gold_spans_follow_the_label_table():
    text = "MRN MRN-000123 for SSN 000-12-3456"
    entities = [
        {"entity": "MRN-000123", "types": ["medical_record_number"]},
        {"entity": "000-12-3456", "types": ["ssn"]},
    ]
    assert gold_spans(text, entities)[0] == [(23, 34, "ssn")]
    assert gold_spans(text, entities, label_table(["medical record number"]))[0] == [
        (4, 14, "medical record number"),
        (23, 34, "ssn"),
    ]
