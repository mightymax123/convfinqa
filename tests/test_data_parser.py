import json
import tempfile
from collections.abc import Iterator

import pytest

from app.data_parser import ConvFinQaDataParser
from app.models import ConvQA, FinancialDoc


@pytest.fixture
def mock_json_file_with_multiple_entries() -> Iterator[str]:
    """
    Given: Sample ConvFinQA data structure with train and dev splits
    When: Creating a temporary JSON file with this data
    Then: Return the file path for testing
    """
    sample_doc = {
        "pre_text": "Some introductory text.",
        "post_text": "Some trailing text.",
        "table": {"2023": {"revenue": 100.0, "profit": 20.0}},
    }
    sample_data = {
        "train": [
            {
                "id": "doc_001",
                "doc": sample_doc,
                "dialogue": {
                    "conv_questions": ["Q1", "Q2"],
                    "conv_answers": ["A1", "A2"],
                },
            },
            {
                "id": "doc_002",
                "doc": sample_doc,
                "dialogue": {
                    "conv_questions": ["Q3", "Q4"],
                    "conv_answers": ["A3", "A4"],
                },
            },
        ],
        "dev": [],
    }

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
        json.dump(sample_data, tmp)
        tmp.flush()
        yield tmp.name


def test_get_all_docs_and_q_and_a_pairs(
    mock_json_file_with_multiple_entries: str,
) -> None:
    """
    Given: A JSON file containing multiple ConvFinQa entries in the train split
    When: get_all_docs_and_q_and_a_pairs() is called on the parsed data
    Then: It should return a list of ConvQA dataclass instances with correct structure
    """
    parser = ConvFinQaDataParser(mock_json_file_with_multiple_entries, load_train_data=True)
    results = parser.parse_all_conversations()

    assert isinstance(results, list)
    assert len(results) == 2

    for convqa in results:
        assert isinstance(convqa, ConvQA)
        assert convqa.id.startswith("doc_")
        assert isinstance(convqa.doc, FinancialDoc)
        assert isinstance(convqa.questions, list)
        assert isinstance(convqa.answers, list)
        assert len(convqa.questions) == len(convqa.answers)
