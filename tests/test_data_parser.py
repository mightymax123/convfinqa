import json
import tempfile
from collections.abc import Iterator

import pytest

from src.data_parser import ConvFinQaDataParser, ConvQA


@pytest.fixture
def mock_json_file_with_multiple_entries() -> Iterator[str]:
    """Mocks a simple set of convfinqa entries"""
    sample_data = {
        "train": [
            {
                "id": "doc_001",
                "doc": "Doc text 1.",
                "dialogue": {
                    "conv_questions": ["Q1", "Q2"],
                    "conv_answers": ["A1", "A2"],
                },
            },
            {
                "id": "doc_002",
                "doc": "Doc text 2.",
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
    GIVEN a JSON file containing multiple ConvFinQa entries in the 'train' split,
    WHEN get_all_docs_and_q_and_a_pairs() is called on the parsed data,
    THEN it should return a list of ConvQA dataclass instances matching the number of entries,
         each with correctly structured document IDs, text, questions, and answers.
    """
    parser = ConvFinQaDataParser(mock_json_file_with_multiple_entries)
    results = parser.get_all_docs_and_q_and_a_pairs(load_train_data=True)

    assert isinstance(results, list)
    assert len(results) == 2

    for convqa in results:
        assert isinstance(convqa, ConvQA)
        assert convqa.id.startswith("doc_")
        assert isinstance(convqa.doc, str)
        assert isinstance(convqa.questions, list)
        assert isinstance(convqa.answers, list)
        assert len(convqa.questions) == len(convqa.answers)
