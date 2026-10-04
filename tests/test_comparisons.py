import json

from roomscan import head_to_head


def test_head_to_head_compares_shared_dimensions(tmp_path):
    ours = {"wall-a": 3.0, "wall-b": 4.0, "only-ours": 2.0}
    incumbent = {"wall-a": 3.0, "wall-b": 4.1, "only-incumbent": 2.0}
    truth = {"wall-a": 3.0, "wall-b": 4.2}
    ours_path = tmp_path / "ours.json"
    incumbent_path = tmp_path / "incumbent.json"
    ours_path.write_text(json.dumps(ours))
    incumbent_path.write_text(json.dumps(incumbent))
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(truth))

    result = head_to_head.compare(ours_path, incumbent_path, truth_path)

    assert result["shared_dimensions"] == 2
    assert result["ours_better_or_tied"] == 1
    assert result["share"] == 0.5
    assert "Head-to-head comparison" in head_to_head.to_markdown(result)