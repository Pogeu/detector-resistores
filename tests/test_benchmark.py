from resistor_detector.benchmark import ObjectRecord, SampleRecord, evaluate


def _sample(sample_id: str, *, predicted: bool = False) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        objects=(
            ObjectRecord(
                bbox=(10, 10, 100, 30),
                bands=("brown", "black", "red", "gold"),
                ohms=1000,
                tolerance_percent=5,
                confidence=0.9 if predicted else 1.0,
            ),
        ),
    )


def test_evaluate_reports_perfect_detection_and_value():
    report = evaluate([_sample("a")], [_sample("a", predicted=True)])

    assert report["objects"]["f1"] == 1.0
    assert report["bands"]["slot_accuracy"] == 1.0
    assert report["value"]["nominal_exact"] == 1.0
    assert report["value"]["object_exact"] == 1.0


def test_evaluate_counts_false_positive_and_missing_object():
    expected = [_sample("a")]
    predicted = [
        SampleRecord(
            sample_id="a",
            objects=(
                ObjectRecord(bbox=(500, 500, 20, 20), confidence=0.8),
            ),
        )
    ]

    report = evaluate(expected, predicted)

    assert report["objects"]["true_positive"] == 0
    assert report["objects"]["false_positive"] == 1
    assert report["objects"]["false_negative"] == 1
