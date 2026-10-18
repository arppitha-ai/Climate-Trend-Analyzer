from src import anomaly_detection, data_simulation, features, preprocessing


def _clean():
    return features.add_features(preprocessing.clean(data_simulation.simulate_climate()))


def test_cleaning_removes_problems():
    df = _clean()
    assert df[preprocessing.NUMERIC].isna().sum().sum() == 0
    assert not df.duplicated(["date", "region"]).any()
    assert df["temperature_c"].between(-60, 60).all()


def test_anomaly_recall_reasonable():
    ev = anomaly_detection.evaluate(anomaly_detection.detect_anomalies(_clean()))
    assert ev.loc[ev["method"] == "flag_any", "recall"].iloc[0] > 0.7
