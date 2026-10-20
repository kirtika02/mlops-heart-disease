
from train import load_and_clean_data, build_pipelines


def test_cleaning_function():
    """Check that cleaning removes missing values and creates a binary target."""
    df = load_and_clean_data()

    assert not df.isnull().any().any()
    assert set(df["target"].unique()).issubset({0, 1})
    assert len(df) > 0
    assert "target" in df.columns


def test_pipeline_output_shape():
    """Check that preprocessing returns the expected number of rows."""
    df = load_and_clean_data()
    X = df.drop(columns=["target"])
    y = df["target"]

    logistic_model, random_forest_model = build_pipelines()

    for model in [logistic_model, random_forest_model]:
        preprocessor = model.named_steps["preprocessor"]
        transformed = preprocessor.fit_transform(X.iloc[:10], y.iloc[:10])

        assert transformed.shape[0] == 10
        assert transformed.shape[1] > 0
