from src.classifier import TextClassifier


def test_predict_returns_tuple():
    clf = TextClassifier()
    result = clf.predict("Samsung Galaxy S24")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_predict_label_type():
    clf = TextClassifier()
    label, confidence = clf.predict("Samsung Galaxy S24")
    assert isinstance(label, str)
    assert isinstance(confidence, float)
    assert label in ("product", "adult", "toxic")


def test_predict_confidence_range():
    clf = TextClassifier()
    _, confidence = clf.predict("Nike Air Max 270")
    assert 0.0 <= confidence <= 1.0


def test_is_allowed_product():
    clf = TextClassifier()
    assert clf.is_allowed("Samsung Galaxy S24 Ultra 256GB") is True


def test_is_allowed_toxic():
    clf = TextClassifier()
    assert clf.is_allowed("fuck you idiot") is False


def test_get_detail_keys():
    clf = TextClassifier()
    detail = clf.get_detail("test text")
    assert "text" in detail
    assert "label" in detail
    assert "confidence" in detail
    assert "allowed" in detail


def test_get_detail_text_truncation():
    clf = TextClassifier()
    long_text = "a" * 200
    detail = clf.get_detail(long_text)
    assert len(detail["text"]) <= 100


def test_predict_batch():
    clf = TextClassifier()
    results = clf.predict_batch(["Samsung Galaxy", "test item"])
    assert len(results) == 2
    assert all(isinstance(r, tuple) for r in results)


def test_predict_batch_results():
    clf = TextClassifier()
    results = clf.predict_batch(["Samsung Galaxy", "Nike Air Max"])
    for label, conf in results:
        assert isinstance(label, str)
        assert isinstance(conf, float)


def test_empty_text():
    clf = TextClassifier()
    label, conf = clf.predict("")
    assert label == "product"
    assert conf == 1.0


def test_whitespace_text():
    clf = TextClassifier()
    label, conf = clf.predict("   ")
    assert label == "product"
    assert conf == 1.0


def test_explain_keys():
    clf = TextClassifier()
    result = clf.explain("Samsung Galaxy S24")
    assert "label" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "top_features" in result


def test_explain_probabilities():
    clf = TextClassifier()
    result = clf.explain("Samsung Galaxy S24")
    assert "product" in result["probabilities"]
    assert "adult" in result["probabilities"]
    assert "toxic" in result["probabilities"]


def test_explain_features_list():
    clf = TextClassifier()
    result = clf.explain("Samsung Galaxy S24")
    assert isinstance(result["top_features"], list)
    if result["top_features"]:
        assert "feature" in result["top_features"][0]
        assert "weight" in result["top_features"][0]
