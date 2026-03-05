from app.services.transcription import should_store_fragment


def test_should_store_fragment_accepts_clean_latin_text():
    row = {"speaker": "Hugo", "low_confidence": False}
    assert should_store_fragment(row, "Hallo Leon, wir starten jetzt.")


def test_should_store_fragment_rejects_unknown_speaker():
    row = {"speaker": "Unknown", "low_confidence": False}
    assert not should_store_fragment(row, "Hello there.")


def test_should_store_fragment_rejects_low_confidence():
    row = {"speaker": "Leon", "low_confidence": True}
    assert not should_store_fragment(row, "All good here.")


def test_should_store_fragment_rejects_non_latin_script():
    row = {"speaker": "Hugo", "low_confidence": False}
    assert not should_store_fragment(row, "Привет как дела")


def test_should_store_fragment_rejects_non_speech_tag():
    row = {"speaker": "Hugo", "low_confidence": False}
    assert not should_store_fragment(row, "[music]")


def test_should_store_fragment_rejects_unsupported_detected_language():
    row = {
        "speaker": "Hugo",
        "low_confidence": False,
        "language": "fr",
        "language_confidence": 0.99,
    }
    assert not should_store_fragment(row, "Bonjour tout le monde.")


def test_should_store_fragment_rejects_weak_language_confidence():
    row = {
        "speaker": "Leon",
        "low_confidence": False,
        "language": "en",
        "language_confidence": 0.2,
    }
    assert not should_store_fragment(row, "This is hard to trust.")


def test_should_store_fragment_allows_heuristic_language_confidence():
    row = {
        "speaker": "Leon",
        "low_confidence": False,
        "language": "en",
        "language_confidence": 0.81,
    }
    assert should_store_fragment(row, "This should pass with model-language fallback.")


def test_should_store_fragment_rejects_low_quality_text():
    row = {
        "speaker": "Hugo",
        "low_confidence": False,
        "language": "de",
        "language_confidence": 0.9,
        "text_quality": 0.2,
    }
    assert not should_store_fragment(row, "Das ist eigentlich klar genug.")


def test_should_store_fragment_rejects_single_filler_word():
    row = {
        "speaker": "Hugo",
        "low_confidence": False,
        "language": "en",
        "language_confidence": 0.9,
        "text_quality": 0.95,
    }
    assert not should_store_fragment(row, "the")
