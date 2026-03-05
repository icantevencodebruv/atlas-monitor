from app.asr.pipeline_local import PipelineLocalBackend
from app.config import Config


def _backend() -> PipelineLocalBackend:
    cfg = Config().asr.pipeline_local
    backend = PipelineLocalBackend(cfg)
    backend._fasttext_disabled = True
    return backend


def test_heuristic_language_detects_german():
    backend = _backend()
    lang, conf = backend._heuristic_detect_language("wir gehen heute zusammen ins büro und sprechen darüber")
    assert lang == "de"
    assert conf >= 0.6


def test_heuristic_language_detects_english():
    backend = _backend()
    lang, conf = backend._heuristic_detect_language("we are going to review the transcript output together")
    assert lang == "en"
    assert conf >= 0.6


def test_text_quality_penalizes_low_value_output():
    backend = _backend()
    low = backend._text_quality_score("the", "en")
    high = backend._text_quality_score("we should improve language selection for cleaner transcripts", "en")
    assert low < 0.4
    assert high > low
