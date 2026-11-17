import threading
import time

import pytest

from pii_extract.backends import create_backend
from pii_extract.backends.base import ModelSession, checked_span, resolve_device
from pii_extract.backends.budget import label_prompt_cost, subword_costs
from pii_extract.spans import OffsetError


class CountingLoader:
    def __init__(self, delay: float = 0.0):
        self.calls = 0
        self.delay = delay

    def __call__(self, device: str) -> dict:
        self.calls += 1
        time.sleep(self.delay)
        return {"device": device, "generation": self.calls}


def test_nothing_is_loaded_until_first_use():
    loader = CountingLoader()
    session = ModelSession(loader, device="cpu")
    assert not session.loaded and loader.calls == 0
    with session.use() as model:
        assert model == {"device": "cpu", "generation": 1}
    assert session.loaded and loader.calls == 1


def test_load_is_idempotent():
    loader = CountingLoader()
    session = ModelSession(loader, device="cpu")
    assert session.load() is session.load()
    assert loader.calls == 1


def test_concurrent_first_requests_load_exactly_once():
    loader = CountingLoader(delay=0.05)
    session = ModelSession(loader, device="cpu")
    barrier = threading.Barrier(16)
    seen = []

    def worker():
        barrier.wait()
        with session.use() as model:
            seen.append(model["generation"])

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert loader.calls == 1
    assert seen == [1] * 16


def test_inference_is_serialized_per_session_but_not_across_sessions():
    active = {"a": 0, "b": 0}
    peak = {"a": 0, "b": 0}
    overlap_between_sessions = threading.Event()
    guard = threading.Lock()
    sessions = {name: ModelSession(lambda device: object(), device="cpu") for name in active}

    def worker(name: str):
        with sessions[name].use():
            with guard:
                active[name] += 1
                peak[name] = max(peak[name], active[name])
                if all(active.values()):
                    overlap_between_sessions.set()
            time.sleep(0.02)
            with guard:
                active[name] -= 1

    threads = [threading.Thread(target=worker, args=(name,)) for name in "ab" * 6]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == {"a": 1, "b": 1}, "one forward pass at a time per session"
    assert overlap_between_sessions.is_set(), "sessions must not share a lock"


def test_close_releases_and_allows_reload():
    loader = CountingLoader()
    session = ModelSession(loader, device="cpu")
    session.load()
    session.close()
    assert not session.loaded
    session.close()  # closing twice is fine
    assert session.load()["generation"] == 2


def test_context_manager_loads_and_closes():
    loader = CountingLoader()
    with ModelSession(loader, device="cpu") as session:
        assert session.loaded
    assert not session.loaded


def test_explicit_device_is_used_verbatim():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"
    assert ModelSession(CountingLoader(), device="cuda:1").device == "cuda:1"


def test_create_backend_is_lazy_and_validates_names():
    backend = create_backend("gliner2", device="cpu")
    assert backend.name == "gliner2" and not backend.loaded
    assert backend.model_id == "fastino/gliner2-privacy-filter-PII-multi"
    assert create_backend("gliner", device="cpu").model_id == "urchade/gliner_multi_pii-v1"
    assert create_backend("gliner", model_id="some/other", device="cpu").model_id == "some/other"
    with pytest.raises(ValueError, match="unknown backend"):
        create_backend("spacy")


def test_checked_span_reads_text_from_the_source():
    text = "mail jane.doe@example.com"
    span = checked_span(text, 5, 25, "email", 0.9, "jane.doe@example.com")
    assert span.text == "jane.doe@example.com"
    assert checked_span(text, 5, 25, "email", 0.9, None).text == "jane.doe@example.com"


def test_checked_span_refuses_drifted_offsets():
    text = "mail jane.doe@example.com"
    with pytest.raises(OffsetError, match="backend reported"):
        checked_span(text, 4, 24, "email", 0.9, "jane.doe@example.com")
    with pytest.raises(OffsetError):
        checked_span(text, 5, 99, "email", 0.9, None)


class CharTokenizer:
    """One subword per character: enough to test the budget arithmetic."""

    def __call__(self, words, add_special_tokens=False):
        return {"input_ids": [list(range(len(word))) for word in words]}


def test_subword_costs_have_a_floor_of_one():
    assert subword_costs(CharTokenizer(), ["jane", "", "ab"]) == [4, 1, 2]
    assert subword_costs(CharTokenizer(), []) == []


def test_label_prompt_cost_counts_a_marker_per_label():
    assert label_prompt_cost(CharTokenizer(), ["ssn", "email"]) == (3 + 1) + (5 + 1)


def test_onnx_provider_is_explicit_and_defaults_to_cpu():
    from pii_extract.backends.onnx_backend import PROVIDERS
    from pii_extract.config import Settings, build_pipeline, load_settings

    backend = create_backend("onnx", device="cuda")
    assert backend.name == "onnx" and not backend.loaded and backend.providers == []
    assert backend.provider == "cpu" and backend.device == "cpu", (
        "device does not pick the provider"
    )
    assert create_backend("onnx", onnx_provider="cuda").device == "cuda"
    assert PROVIDERS["cpu"] == ["CPUExecutionProvider"]
    with pytest.raises(ValueError, match="unknown ONNX provider"):
        create_backend("onnx", onnx_provider="tensorrt")

    assert Settings().onnx_provider == "cpu"
    settings = load_settings(env={"PII_BACKEND": "onnx", "PII_ONNX_PROVIDER": "cuda"})
    assert build_pipeline(settings).backend.provider == "cuda"
    # Other backends ignore the setting.
    assert create_backend("gliner", device="cpu", onnx_provider="cuda").name == "gliner"


def test_cuda_probe_reports_a_crashing_child_instead_of_crashing():
    import sys

    from pii_extract.backends.onnx_probe import probe_cuda_provider

    def child(code: str) -> list[str]:
        return [sys.executable, "-c", code]

    crashed = probe_cuda_provider(child("import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"))
    assert not crashed.ok and "SIGSEGV" in crashed.detail and "host" in crashed.detail

    refused = probe_cuda_provider(child("import sys; print('no CUDA provider here'); sys.exit(3)"))
    assert (
        not refused.ok
        and "status 3" in refused.detail
        and "no CUDA provider here" in refused.detail
    )

    hung = probe_cuda_provider(child("import time; time.sleep(30)"), timeout=0.5)
    assert not hung.ok and "did not finish" in hung.detail

    missing = probe_cuda_provider(["/nonexistent/python"])
    assert not missing.ok and "could not be started" in missing.detail

    assert probe_cuda_provider(child("print('ok')")).ok


def test_onnx_cuda_backend_refuses_to_load_when_the_probe_fails(monkeypatch):
    from pii_extract.backends import onnx_probe

    calls = []

    def failing_probe():
        calls.append(1)
        return onnx_probe.ProbeResult(False, "the CUDA probe process was killed by SIGSEGV")

    monkeypatch.setattr(onnx_probe, "probe_cuda_provider", failing_probe)
    backend = create_backend("onnx", onnx_provider="cuda")
    with pytest.raises(
        RuntimeError, match="not usable: the CUDA probe process was killed by SIGSEGV"
    ):
        backend.load()
    assert calls == [1] and not backend.loaded


def test_onnx_cpu_backend_never_runs_the_probe(monkeypatch):
    from pii_extract.backends import onnx_probe

    def unexpected():
        raise AssertionError("the probe must not run for the cpu provider")

    monkeypatch.setattr(onnx_probe, "probe_cuda_provider", unexpected)
    backend = create_backend("onnx", model_id="/nonexistent/model-dir", onnx_provider="cpu")
    with pytest.raises(Exception) as raised:  # the missing model, not the probe
        backend.load()
    assert not isinstance(raised.value, AssertionError)
