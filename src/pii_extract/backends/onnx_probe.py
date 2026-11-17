"""Out-of-process check that ONNX Runtime's CUDA provider works on this host.

A broken GPU stack can kill the process with a segmentation fault while a
CUDA session is being created (this project met exactly that: a stale driver
library on a WSL host). A segfault cannot be caught, so the only safe way to
find out is to try it in a child process and read its exit status.

    python -c "from pii_extract.backends.onnx_probe import main; main()"   # exit 0 if it works

The child builds a session for a one-node model on the CUDA provider and runs
it once. That exercises provider loading, the CUDA runtime, cuDNN and the
driver, without loading the real model.
"""

from __future__ import annotations

import base64
import signal
import subprocess
import sys
from dataclasses import dataclass

CUDA_PROVIDER = "CUDAExecutionProvider"
EXIT_NO_CUDA_PROVIDER = 3
EXIT_WRONG_RESULT = 4
DEFAULT_TIMEOUT_SECONDS = 120.0

# A serialized ONNX model: y = Relu(x), x float32[1, 4], opset 17.
_RELU_MODEL = base64.b64decode(
    "CAk6PwoMCgF4EgF5IgRSZWx1EgVwcm9iZVoTCgF4Eg4KDAgBEggKAggBCgIIBGITCgF5Eg4KDAgBEggKAggBCgIIBEIECgAQEQ=="
)


_CHILD_COMMAND = "from pii_extract.backends.onnx_probe import main; main()"


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    detail: str


def run_child() -> int:
    """Runs in the child process. May crash; that is the point."""
    import numpy as np
    import onnxruntime

    if CUDA_PROVIDER not in onnxruntime.get_available_providers():
        print("this ONNX Runtime build has no CUDA provider (install onnxruntime-gpu)")
        return EXIT_NO_CUDA_PROVIDER
    session = onnxruntime.InferenceSession(_RELU_MODEL, providers=[CUDA_PROVIDER])
    if CUDA_PROVIDER not in session.get_providers():
        print(f"ONNX Runtime could not initialise CUDA and is using {session.get_providers()}")
        return EXIT_NO_CUDA_PROVIDER
    (output,) = session.run(None, {"x": np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)})
    if output.tolist() != [[0.0, 0.0, 1.0, 2.0]]:
        print(f"the CUDA provider returned a wrong result: {output.tolist()}")
        return EXIT_WRONG_RESULT
    print("ok")
    return 0


def probe_cuda_provider(
    command: list[str] | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> ProbeResult:
    """Try the CUDA provider in a child process and say what happened.

    command is the child to run; it is a parameter so tests can substitute
    a process that crashes on purpose.
    """
    command = command or [sys.executable, "-c", _CHILD_COMMAND]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return ProbeResult(False, f"the CUDA probe did not finish within {timeout:.0f} seconds")
    except OSError as error:
        return ProbeResult(False, f"the CUDA probe could not be started: {error}")

    if done.returncode == 0:
        return ProbeResult(True, "the CUDA provider created a session and ran it")
    if done.returncode < 0:
        try:
            name = signal.Signals(-done.returncode).name
        except ValueError:
            name = f"signal {-done.returncode}"
        return ProbeResult(
            False,
            f"the CUDA probe process was killed by {name} while creating a CUDA session; "
            "this points at the GPU driver or CUDA libraries on the host, not at the model",
        )
    lines = [line for line in (done.stdout + done.stderr).strip().splitlines() if line.strip()]
    message = lines[-1].strip() if lines else "no output"
    return ProbeResult(False, f"the CUDA probe exited with status {done.returncode}: {message}")


def main() -> None:
    sys.exit(run_child())
