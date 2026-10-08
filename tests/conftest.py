import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from fakes import SECRET_NAMES

REPO = Path(__file__).resolve().parent.parent


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def start_streamlit(tmp_path_factory):
    """Boot tests/smoke_app.py headlessly; returns (url, log_path)."""
    procs = []

    def start(**extra_env):
        port = _free_port()
        log_path = tmp_path_factory.mktemp("streamlit") / "server.log"
        # Never hand real credentials to the no-secret smoke test.
        env = {k: v for k, v in os.environ.items() if k not in SECRET_NAMES}
        env.update(extra_env)
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "tests/smoke_app.py",
                    "--server.headless=true",
                    "--server.address=127.0.0.1",
                    f"--server.port={port}",
                    "--browser.gatherUsageStats=false",
                ],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        procs.append(proc)
        url = f"http://127.0.0.1:{port}"
        deadline = time.time() + 60
        while True:
            if proc.poll() is not None:
                pytest.fail(f"Streamlit exited early:\n{log_path.read_text()}")
            try:
                with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as r:
                    if r.status == 200:
                        return url, log_path
            except OSError:
                pass
            if time.time() > deadline:
                pytest.fail(f"Streamlit did not start in 60s:\n{log_path.read_text()}")
            time.sleep(0.5)

    yield start
    for proc in procs:
        proc.terminate()
        proc.wait(timeout=10)
