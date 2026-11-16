"""Shared helper: run ``demo.source`` on an ephemeral port for tests."""

from __future__ import annotations

import socket
import threading
import time

import uvicorn


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class DemoServer:
    def __init__(self, host: str = "127.0.0.1") -> None:
        import demo.source as source

        self.app = source.create_app("demo-test")
        self.host = host
        self.port = _free_port()
        config = uvicorn.Config(
            self.app, host=host, port=self.port, log_level="warning"
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> "DemoServer":
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started and time.time() < deadline:
            time.sleep(0.02)
        assert self.server.started, "demo server did not start"
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def requests(self) -> list[dict]:
        return self.app.state.log.all()

    def reset(self) -> None:
        self.app.state.log.clear()
        self.app.state.counter.reset()
