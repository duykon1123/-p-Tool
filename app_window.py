"""Mở Đớp Tool như một ỨNG DỤNG CỬA SỔ trên Windows (không cần gõ lệnh, không thấy thanh địa chỉ).
Thứ tự ưu tiên: pywebview (cửa sổ riêng) → Edge/Chrome chế độ --app → trình duyệt mặc định.
Chạy:  python app_window.py      (hoặc bấm đúp Dop_Tool_Window.bat)"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
TITLE = "Đớp Tool"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(url: str, proc: subprocess.Popen, timeout: float = 60.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(url + "/_stcore/health", timeout=1)
            return True
        except Exception:
            time.sleep(0.4)
    return False


def find_chromium() -> str | None:
    cands = [shutil.which(n) for n in ("msedge", "chrome", "google-chrome", "chromium")]
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if base:
            cands += [os.path.join(base, r"Microsoft\Edge\Application\msedge.exe"),
                      os.path.join(base, r"Google\Chrome\Application\chrome.exe")]
    return next((c for c in cands if c and os.path.exists(c)), None)


def open_window(url: str, proc: subprocess.Popen) -> None:
    try:
        import webview  # pip install pywebview
        webview.create_window(TITLE, url, width=1440, height=900, min_size=(1000, 650))
        webview.start()
        return
    except Exception:
        pass
    exe = find_chromium()
    if exe:
        # profile riêng để cửa sổ độc lập; khi đóng cửa sổ thì dừng server
        prof = os.path.join(os.environ.get("TEMP", HERE), "dop_tool_profile")
        browser = subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={prof}", "--window-size=1440,900"])
        browser.wait()
        return
    webbrowser.open(url)
    while proc.poll() is None:          # trình duyệt thường: giữ server chạy tới khi bấm Ctrl+C
        time.sleep(1)


def main() -> int:
    port = int(os.environ.get("DOP_PORT", "0")) or free_port()
    url = f"http://127.0.0.1:{port}"
    cmd = [sys.executable, "-m", "streamlit", "run", os.path.join(HERE, "web_app.py"),
           "--server.port", str(port), "--server.address", "127.0.0.1", "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    proc = subprocess.Popen(cmd, cwd=HERE)
    try:
        if not wait_ready(url, proc):
            print("Không khởi động được server Streamlit (xem thông báo lỗi ở trên).")
            return 1
        if os.environ.get("DOP_NO_WINDOW"):
            print("READY", url)
            return 0
        open_window(url, proc)
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except Exception:
            proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
