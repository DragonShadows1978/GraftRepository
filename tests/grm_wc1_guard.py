"""Enforce the dispatch's write and command boundaries during legacy pytest."""
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = (ROOT / "tests", ROOT / "logs", ROOT / "artifacts/grm_wc1", Path("/tmp"))


def _writable(value, dir_fd=None):
    if isinstance(value, int) or value is None:
        return
    path = Path(os.fsdecode(value))
    if not path.is_absolute() and dir_fd is not None and dir_fd >= 0:
        path = Path(os.readlink(f"/proc/self/fd/{dir_fd}")) / path
    path = path.resolve()
    if path == Path('/dev/null') or any(path.is_relative_to(root) for root in ALLOWED):
        return
    raise PermissionError(f"WC1 dispatch write boundary: {path}")


def _audit(event, args):
    if event == "open":
        _, mode, flags = args
        if (mode and any(c in mode for c in "wax+")) or (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
            _writable(args[0])
    elif event in ("os.remove", "os.rmdir"):
        _writable(args[0], args[1])
    elif event in ("os.mkdir", "os.chmod"):
        _writable(args[0], args[2])
    elif event == "os.truncate":
        _writable(args[0])
    elif event in ("os.rename", "os.link"):
        _writable(args[0], args[2]); _writable(args[1], args[3])
    elif event == "os.symlink":
        _writable(args[1], args[2])
    elif event == "subprocess.Popen":
        command = args[1]
        tokens = shlex.split(command) if isinstance(command, str) else command
        forbidden = {"git", "kill", "pkill", "killall", "pgrep", "ps", "fuser", "systemctl"}
        if any(Path(str(t)).name in forbidden for t in tokens):
            raise PermissionError("WC1 dispatch forbids command: " + repr(command))


def pytest_configure(config):
    sys.addaudithook(_audit)
