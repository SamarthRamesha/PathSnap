"""Run the real PathSnap application in the foreground on a private Mac LAN."""

import errno
from importlib.metadata import PackageNotFoundError, version
from ipaddress import IPv4Address, IPv4Interface, IPv4Network
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


PRIVATE_LAN = tuple(IPv4Network(value) for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))


def lan_interface() -> IPv4Interface:
    selected = os.environ.get("PATHSNAP_LAN_INTERFACE")
    for name in (selected,) if selected else ("en0", "en1"):
        result = subprocess.run(
            ["/sbin/ifconfig", name], capture_output=True, text=True, check=False
        )
        match = re.search(r"\binet (\d+\.\d+\.\d+\.\d+) netmask (0x[0-9a-f]+)", result.stdout)
        if match and "status: active" in result.stdout:
            address = IPv4Address(match[1])
            netmask = IPv4Address(int(match[2], 16))
            interface = IPv4Interface(f"{address}/{netmask}")
            if any(interface.network.subnet_of(network) for network in PRIVATE_LAN):
                return interface
    raise RuntimeError("No active private Mac LAN found. Join Wi-Fi or set PATHSNAP_LAN_INTERFACE.")


def demo_port() -> int:
    requested = os.environ.get("PATHSNAP_PORT")
    for port in (int(requested),) if requested else (8000, 8001):
        if not 1 <= port <= 65535:
            raise RuntimeError("PATHSNAP_PORT must be between 1 and 65535")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("0.0.0.0", port))
            return port
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
        print(f"Port {port} is occupied. Existing listeners:")
        subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], check=False)
        print("To release it, use Ctrl+C in that server's own terminal if it is yours.")
    raise RuntimeError("Pick a free port with PATHSNAP_PORT=8002 ./start_demo.sh")


def verify_packages() -> None:
    for file in (Path("requirements-backend.txt"), Path("model_artifacts/requirements-model.txt")):
        for line in file.read_text(encoding="utf-8-sig").splitlines():
            requirement = line.strip()
            if not requirement or requirement.startswith("#"):
                continue
            package, expected = requirement.split("==")
            try:
                installed = version(package)
            except PackageNotFoundError as exc:
                raise RuntimeError(f"{package} is missing. Install requirements.txt.") from exc
            if installed != expected:
                raise RuntimeError(f"{package} is {installed}; install {package}=={expected} from requirements.txt.")


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use a project-local Python 3.12 environment for the trained model")
    verify_packages()
    try:
        import lightgbm
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"LightGBM could not load. On macOS verify brew install libomp. {exc}") from exc

    import uvicorn
    from fastapi import Request
    from fastapi.responses import JSONResponse

    from src.api.main import create_app
    from src.api.pipeline import Settings

    interface = lan_interface()
    port = demo_port()
    settings = Settings.from_environment()
    app = create_app(settings, demo=True)

    @app.middleware("http")
    async def private_network_only(request: Request, call_next):
        try:
            address = IPv4Address(request.client.host if request.client else "")
        except ValueError:
            return JSONResponse(status_code=403, content={"detail": "PathSnap demo is LAN-only"})
        if not address.is_loopback and address not in interface.network:
            return JSONResponse(status_code=403, content={"detail": "PathSnap demo is LAN-only"})
        return await call_next(request)

    print(f"Python {sys.version.split()[0]} | LightGBM {lightgbm.__version__} imported")
    print(f"Model directory: {settings.model_directory}")
    print(f"Evaluation directory: {settings.evaluator_directory}")
    print(f"Mac LAN address: {interface.ip} ({interface.network})")
    print(f"Windows browser: http://{interface.ip}:{port}/ui")
    print(f"Mac readiness: http://127.0.0.1:{port}/ready")
    print("Uvicorn will show incoming requests below. Ctrl+C stops this server.")
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=False, access_log=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"PathSnap startup failed: {error}") from error
