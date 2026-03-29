import platform
import re
import shutil
import subprocess
import sys
import os


def get_uv_path():
    uv = shutil.which("uv")
    if uv:
        return uv
    local_bin = os.path.expanduser("~/.local/bin/uv")
    if os.path.exists(local_bin):
        return local_bin
    return None


def install_uv():
    system = platform.system().lower()
    print("Installing uv...")
    try:
        if system == "windows":
            subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
        else:
            subprocess.check_call(
                "curl -Ls https://astral.sh/uv/install.sh | sh",
                shell=True
            )
    except subprocess.CalledProcessError:
        print("Failed to install uv")
        sys.exit(1)


def get_required_python():
    if not os.path.exists("pyproject.toml"):
        print("No pyproject.toml found")
        sys.exit(1)

    with open("pyproject.toml", encoding="utf-8-sig") as f:  # utf-8-sig strips BOM
        content = f.read()

    for line in content.splitlines():
        if "requires-python" in line:
            print(f"  Found line: {repr(line)}")  # repr shows hidden chars
            match = re.search(r"(\d+\.\d+(?:\.\d+)?)", line)
            if match:
                version = match.group()
                if version.count(".") == 1:
                    version += ".0"
                return version

    print("Could not detect required Python version from pyproject.toml")
    sys.exit(1)


def install_python(uv_path, version):
    print(f"\n[1/3] Installing Python {version} via uv...")
    try:
        env = os.environ.copy()
        env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")

        with open(".python-version", "w") as f:
            f.write(version + "\n")
        print(f"  Pinned .python-version to {version}")

        subprocess.check_call([uv_path, "python", "install", version], env=env)
        print(f"  Python {version} installed successfully")
    except subprocess.CalledProcessError:
        print("Failed to install Python version")
        sys.exit(1)


def create_venv(uv_path, version):
    print(f"\n[2/3] Creating virtual environment with Python {version}...")
    try:
        env = os.environ.copy()
        env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
        subprocess.check_call([uv_path, "venv", "--python", version], env=env)
        print("  Virtual environment created at .venv/")
    except subprocess.CalledProcessError:
        print("Failed to create virtual environment")
        sys.exit(1)


def sync_project(uv_path):
    print("\n[3/3] Running uv sync...")
    try:
        env = os.environ.copy()
        env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
        subprocess.check_call([uv_path, "sync"], env=env)
        print("  Dependencies synced successfully")
    except subprocess.CalledProcessError:
        print("uv sync failed")
        sys.exit(1)


def main():
    print(f"Detected OS: {platform.system()}")

    uv_path = get_uv_path()
    if not uv_path:
        print("uv is NOT installed")
        install_uv()
        uv_path = get_uv_path()
        if not uv_path:
            print("uv installation failed")
            sys.exit(1)
    else:
        print(f"uv found at: {uv_path}")

    version = get_required_python()
    install_python(uv_path, version)
    create_venv(uv_path, version)
    sync_project(uv_path)

    print("\nSetup complete!")


if __name__ == "__main__":
    main()