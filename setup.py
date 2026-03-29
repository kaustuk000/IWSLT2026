import platform
import shutil
import subprocess
import sys
import os


FOLDERS = [
    "data/ASR_data",
    "data/MT_data",
    "helper",
    "model",
    "results",
    "bho_hin_mt",
]

FILES = [
    "data/MT_data/.gitkeep",
    "data/ASR_data/.gitkeep",
    "results/.gitkeep",
    "bho_hin_mt/.gitkeep",
    "README.md",
    ".gitignore",
    ".python-version",
    "pyproject.toml",
]


def create_project_structure():
    print("Creating project structure...")
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        print(f"  Created {folder}/")

    for file in FILES:
        if not os.path.exists(file):
            open(file, "w").close()
            print(f"  Created {file}")

    print("Project structure ready")


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


def sync_project(uv_path):
    if not os.path.exists("pyproject.toml"):
        print("No pyproject.toml found")
        sys.exit(1)

    print("Running uv sync...")
    try:
        env = os.environ.copy()
        env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
        subprocess.check_call([uv_path, "sync"], env=env)
        print("Dependencies synced successfully")
    except subprocess.CalledProcessError:
        print("uv sync failed")
        sys.exit(1)


def main():
    print(f"Detected OS: {platform.system()}")

    create_project_structure()

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

    sync_project(uv_path)


if __name__ == "__main__":
    main()