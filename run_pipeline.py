import subprocess
import sys


def run_step(command: list):
    """Run a command via subprocess, exiting on error."""
    print(f"\n▶ Running: {' '.join(command)}")
    result = subprocess.run(command, check=True)
    print(f"✔ Success: {' '.join(command)}")


def main():
    # Use the same Python interpreter that's running this script
    python = sys.executable

    steps = [
        [python, "data_collection.py"],
        [python, "preprocess.py"],
        [python, "embed.py"],
        [python, "vectorstore_setup.py"],
    ]

    for cmd in steps:
        run_step(cmd)

    print("\n✅ All steps completed successfully!")


if __name__ == "__main__":
    main()
