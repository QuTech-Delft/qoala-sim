import os
import subprocess


def main():
    path_to_here = os.path.dirname(os.path.abspath(__file__))
    for root, _, files in os.walk(path_to_here):
        for filename in files:
            if filename.startswith("example") and filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                print(f"Running example {filepath}")
                result = subprocess.run(
                    ["python3", filepath],
                    stdout=subprocess.DEVNULL,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Example {filepath} failed!")

    print("All examples work!")


if __name__ == "__main__":
    main()
