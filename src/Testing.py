# Runs the current project stages
import subprocess
import sys


def run_command(command):
    result = subprocess.run(command)

    # Stop if something went wrong
    if result.returncode != 0:
        print("ERROR!!")
        sys.exit(1)


def main():
    # Run the data preparation stage
    run_command([sys.executable, "scripts/01_prepare_data.py"])

    # Run the model training stage
    print()
    run_command([sys.executable, "-m", "scripts.02_train_model"])

if __name__ == "__main__":
    main()