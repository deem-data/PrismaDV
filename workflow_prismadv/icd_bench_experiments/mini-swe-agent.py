import os
import subprocess

from prismadv.utils import get_project_root

# Note this script will be run in a mini-swe-agent venvc, so it cannot use prismaDV code

path = get_project_root() / "data_processed" / "handcrafted_evaluation/"
directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

for directory in directories:
    os.chdir(f"{path}/{directory}/minisweagent/")

    if os.path.isfile("constraints.json"):
        print(f"Skipping {directory}, already done.")
    else:
        print(f"Running Mini SWE Agent in {directory}")
        command = "mini --model \"openai/gpt-5\" --task \"Solve the task specified in task.txt.\" --output trajectory.json --cost-limit 0.50 --yolo --exit-immediately"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
