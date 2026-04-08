# Execution file
import argparse
from pathlib import Path
from project.main import main


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cab", default="cab1", type=str)
    parser.add_argument("--test_step", default="", type=str)
    parser.add_argument("--config", default="predict", type=str)
    args = parser.parse_args()

    # Interface Input
    if 1:
        cab = args.cab
        test_step = args.test_step
        config = args.config
    else:
        cab = "cab1"
        test_step = "Occupy cab1"
        config = "predict"

    print(f"cab: {cab}, test_step: {test_step}, config: {config}")

    # Training values
    class_cnt = 28
    train_cnt = 250
    feat = 9
    k = 5
    c = 10
    kernel = "linear"

    # Paths
    path_train = Path(__file__).resolve().parent / "project" / "Train data" / f"TRAXX_AC3_Training_allCabs_{class_cnt}class_cnt{train_cnt}.xlsx"
    dir_name = ""

    main(cab, k, feat, kernel, c, path_train, dir_name, config, test_step)



'''    
In prototype:

    project_path = Path("/pfad/zu/project_b") 
    python = project_path / "venv/bin/python" 
    entrypoint = project_path / "run.py"

    cmd = [
        str(python),
        str(entrypoint),
        "--cab", cab, 
        "--test_step", test_step,
        "--config", config
        ]
    result = subprocess.run(cmd)
    element_name = result.stdout.strip().splitlines()[-1] if result else None

'''