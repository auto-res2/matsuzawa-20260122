import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig

################################################################################
# Experiment orchestrator – launches src.train as subprocess with overrides    #
################################################################################

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cmd: List[str] = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]

    # Forward arbitrary additional overrides provided via Hydra CLI -----------
    reserved = {"run", "results_dir", "mode", "wandb"}
    for k, v in cfg.items():
        if k not in reserved and not isinstance(v, DictConfig):
            cmd.append(f"{k}={v}")

    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
