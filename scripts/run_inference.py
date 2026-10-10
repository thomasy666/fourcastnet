"""Run the NVlabs FourCastNet inference script on the ERA5 file built by prefetch.py."""
import argparse
import os
import subprocess
import sys

from ruamel.yaml import YAML


def write_config(path, nsteps):
    """Copy the NVlabs afno_backbone config, pointing it at our data and init dates."""
    repo, weights, data = os.environ["FCN_REPO_DIR"], os.environ["WEIGHTS_DIR"], os.environ["DATA_DIR"]
    yaml = YAML(typ="safe")
    with open(os.path.join(repo, "config/AFNO.yaml")) as f:
        cfg = yaml.load(f)["afno_backbone"]
    cfg.update(
        inf_data_path=os.path.join(data, "out_of_sample"),
        global_means_path=os.path.join(weights, "stats_v0/global_means.npy"),
        global_stds_path=os.path.join(weights, "stats_v0/global_stds.npy"),
        time_means_path=os.path.join(weights, "stats_v0/time_means.npy"),
        prediction_length=nsteps + 1,  # lead 0 plus nsteps forecasts
        ics_type="datetime",
        date_strings=[t.replace("T", " ") for t in os.environ["INIT_TIMES"].split()],
        log_to_wandb=False,
    )
    with open(path, "w") as f:
        yaml.dump({"afno_backbone": cfg}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsteps", type=int, default=int(os.environ["NSTEPS"]))
    parser.add_argument("--out", required=True, help="output directory (in scratch)")
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    config = os.path.join(out, "config.yaml")
    write_config(config, args.nsteps)

    subprocess.run(
        [sys.executable, "inference/inference.py",
         "--yaml_config", config, "--config", "afno_backbone",
         "--override_dir", out,
         "--weights", os.path.join(os.environ["WEIGHTS_DIR"], "backbone.ckpt"),
         "--vis"],  # also save the full forecast and truth fields, for plots
        cwd=os.environ["FCN_REPO_DIR"], check=True,
    )


if __name__ == "__main__":
    main()
