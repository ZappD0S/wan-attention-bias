import argparse
import json
from pathlib import Path

from sklearn.model_selection import ParameterGrid

from .utils import get_folder_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file", required=True, type=Path)
    parser.add_argument("--param-grid-file", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)

    args = parser.parse_args()
    output_path = args.output_path

    with open(args.param_grid_file) as f:
        param_grid = json.load(f)

    with open(args.prompts_file) as f:
        prompts_data_list = json.load(f)

    for i, prompt_data in enumerate(prompts_data_list):
        for param_config in ParameterGrid(param_grid):
            if "repeat" in param_config:
                del param_config["repeat"]

            # iterate over actions prompts
            for prompt_type, action_prompt_data in prompt_data["action_prompts"].items():
                assert prompt_type in {"default", "first_action", "second_action", "no_locative"}
                # generate only baseline for single action prompts
                if param_config["bias_method"] != "none" and prompt_type in {
                    "first_action",
                    "second_action",
                }:
                    continue

                old_config = param_config.copy()
                old_config["prompt"] = action_prompt_data | {"type": prompt_type}
                old_folder_name = get_folder_name(old_config)

                new_config = {}
                new_config["params"] = param_config
                new_config["prompt_data"] = prompt_data
                new_config["prompt_type"] = prompt_type

                old_folder_path = output_path / old_folder_name

                if not old_folder_path.exists():
                    print(f"Folder {old_folder_path} does not exist!")
                    continue

                new_folder_name = get_folder_name(new_config)
                new_folder_path = old_folder_path.rename(output_path / new_folder_name)

                config_path = new_folder_path / "config.json"
                config_path.rename(config_path.with_suffix(".json.bak"))

                assert not config_path.exists()
                with config_path.open("w") as f:
                    json.dump(new_config, f, indent=2)


if __name__ == "__main__":
    main()
