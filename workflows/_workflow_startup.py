from copy import deepcopy
from pathlib import Path


def prepare_config(namespace, workflow_name, config=None, show_gui=True):
    """Prepare launch values without treating them as the final GUI config.

    Normal launches load/create matching YAML before Lash_E reviews it. Supplied
    configs are complete mappings used only with show_gui=False, with no YAML
    load or write. Copy them into the workflow namespace because legacy workflow
    helpers read module constants. The caller must disable Lash_E's YAML loading
    by passing workflow_globals=None when config is supplied.

    Use explicit _CONFIG_KEYS when available; otherwise use the same public
    uppercase constant types as ConfigManager. Each job needs a fresh process;
    this helper does not make mutable workflow globals concurrency-safe.
    """
    if config is not None and show_gui:
        raise ValueError("Supplied config requires show_gui=False; use execute() for GUI review.")
    if "_CONFIG_KEYS" in namespace:
        keys = namespace["_CONFIG_KEYS"]
    else:
        keys = [
            key for key, value in namespace.items()
            if key.isupper() and not key.startswith("_")
            and isinstance(value, (str, int, float, bool, list, dict))
        ]
    if config is None:
        from workflow_config_manager import ConfigManager
        ConfigManager.setup_and_reload_config(workflow_name, namespace)
        return deepcopy({key: namespace[key] for key in keys})
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(f"Incomplete config for {workflow_name}; missing keys: {', '.join(missing)}. Review and save the workflow config.")
    launch = deepcopy({key: config[key] for key in keys})
    if type(launch["SIMULATE"]) is not bool:
        raise ValueError("SIMULATE must be a boolean.")
    if not Path(launch["INPUT_VIAL_STATUS_FILE"]).is_file():
        raise FileNotFoundError(launch["INPUT_VIAL_STATUS_FILE"])
    namespace.update(deepcopy(launch))
    return launch


def confirmed_config(namespace, launch, lash_e):
    """Snapshot reviewed values and reject a controller/config disagreement."""
    config = deepcopy({key: namespace[key] for key in launch})
    if type(config["SIMULATE"]) is not bool or config["SIMULATE"] != lash_e.simulate:
        raise ValueError("Confirmed workflow config and controller simulation modes do not match.")
    if Path(config["INPUT_VIAL_STATUS_FILE"]).resolve() != Path(lash_e.nr_robot.VIAL_FILE).resolve():
        raise ValueError("Vial file changed during review; restart with the confirmed vial file.")
    return config