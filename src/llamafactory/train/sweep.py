import os
import wandb
import yaml
from typing import Dict, Any, Optional
from ..hparams import get_train_args, read_args

def run_sweep(
    sweep_config_path: str,
    base_config: Optional[Dict[str, Any]] = None,
    project: str = "llamafactory",
    entity: Optional[str] = None
) -> None:
    """
    Run a wandb sweep with the given configuration file.
    
    Args:
        sweep_config_path: Path to the wandb sweep yaml config
        base_config: Base training configuration to override with sweep params
        project: The wandb project name
        entity: The wandb entity/username
    """
    # Load sweep config from yaml
    with open(sweep_config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    def sweep_train(config: Optional[Dict[str, Any]] = None) -> None:
        wandb.init()
        
        # Start with base config if provided
        args = base_config.copy() if base_config else {}
        
        # Override with sweep config
        if config:
            for key, value in wandb.config.items():
                # Handle nested configs like training_args.learning_rate
                parts = key.split('.')
                target = args
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = value

        # Ensure wandb logging is enabled
        if "report_to" not in args:
            args["report_to"] = ["wandb"]
        elif "wandb" not in args["report_to"]:
            args["report_to"].append("wandb")

        # Run training
        from ..train.tuner import run_exp
        run_exp(args=args)

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    
    # Run agent
    wandb.agent(sweep_id, function=sweep_train)