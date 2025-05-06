import wandb

def init_wandb(config, project_name, entity_name, job_type):
    """
    Initialize wandb with the given configuration, project name, and run name.
    
    Args:
        config (dict): Configuration dictionary.
        project_name (str): Name of the wandb project.
        run_name (str): Name of the wandb run.
    """
    wandb.init(
        project=project_name,
        config=config,
        sync_tensorboard=True,
        entity=entity_name,
        job_type=job_type,
    )
    