from dataclasses import dataclass


@dataclass
class ARGS:
    """
    Class to store the arguments for the environment.
    """
    # Environment parameters
    env_name: str
    obj_type: str
    env_version: str
    number_of_products: int
    product_step_cost: int
    
    # Reward function
    reward_function: dict
    # Number of machines
    number_of_machines: int
    
    # Machines
    components_list: list
    prod_per_step: int
    in_buffer_size: int
    out_buffer_size: int
    calibration_consequence_prob: float
    components_failures_nerfs: dict
    fast_degradation: bool
       
    # Technicians
    number_of_technicians: int
    max_skill_reduction: float
    max_fatigue_increase: float
    disobedience_profile: str
    technician_ai_trusts: float
    technician_knowledges: float
    fatigue_recovery_profile: str
    
    # Episode length
    episode_length: int
    
    # Seed
    seed: int
    
    
DEFAULT_DICT = {
    "env_name": "FactoReal",
    "obj_type": "single",
    "env_version": "v0.1",
    "number_of_products": 50,
    "product_step_cost": 50,
    "reward_function": {
        "uptime": True,
        "technicians_fatigue": True,
        "maintenance_costs": True,
    },
    "components_list": [
        "sensor",
        "calibration",
        "pump",
        "bearing",
        "rotor"
    ],
    "prod_per_step": 5,
    "in_buffer_size": 10,
    "out_buffer_size": 10,
    "number_of_machines": 3,
    "calibration_consequence_prob": 0.5,
    "components_failures_nerfs": {
        "sensor": True,
        "calibration": True,
    },
    "fast_degradation": False,
    "number_of_technicians": 3,
    "max_skill_reduction": 0.3,
    "max_fatigue_increase": 0.5,
    "disobedience_profile": "linear",
    "technician_ai_trusts": 0.95,
    "technician_knowledges": 0.8,
    "fatigue_recovery_profile": "fast",
    "episode_length": 1000,
    "seed":
        42,
}

SIMPLE_DICT = {
    "env_name": "FactoReal",
    "obj_type": "single",
    "env_version": "v0.1",
    "number_of_products": 50,
    "product_step_cost": 50,
    "reward_function": {
        "uptime": True,
        "technicians_fatigue": True,
        "maintenance_costs": True,
    },
    "components_list": [
        "sensor",
        "calibration",
        "pump",
        "bearing",
        "rotor"
    ],
    "prod_per_step": 5,
    "in_buffer_size": 10,
    "out_buffer_size": 10,
    "number_of_machines": 3,
    "calibration_consequence_prob": 0.5,
    "components_failures_nerfs": {
        "sensor": False,
        "calibration": False,
    },
    "fast_degradation": False,
    "number_of_technicians": 3,
    "max_skill_reduction": 0.3,
    "max_fatigue_increase": 0.5,
    "disobedience_profile": "linear",
    "technician_ai_trusts": 1.0,
    "technician_knowledges": 0.8,
    "fatigue_recovery_profile": "fast",
    "episode_length": 1000,
    "seed":
        42,
}

DEFAULT_ARGS = ARGS(**DEFAULT_DICT)

def make_config(which: str = "default") -> ARGS:
    """
    Function to create the configuration for the environment.
    :param which: The name of the configuration to create.
    :return: The configuration for the environment.
    """
    if which == "default":
        return DEFAULT_ARGS
    if which == "simple":
        return ARGS(**SIMPLE_DICT)
    else:
        raise ValueError(f"Unknown configuration: {which}")