import gymnasium as gym
from gymnasium import spaces
from collections import deque
from src.entities.machine import Product, Machine
from src.entities.technician import Technician
import numpy as np

from enum import IntEnum


class FactoReal(gym.Env):
    
    metadata = {'render_modes':[]}
    
    def __init__(self, ARGS):
        
        #### Intialize the environment
        self.ARGS = ARGS
        
        ## Initialize the production line
        self.initial_products = deque(maxlen=ARGS.number_of_products)
        self.final_products = deque(maxlen=ARGS.number_of_products)
        
        # Fill the initial products buffer
        for _ in range(ARGS.number_of_products):
            self.initial_products.append(Product(faulty=False, step_cost=ARGS.product_step_cost))
            
        ## Initialize the machines
        self.machines = []
        for i in range(ARGS.number_of_machines):
            self.machines.append(ARGS.machine_class(id=i, components_list=ARGS.components_list, prod_per_step=ARGS.prod_per_step, in_buffer_size=ARGS.in_buffer_size, out_buffer_size=ARGS.out_buffer_size, calibration_consequence_prob=ARGS.calibration_consequence_prob, components_failures_nerfs=ARGS.components_failures_nerfs))
            
        ## Initialize the technicians
        self.technicians = []
        for i in range(ARGS.number_of_technicians):
            self.technicians.append(Technician(id=i, name=f"Technician {i}", ai_trust=ARGS.technician_ai_trusts, knowledge=ARGS.technician_knowledges, status="idle", available=True, unavailability_length=0, fatigue=0.0))
            
        ## Initialize the action and observation spaces
        # Get the features for a machine
        dummy_machine_features = self.machines[0].get_machine_features()
        # Get the features for a technician
        dummy_technician_features = self.technicians[0].get_technician_features()
        
        observation_space_size = len(dummy_machine_features) * len(self.machines) + len(dummy_technician_features) * len(self.technicians)
        observation_space_size += 2 # Add the initial and final products buffers capacity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_space_size,), dtype=np.float32)
        
        # Define the action space
        self.action_space = spaces.Discrete(ARGS.number_of_actions)
        
        