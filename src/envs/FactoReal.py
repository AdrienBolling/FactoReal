import gymnasium as gym
from gymnasium import spaces
from collections import deque
from src.entities.machine import Product, Machine, MachineStatus
from src.entities.technician import Technician
import numpy as np
from src.entities.actions import DISPATCHING_RULES
from src.entities.actions import TECH_RULES, MACHINE_RULES, COMPONENT_RULES

class FactoReal(gym.Env):
    
    metadata = {'render_modes':["dict"]}
    
    def __init__(self, ARGS):
        
        #### Intialize the environment
        self.ARGS = ARGS
        
        
        ## Set random seed
        self.seed = ARGS.seed
        np.random.seed(ARGS.seed)
        
        ## Initialize the production line
        self.initial_products = deque(maxlen=ARGS.number_of_products)
        self.final_products = deque(maxlen=ARGS.number_of_products)
        
        # Fill the initial products buffer
        for _ in range(ARGS.number_of_products):
            self.initial_products.append(Product(faulty=False, step_cost=ARGS.product_step_cost))
            
        ## Initialize the machines
        self.machines = []
        for i in range(ARGS.number_of_machines):
            self.machines.append(Machine(id=i, components_list=ARGS.components_list, prod_per_step=ARGS.prod_per_step, in_buffer_size=ARGS.in_buffer_size, out_buffer_size=ARGS.out_buffer_size, calibration_consequence_prob=ARGS.calibration_consequence_prob, components_failures_nerfs=ARGS.components_failures_nerfs, fast_degradation=ARGS.fast_degradation))
            
        ## Initialize the technicians
        self.technicians = []
        for i in range(ARGS.number_of_technicians):
            self.technicians.append(Technician(id=i, name=f"Technician {i}", ai_trust=ARGS.technician_ai_trusts, knowledge=ARGS.technician_knowledges, fatigue_recovery_profile=ARGS.fatigue_recovery_profile))
            
        ## Initialize the action and observation spaces
        # Get the features for a machine
        dummy_machine_features = self.machines[0].get_machine_features()
        # Get the features for a technician
        dummy_technician_features = self.technicians[0].get_technician_features()
        
        observation_space_size = len(dummy_machine_features) * len(self.machines) + len(dummy_technician_features) * len(self.technicians)
        observation_space_size += 2 # Add the initial and final products buffers capacity
        # Deinf the observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_space_size,), dtype=np.float32)
        
        # Define the action space
        self.action_space = spaces.Discrete(len(DISPATCHING_RULES))
        
        ## Initialize the env mechanisms
        self.disobedience_profile = ARGS.disobedience_profile
        self.rewards = ARGS.reward_function
        
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        # Reset the initial and final products buffers
        self.initial_products.clear()
        self.final_products.clear()
        
        # Fill the initial products buffer
        for _ in range(self.ARGS.number_of_products):
            self.initial_products.append(Product(faulty=False, step_cost=self.ARGS.product_step_cost))
            
        # Reset the machines
        for machine in self.machines:
            machine.reset()
            
        # Reset the technicians
        for technician in self.technicians:
            technician.reset()
        
        # Return the initial observation
        return self.get_observation(), {}
        
    def step(self, action):
        """
        Take a step in the environment.
        """
        # Get the action
        action_idx = action
        action = DISPATCHING_RULES[action]
        # Check if the action is legal
        legal_actions = self._legal_filter()
        if not legal_actions[action_idx]:
            # If the action is illegal, return the observation, a negative reward, and terminate the episode
            return self.get_observation(), -500, True, False, {}
        
        # Choose the technician based on the action
        tech = self._select_technician(action)
        # Simulate disobedience of the technician due to AI trust
        if action_idx != 0:
            action = self._disobedience_simulation(action, tech)
        
        if action[0] != "No maintenance":
            machine = self._select_machine(action)
            component = self._select_component(action, machine)      
            # Create the associated productted maintenance object
            maintenance = tech.maintenance_order(component=component)
            
            # Perform the maintenance
            machine.act_maintenance(maintenance)
        
        # Step the products though the buffers
        self._step_products()
        
        # Step the machines
        for machine in self.machines:
            machine.step()
        # Step the technicians
        for technician in self.technicians:
            technician.step()
        
        # Get the reward
        reward = self._get_reward()
        # Get the observation
        observation = self.get_observation()
        # Check if the episode is done
        done = self._is_done()
        # Return the observation, reward, done, and info
        truncated = False
        info = {"status": "ok"}
        return observation, reward, done, truncated, info
    
    def get_observation(self):
        """
        Get the observation of the environment.
        """
        # Get the features for each machine
        machines_features = []
        for machine in self.machines:
            machines_features.append(machine.get_machine_features())
            
        # Get the features for each technician
        technicians_features = []
        for technician in self.technicians:
            technicians_features.append(technician.get_technician_features())
            
        # Concatenate the features
        observation = np.concatenate(machines_features + technicians_features, axis=0)
        
        # Add the initial and final products buffers capacity
        observation = np.concatenate((observation, np.asarray([len(self.initial_products), len(self.final_products)])), axis=0)
        
        return observation
    
    def _step_products(self):
        """
        Step the products through the buffers.
        """
        # Step the initial products buffer
        if len(self.initial_products) > 0:
            if len(self.machines[0].in_buffer) < self.machines[0].in_buffer.maxlen:
                product = self.initial_products.popleft()
                self.machines[0].in_buffer.append(product)
                
        # Step the machines
        for i in range(len(self.machines) - 1):
            if len(self.machines[i].out_buffer) > 0:
                if len(self.machines[i + 1].in_buffer) < self.machines[i + 1].in_buffer.maxlen:
                    product = self.machines[i].out_buffer.popleft()
                    self.machines[i + 1].in_buffer.append(product)
        
        # Step the final products buffer
        if len(self.machines[-1].out_buffer) > 0:
            product = self.machines[-1].out_buffer.popleft()
            self.final_products.append(product)
        
    def _get_reward(self):
        """
        Get the reward of the environment.
        Three different rewards are implemented:
        1. The uptime of the machines
        2. The technicians' fatigue
        3. The maintenance costs
        """
        
        if self.rewards["uptime"]:
            uptime = 0
            for machine in self.machines:
                match machine.status:
                    case MachineStatus.RUNNING:
                        uptime += 1
                    case MachineStatus.MAINTENANCE:
                        uptime -= 0.5
                    case MachineStatus.IDLE:
                        uptime -= 0
                    case MachineStatus.BROKEN:
                        uptime -= 1
            uptime /= len(self.machines)
            
        if self.rewards["technicians_fatigue"]:
            fatigue = 0
            for technician in self.technicians:
                fatigue += technician.fatigue
            fatigue /= len(self.technicians)
            
        if self.rewards["maintenance_costs"]:
            maintenance_costs = 0
            for machine in self.machines:
                maintenance_costs += machine.current_maintenance_cost
                machine.current_maintenance_cost = 0
            maintenance_costs /= len(self.machines)
            
        # Combine the rewards
        reward = 0
        if self.rewards["uptime"]:
            reward += uptime
        if self.rewards["technicians_fatigue"]:
            reward -= fatigue
        if self.rewards["maintenance_costs"]:
            reward -= maintenance_costs
        return reward
    
    def _is_done(self):
        """
        Check if the episode is done.
        The episode is done if the initial products buffer is empty and the final products buffer is full.
        """
        if len(self.initial_products) == 0 and len(self.final_products) == self.ARGS.number_of_products:
            return True
        return False
    
    def _select_technician(self, action):
        if action[0] == "Least Fatigued Technician":
            return min([t for t in self.technicians if t.available], key=lambda x: x.fatigue)
        elif action[0] == "Most Knowledgeable Technician":
            return max([t for t in self.technicians if t.available], key=lambda x: x.knowledge)
        elif action[0] == "No maintenance":
            return None
        
    def _select_machine(self, action):
        if action[1] == "Most Broken Machine":
            # Compute the number of broken components for each machine
            broken_components = [sum([1 for component in machine.components if component.broken]) for machine in self.machines if not machine.status == MachineStatus.MAINTENANCE]
            # Get the machine with the most broken components
            machine = [m for m in self.machines if not m.status == MachineStatus.MAINTENANCE][np.argmax(broken_components)]
        elif action[1] == "Least In-Buffered Machine":
            machine = min([m for m in self.machines if not m.status == MachineStatus.MAINTENANCE], key=lambda x: len(x.in_buffer))
        elif action[1] == "Most In-Buffered Machine":
            machine = max([m for m in self.machines if not m.status == MachineStatus.MAINTENANCE], key=lambda x: len(x.in_buffer))
        elif action[1] == "Least Productive Machine":
            machine = min([m for m in self.machines if not m.status == MachineStatus.MAINTENANCE], key=lambda x: x.productivity)
        return machine
        
    def _select_component(self, action, machine):
        if action[2] == "All Components":
            return machine.components
        else:
            # Get the component name
            component_name = action[2].lower()
            # Get the component
            for component in machine.components:
                if component.name == component_name:
                    return component
                
    def _disobedience_simulation(self, action, technician):
        """
        Simulate the disobedience of the technician due to AI trust.
        """
        match self.disobedience_profile:
            case "none":
                return action
            case "linear":
                disobedience = technician.ai_trust
            case "quadratic":
                disobedience = technician.ai_trust ** 2
            case "root":
                disobedience = technician.ai_trust ** 0.5
                
        # The disobedience is a probability to do nothing
        if np.random.rand() > disobedience:
            return ("No maintenance",)
        else:
            return action
        
    def sample_legal_action(self):
        """
        Sample a legal action from the action space.
        """
        # Get the legal actions
        legal_actions = self._legal_filter()
        # Sample a random action from the legal filter
        action = np.random.choice(np.where(legal_actions)[0])
        # Return the action
        return action
        
    def _legal_filter(self):
        """
        Outputs a numpy array of bool values indicating whether the action is legal or not currently.
        An action is a maintenance act. A healthy machine may be repaired, but a technician is needed to do so.
        Also, two technicians may not be assigned to the same machine at the same time.
        """
        # Check if there is a technician available
        is_a_technician_available = any([t.available for t in self.technicians])
        # Check if there is a machine available
        is_a_machine_available = any([m.status != MachineStatus.MAINTENANCE for m in self.machines])
        
        # If any of these conditions are not met, any action other than "No maintenance" is illegal
        if not is_a_technician_available or not is_a_machine_available:
            # Create a legal action array
            legal_actions = np.zeros(len(DISPATCHING_RULES), dtype=bool)
            # Set the "No maintenance" action to True
            legal_actions[0] = True
            return legal_actions
        else:
            # Create a legal action array
            legal_actions = np.ones(len(DISPATCHING_RULES), dtype=bool)
            return legal_actions
    
    # Modify the __str__ method to return the environment's state
    def __str__(self) -> str:
        lines = [
            f"FactoReal – initial products: {len(self.initial_products)}"
            f" – final products: {len(self.final_products)}"
            f" – machines: {len(self.machines)}"
            f" – technicians: {len(self.technicians)}",
            "Machines: \n" + "\n".join(str(m) for m in self.machines),
            "Technicians: \n" + "\n".join(str(t) for t in self.technicians),
        ]
        return "\n".join(lines)
    
    
    def _render_dict(self):
        """
        Render the environment as a JSON object.
        """
        rendering = {}
        rendering["initial_products"] = len(self.initial_products)
        rendering["final_products"] = len(self.final_products)
        rendering["machines"] = {m.id: m._render_dict() for m in self.machines}
        rendering["technicians"] = {t.id: t._render_dict() for t in self.technicians}
        return rendering
        
    def render(self, mode="dict"):
        """
        Render the environment.
        """
        if mode == "dict":
            return self._render_dict()
        else:
            raise NotImplementedError(f"Render mode {mode} not supported")