from collections import deque, namedtuple
from typing import List
from enum import Enum
import numpy as np
from src.entities.component import ComponentFactory, MachineComponent

class MachineStatus(Enum):
    IDLE = 0
    RUNNING = 1
    MAINTENANCE = 2
    BROKEN = 3
    
Product = namedtuple("Product", ["faulty", "step_cost"])
Maintenance = namedtuple("Maintenance", ["component", "step_cost"])

component_factory = ComponentFactory()

def is_empty(buffer: deque) -> bool:
    """
    Check if the buffer is empty.
    """
    return len(buffer) == 0

def is_full(buffer: deque) -> bool:
    """
    Check if the buffer is full.
    """
    return len(buffer) == buffer.maxlen

class Machine:
    
    def __init__(
        self,
        id: int,
        components_list: List[str],
        prod_per_step: int,
        in_buffer_size: int,
        out_buffer_size: int,
        calibration_consequence_prob: float | None,
        components_failures_nerfs: dict,
        fast_degradation: bool,
    ):
        
        self.id = id
        self.components_list = components_list
        self.prod_per_step = prod_per_step
        self.fast_degradation = fast_degradation
        
        self.components = [ComponentFactory.create_component(component_type=component, fast_degradation=fast_degradation) for component in components_list]
        self.status = MachineStatus.IDLE
        
        # Initialize the machine's in_buffer and out_buffer 
        self.in_buffer = deque(maxlen=in_buffer_size)
        self.out_buffer = deque(maxlen=out_buffer_size)

        
        # Current production step
        self.current_prod_step = None
        self.current_product = None
        
        # Current maintenance step
        self.current_maintenance = None
        self.current_maintenance_step = None
        
        # Components breakdown consequences
        self.components_failure_nerfs = components_failures_nerfs
        self.calibration_consequence_prob = calibration_consequence_prob
        self.current_failures_nerfs = {
            "sensor": False,
            "calibration": False,
        }
        self.current_maintenance_cost = 0
        
        
    
    def step(
        self,
    ):
        """
        Step function to update the machine's state.
        """
        ## Machine Maintenance :
        # Check if the machine is in maintenance
        if self.status == MachineStatus.MAINTENANCE:
            self.current_maintenance_step -= 1
            if self.current_maintenance_step <= 0:
                self.status = MachineStatus.IDLE

                repaired_component = self.current_maintenance.component
                # Repair the component
                if repaired_component == "all":
                    for component in self.components:
                        component.repair()
                else:
                    for component in self.components:
                        if component.name == repaired_component:
                            
                            # Potentially repair the nerfs
                            if component.name == "sensor":
                                if self.current_failures_nerfs["sensor"] and self.components_failure_nerfs["sensor"]:
                                    self.current_failures_nerfs["sensor"] = False
                            elif component.name == "calibration":
                                if self.current_failures_nerfs["calibration"] and self.components_failure_nerfs["calibration"]:
                                    self.prod_per_step *= 2
                                    self.current_failures_nerfs["calibration"] = False
                            
                            component.repair()
                            break
                # Reset the current maintenance step
                self.current_maintenance = None
                self.current_maintenance_step = None
                return
        if (self.current_product != None) and self.status != MachineStatus.BROKEN:
            ## Machine components :
            for component in self.components:
                component.step()
            # Check if the machine is broken
            self._component_breakdown()
            
        ## If the machine can work :
        # Assume the in_buffer has always been updated by the Factory env
        if self.status == MachineStatus.IDLE:
            if not is_empty(self.in_buffer):
                self.status = MachineStatus.RUNNING
                self.current_product = self.in_buffer.popleft()
                self.current_prod_step = 0
                self.current_maintenance_step = None
            else:
                return
        elif self.status == MachineStatus.RUNNING:
            self.current_prod_step += 1
            if self.current_prod_step >= self.current_product.step_cost:
                self.current_prod_step = None
                self.status = MachineStatus.IDLE
                
                # Check if the calibration nerf is active
                if self.current_failures_nerfs["calibration"] and self.components_failure_nerfs["calibration"]:
                    if np.random.rand() < self.calibration_consequence_prob:
                        # If the calibration nerf is active, the product is faulty
                        self.current_product = Product(faulty=True, step_cost=self.current_product.step_cost)
                
                self.out_buffer.append(self.current_product)
                self.current_product = None
                return
                
        return            
            
        
    def _component_breakdown(
        self,
    ):
        """
        Check if any component is broken and act accordingly
        """
        
        for component in self.components:
            if component.broken:
                match component.name:
                    case "rotor" | "bearing" | "pump":
                        self.status = MachineStatus.BROKEN
                        # Reset current product step if the machine is broken and the fast degradation nerf is not active (else, no machine will have enough time to produce anything)
                        self.current_prod_step = 0 if not self.fast_degradation else None
                        
                    case "sensor":
                        self.current_failures_nerfs["sensor"] = True
                        # Reset current product step
                        
                    case "calibration":
                        if (not self.current_failures_nerfs["calibration"]) and self.components_failure_nerfs["calibration"]:
                            self.prod_per_step /= 2
                            self.current_failures_nerfs["calibration"] = True
                            
                            
                            
    def get_machine_features(
        self,
    ):
        """
        Get the machine's components features.
        This function should be called to get the machine's components features.
        It will return a numpy array with the machine's components features.
        """
        features = [component.get_component_features() for component in self.components]
        
        if self.current_failures_nerfs["sensor"] and self.components_failure_nerfs["sensor"]:
            # If the sensor is broken and the sensor breakdown nerf is allowed, give a random expected-time-to-failure value for each component (to simulate non-working sensor)
            for i, component in enumerate(features):
                component[-1] = np.random.uniform(0, self.components[i].wb_eta)
        
        
        # Add the machine's status
        features.append(np.array([self.status.value]))
        
        # Add the steps until the product is finished
        if self.current_prod_step is not None:
            features.append(np.array([self.current_product.step_cost - self.current_prod_step]))
        else:
            features.append(np.array([-1.0]))
            
        # Add the steps until the maintenance is finished
        if self.current_maintenance_step is not None:
            features.append(np.array([self.current_maintenance_step]))
        else:
            features.append(np.array([-1.0]))
            
        # Add the in_buffer and out_buffer fill percentage
        features.append(np.array([len(self.in_buffer) / self.in_buffer.maxlen]))
        features.append(np.array([len(self.out_buffer) / self.out_buffer.maxlen]))
            
        # Return the features as a numpy array
        return np.concatenate(features, axis=0)

        
    def act_maintenance(
        self,
        maintenance: Maintenance,
    ):
        """
        Act on the machine to perform maintenance.
        """
        if self.status == MachineStatus.IDLE:
            self.status = MachineStatus.MAINTENANCE
            self.current_maintenance = maintenance
            self.current_maintenance_step = maintenance.step_cost
            self.current_maintenance_cost = maintenance.component.repair_cost if isinstance(maintenance.component, MachineComponent) else sum([c.repair_cost for c in maintenance.component])
            return True
        else:
            return False
        
    def new_product(
        self,
        product: Product,
    ):
        """
        Add a new product to the machine's in_buffer.
        """
        if not is_full(self.in_buffer):
            self.in_buffer.append(product)
            return True
        else:
            return False

    def reset(
        self,
    ):
        """
        Reset the machine to its initial state.
        """
        self.status = MachineStatus.IDLE
        self.current_product = None
        self.current_prod_step = None
        self.current_maintenance = None
        self.current_maintenance_step = None
        self.in_buffer.clear()
        self.out_buffer.clear()
        
        for component in self.components:
            component.reset()
        self.current_failures_nerfs = {
            "sensor": False,
            "calibration": False,
        }
        self.current_maintenance_cost = 0


    # Modify the __str__ method to print the machine's state
    def __str__(self):
        """
        Print the machine's state.
        """
        return f"Machine {self.id} - Status: {self.status.name} - In buffer: {len(self.in_buffer)} - Out buffer: {len(self.out_buffer)} - Current product: {self.current_prod_step} / {self.current_product.step_cost if self.current_product else None} ({self.current_product.faulty if self.current_product else None}) - Current maintenance: {self.current_maintenance_step} / {self.current_maintenance.step_cost if self.current_maintenance else None} ({self.current_maintenance.component if self.current_maintenance else None})"
    
    def _render_dict(self):
        """
        Render the machine's state as a dictionary.
        """
        return {
            "id": self.id,
            "status": self.status.name,
            "in_buffer": len(self.in_buffer),
            "out_buffer": len(self.out_buffer),
            "current_product": {
                "step_cost": self.current_prod_step,
                "faulty": self.current_product.faulty if self.current_product else None,
            },
            "current_maintenance": {
                "step_cost": self.current_maintenance_step,
                "component": self.current_maintenance.component if self.current_maintenance else None,
            },
            "components": {
                component.name: {
                    "broken": component.broken,
                    "etf": component.etf,
                    "f_prob": component.prev_cdf,
                }
                for component in self.components
            }
        }