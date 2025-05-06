import numpy as np
from src.entities.component import MachineComponent
from src.entities.machine import Maintenance
from src.config.config import DEFAULT_ARGS
from typing import List

def fatigue_update(F_t: float, delta: float, kappa: float) -> float:
    """
    One time‐step update of fatigue during work.
    
    F_{t+1} = F_t + (1 - F_t) * (1 - exp(-k * Δt))
    
    Parameters:
    -----------
    F_t : float
        Current fatigue level (0 = fresh … 1 = fully fatigued)
    delta : float
        Time‐step length (same units as k, e.g. minutes)
    kappa : float
        Fatigue accumulation rate
    
    Returns:
    --------
    F_{t+1} : float
        Fatigue after Δt of work
    """
    return F_t + (1 - F_t) * (1 - np.exp(-kappa * delta))

def recovery_update(F_t: float, delta: float, gamma: float, epsilon = 10e-3) -> float:
    """
    One time‐step update of recovery during rest.
    
    F_{t+1} = F_t * exp(-ℓ * Δt)
    
    Parameters:
    -----------
    F_t : float
        Current fatigue level (0 = fresh … 1 = fully fatigued)
    delta : float
        Time‐step length (same units as ℓ)
    gamma : float
        Recovery rate
    
    Returns:
    --------
    F_{t+1} : float
        Fatigue after Δt of rest
    """
    return F_t * np.exp(-gamma * delta) if F_t > epsilon else 0.0


class Technician:
    
    max_skill_reduction = DEFAULT_ARGS.max_skill_reduction
    max_fatigue_increase = DEFAULT_ARGS.max_fatigue_increase

    def __init__(
        self,
        id: int,
        name: str,
        ai_trust: float,
        knowledge: float,
        fatigue_recovery_profile: str,
    ):
        """
        Initialize the technician with the given parameters.
        
        Parameters
        ----------
        id : int
            The ID of the technician.
        name : str
            The name of the technician.
        ai_trust : float
            The trust level of the technician in the AI system.
        knowledge : dict
            The knowledge of the technician.
        fatigue_recovery_profile : str
            The fatigue recovery profile of the technician. (can be either "slow", "medium", or "fast")
            Fatigue model as established by https://doi.org/10.1016/j.apm.2013.02.028
        """
        self.id = id
        self.name = name
        self.ai_trust = ai_trust
        self.knowledge = knowledge
        
        self.status = "idle"
        self.available = True
        self.unavailability_length = 0
        self.fatigue = 0.0
        
        if fatigue_recovery_profile == "slow":
            self.fatigue_accumulation_rate = 0.01
            self.fatigue_recovery_rate = 0.03
            
        elif fatigue_recovery_profile == "medium":
            self.fatigue_accumulation_rate = 0.03
            self.fatigue_recovery_rate = 0.05
            
        elif fatigue_recovery_profile == "fast":
            self.fatigue_accumulation_rate = 0.05
            self.fatigue_recovery_rate = 0.07
        
    
    def reset(self):
        """
        Reset the technician to its initial state.
        """
        self.status = "idle"
        self.available = True
        self.unavailability_length = 0
        self.fatigue = 0.0
        
    def step(self):
        """
        Step function to update the technician's state.
        """
        if self.status == "working":
            self.fatigue = fatigue_update(self.fatigue, 1, self.fatigue_accumulation_rate)
            
        elif self.status == "idle":
            self.fatigue = recovery_update(self.fatigue, 1, self.fatigue_recovery_rate)
            
        # Check if the technician is available
        if self.unavailability_length > 0:
            self.unavailability_length -= 1
            if self.unavailability_length == 0:
                self.available = True
                self.status = "idle"
                
    def maintenance_order(self, component: MachineComponent | List[MachineComponent]):
        """
        Set the technician's status to "working" and set the unavailability length.
        
        Parameters
        ----------
        component : MachineComponent
            The component to be repaired.
        """
        self.status = "working"
        self.available = False
        base_repair_time = component.repair_time if isinstance(component, MachineComponent) else sum([c.repair_time for c in component])
        # Compute the maintenance time
        skill_factor = 1 - self.knowledge * self.max_skill_reduction
        fatigue_factor = 1 + self.fatigue * self.max_fatigue_increase
        self.unavailability_length = int(base_repair_time * skill_factor * fatigue_factor)
        self.unavailability_length = max(1, self.unavailability_length)
        
        return Maintenance(component=component, step_cost=self.unavailability_length)
    
    def get_technician_features(self):
        """
        Get the technician features.
        This function should be called to get the technician's features.
        It will return a numpy array with the technician's features
        """
        fatigue = np.asarray([self.fatigue])
        knowledge = np.asarray([self.knowledge])
        available = np.asarray([1.0]) if self.available else np.asarray([0.0])
        return np.concatenate((fatigue, knowledge, available), axis=0)
    
    
    # Modify the str method to return the technician's state
    def __str__(self):
        """
        Return the technician's state.
        This function should be called to get the technician's state.
        It will return a string with the technician's state.
        """
        return f"{self.name} - available: {self.available} - fatigue: {self.fatigue} - knowledge: {self.knowledge}"
    
    
    def _render_dict(self):
        """
        Return the technician's state as a dictionary.
        This function should be called to get the technician's state.
        It will return a dictionary with the technician's state.
        """
        return {
            "name": self.name,
            "available": self.available,
            "fatigue": self.fatigue,
            "knowledge": self.knowledge,
            "status": self.status,
            "unavailability_length": self.unavailability_length
        }