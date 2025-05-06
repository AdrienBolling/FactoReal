from scipy.special import gamma as _gamma
import numpy as np

def weibull_mean(beta, eta, gamma):
    """
    Calculate the mean of the Weibull distribution.
    
    Parameters
    ----------
    beta : float
        The shape parameter of the Weibull distribution.
    eta : float
        The scale parameter of the Weibull distribution.
    gamma : float
        The location parameter of the Weibull distribution.
        
    Returns
    -------
    float
        The mean of the Weibull distribution.
    """
    return eta * _gamma(1 + 1 / beta) + gamma

def weibull_cdf_generator(beta, eta, start=0, step=1):
    """
    Generate the cumulative distribution function (CDF) of the Weibull distribution.
    
    Parameters
    ----------
    beta : float
        The shape parameter of the Weibull distribution.
    eta : float
        The scale parameter of the Weibull distribution.
    gamma : float
        The location parameter of the Weibull distribution.
    start : int, optional
        The starting point of the CDF. Default is 0.
    step : int, optional
        The step size for the CDF. Default is 1.
        
    Returns
    -------
    list
        A generator that yields the CDF values.
    """
    t = start
    while True:
        yield 1.0 - np.exp(- np.power( t / eta, beta))
        t += step

class MachineComponent:
    
    name = None
    wb_beta = None
    wb_eta = None
    wb_gamma = None
    repair_time = None
    repair_cost = None
    
    def __init__(self, fast_degradation: bool):
        """
        Initialize the component.
        This function should be called to initialize the component.
        It will initialize the component's parameters and the environment's hyperparameters.
        """
        # Initialize the component's state
        self.broken = False  # The component is ok by default
        if fast_degradation:
            self.wb_eta = self.wb_eta / 50  # Fast degradation
        
        # Initialize the weibull_cdf_generator
        self.weibull_cdf = weibull_cdf_generator(self.wb_beta, self.wb_eta, start=0, step=1)  # TODO: to be implemented - taking into account the gamma parameter
        self.etf = weibull_mean(self.wb_beta, self.wb_eta, self.wb_gamma)  # TODO: to be implemented - taking into account the gamma parameter
        self.prev_cdf = 0.0  # The current failure probability is 0 by default
        
    def step(
        self,
    ):
        """
        Step function to update the component's state.
        """
        if self.broken:
            return
        self.etf -= 1
        new_failure_prob = next(self.weibull_cdf)

        # Sample a boolean according to the Weibull distribution
        if np.random.uniform(0, 1) < new_failure_prob - self.prev_cdf:
            self.broken = True
        self.prev_cdf = new_failure_prob
            
    def get_component_features(
        self,
    ):
        """
        Get the component features.
        This function should be called to get the component's features.
        It will return a numpy array with the component's features, with the "broken" status as 0 or 1
        """
        broken = np.asarray([1.0]) if self.broken else np.asarray([0.0])
        return np.concatenate((broken, np.asarray([self.etf])), axis=0)
    
    def repair(
        self,
    ):
        """
        Repair the component.
        This function should be called to repair the component.
        It will set the component's state to "ok" and reset the Weibull distribution.
        """
        self.broken = False
        self.weibull_cdf = weibull_cdf_generator(self.wb_beta, self.wb_eta, start=0, step=1)
        self.etf = weibull_mean(self.wb_beta, self.wb_eta, self.wb_gamma)
    
    def reset(
        self,
    ):
        """
        Reset the component to its initial state.
        This function should be called to reset the component's state.
        """
        self.broken = False
        self.weibull_cdf = weibull_cdf_generator(self.wb_beta, self.wb_eta, start=0, step=1)
        self.etf = weibull_mean(self.wb_beta, self.wb_eta, self.wb_gamma)
        
        
    # Modify the __str__ method to return the component's state
    def __str__(self):
        """
        Return the component's state.
        This function should be called to get the component's state.
        It will return a string with the component's state.
        """
        return f"{self.name} - broken: {self.broken} - etf: {self.etf}"
        
    
class ComponentFactory:
    
    def __init__(self):
        """
        Initialize the component.
        This function should be called to initialize the component.
        It will initialize the component's parameters and the environment's hyperparameters.
        """
        pass
    @staticmethod
    def create_component(component_type:str, **kwargs) -> MachineComponent:
        """
        Create a component of the given type.
        This function should be called to create a component of the given type.
        It will return a component of the given type.
        """
        if component_type == "component":
            return MachineComponent(**kwargs)
        elif component_type == "rotor":
            return Rotor(**kwargs)
        elif component_type == "bearing":
            return Bearing(**kwargs)
        elif component_type == "pump":
            return Pump(**kwargs)
        elif component_type == "sensor":
            return Sensor(**kwargs)
        elif component_type == "calibration":
            return Calibration(**kwargs)
        else:
            raise ValueError(f"Unknown component type: {component_type}")
        
class Rotor(MachineComponent):
    
    name = "rotor"
    wb_beta = 1.5
    wb_eta = 30000
    wb_gamma = 0
    repair_time = 10
    repair_cost = 50
    
    def __init__(self, **kwargs):
        """
        Initialize the rotor.
        This function should be called to initialize the rotor.
        It will initialize the rotor's parameters and the environment's hyperparameters.
        """
        super().__init__(**kwargs)
    
class Bearing(MachineComponent):
    
    name = "bearing"
    wb_beta = 2.5
    wb_eta = 20000
    wb_gamma = 0
    repair_time = 5
    repair_cost = 20
    
    def __init__(self, **kwargs):
        """
        Initialize the bearing.
        This function should be called to initialize the bearing.
        It will initialize the bearing's parameters and the environment's hyperparameters.
        """
        super().__init__(**kwargs)

class Pump(MachineComponent):
    
    name = "pump"
    wb_beta = 2.0
    wb_eta = 25000
    wb_gamma = 0
    repair_time = 20
    repair_cost = 100
    
    def __init__(self, **kwargs):
        """
        Initialize the pump.
        This function should be called to initialize the pump.
        It will initialize the pump's parameters and the environment's hyperparameters.
        """
        super().__init__(**kwargs)
    
class Sensor(MachineComponent):
    
    name = "sensor"
    wb_beta = 1.0
    wb_eta = 50000
    wb_gamma = 0
    repair_time = 6
    repair_cost = 10
    
    def __init__(self, **kwargs):
        """
        Initialize the sensor.
        This function should be called to initialize the sensor.
        It will initialize the sensor's parameters and the environment's hyperparameters.
        """
        super().__init__(**kwargs)
    
class Calibration(MachineComponent):
    
    name = "calibration"
    wb_beta = 1.0
    wb_eta = 30000
    wb_gamma = 0
    repair_time = 30
    repair_cost = 150
    
    def __init__(self, **kwargs):
        """
        Initialize the calibration.
        This function should be called to initialize the calibration.
        It will initialize the calibration's parameters and the environment's hyperparameters.
        """
        super().__init__(**kwargs)