from itertools import product

TECH_RULES = [
    "Least Fatigued Technician",
    "Most Knowledgeable Technician",
]

MACHINE_RULES = [
    "Most Broken Machine",
    "Least In-Buffered Machine",
    "Most In-Buffered Machine",
    #"Least Productive Machine",
]

COMPONENT_RULES = [
    "All Components",
    "Sensor",
    "Calibration",
    "Pump",
    "Bearing",
    "Rotor"
]

# Create cross product of TECH_RULES x MACHINE_RULES x COMPONENT_RULES
DISPATCHING_RULES = list(product(TECH_RULES, MACHINE_RULES, COMPONENT_RULES))
DISPATCHING_RULES.insert(0, ("No maintenance",))