import numpy as np

from src.simulation.process_model import ThermalProcessModel, simple_input_profile


def test_simulation_runs():
    model = ThermalProcessModel(random_state=0)
    inputs = simple_input_profile(10)
    disturbances = np.zeros_like(inputs)
    x0 = np.array([500.0, 0.0])
    states, outputs = model.simulate(x0, inputs, disturbances, dt=1.0)
    assert states.shape[0] == 11
    assert outputs.shape[0] == 11
