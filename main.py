import os

from simulation import Simulation

if __name__ == '__main__':
    os.makedirs("./plots", exist_ok=True)
    simulation = Simulation()
    simulation.run()
