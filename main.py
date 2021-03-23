import os

from simulation import Simulation

if __name__ == '__main__':
    os.makedirs("./plots", exist_ok=True)
    simulation = Simulation()
    simulation.simulate(start_time="2020-04-30",
                        end_time="2020-12-26",
                        c=0.15)
