# QSim: A Framework for Developing and Testing Autonomous Driving Algorithms
## Introduction
This project is a comprehensive framework designed for developing and testing autonomous driving algorithms. It leverages Quanser Interactive Labs (QLabs) for realistic simulation environments and integrates various components such as control policies, environment simulators, and data handling utilities.
## Features
- Control Policies: Includes a variety of control policies such as vision lane following and pure pursuit, as well as a customizable framework for adding new control policies.
- Environment Simulation: Tools for setting up and managing simulation environments, including traffic objects, cross-road scenarios, and online training demos.
- Data Handling: Utilities for handling data preprocessing, model training, and online training data generation.
- Traffic Objects: Management of traffic lights, stop signs, and other traffic objects within the simulation.
- Hardware Compatability: Supports both Windows and Linux operating systems, and can be used with both QCar hardware and virtual hardware.
## Installation
To set up the QSim project, ensure you have the necessary dependencies installed. This project typically requires Python 3.6+ and several libraries including numpy, torch, cv2
```
pip install -r requirements.txt
```
## Demos and Tests
This project includes several demos for different use cases, you can view the code in the `demos` directory. The demos include:
- Render Demo: A simple demo that renders the simulation environment.
- Cross-Road Demo: A demo that simulates a cross-road environment without traffic lights, 1 ego agent, and 3 hazard agents are spawned.
- Train Env Demo: A demo that sets up a training environment with traffic lights, 1 ego agent, and 1 hazard agents.
- Micro Kernel Demo: A demo that demonstrates the pseudo-RTOS architecture for the QCar.
- Online Training Demo: A demo that generates training data and trains a model using this data.
- Traffic Object Demo: A demo that demonstrates the management of traffic objects within the simulation.<br>

To run a demo, simply run the `main.py` with arguments specifying the demo to run. For example, to run the cross-road demo, run the following command:
```
python main.py --run cross_road_demo
```
The project also includes several unit tests for testing the functionality of the different components. To run the unit tests, run the following command:
```
python main.py --run tests
```
## Usage
### Setting Up the Environment
To start the oneline training, you need to set up the simulation environment. This involves initializing the QLabs simulator, configuring the roadmap, and spawning actors.
```
from qvl.qlabs import QuanserInteractiveLabs
from core.environment.builder import GeneralMapBuilder
from core.environment.director import GeneralDirector
from core.environment.simulator import QLabSimulator
from core.environment.constants import FULL_CONFIG

qlabs = QuanserInteractiveLabs()
qlabs.open("localhost")

builder = GeneralMapBuilder(qlabs)
director = GeneralDirector(builder, False)
sim = QLabSimulator([0, 0], False)
sim.render_map(director, FULL_CONFIG)
```
### Running a Simulation
Once the environment is set up, you can run a simulation by defining agents and their policies. For example, in the cross-road demo, we can define an ego agent and hazard agents, and use the PurePursuiteAdaptor policy to run a simulation with an ego agent and 3 hazard agents:
```
from core.environment.builder import GeneralMapBuilder
from core.environment.director import GeneralDirector
from core.environment.simulator import QLabSimulator
from core.policies import PurePursuiteAdaptor
from .environment import CrossRoadEnvironment, OnlineQLabEnv

env: OnlineQLabEnv = CrossRoadEnvironment(sim, roadmap, privileged=True)
env.set_ego_policy(PurePursuiteAdaptor())
env.set_hazard_policy(PurePursuiteAdaptor())

for i in range(5):
    print(f"Starting episode {i}...")
    env.reset() # use this to set the agents to their initial positions
    for _ in range(200):
        _, _, done, _ = env.step()
        if done:
            print(f"Episode {i + 1} complete")
            break
    env.stop_all_agents()
    time.sleep(1)
```
### Online Training
This project supports online training of models. In the online training demo, the DemoGenerator class is used to generate training data, and the DemoTrainer class is used to train models using this data.
```
import time
import logging
from typing import List
from multiprocessing import Process, Queue, Event

from demos.online_training.system import configure_logging
from demos.online_training.generator import DemoGenerator
from demos.online_training.td3 import DemoTrainer


def run_trainer(event, queue: Queue) -> None:
    trainer: DemoTrainer = DemoTrainer(queue)
    event.wait() # wait for the signal to start training
    print("Start training...")
    while not trainer.done:
        trainer.train()


def run_generator(event, queue: Queue) -> None:
    generator: DemoGenerator = DemoGenerator(event, queue)
    generator.generate()


def check_process(processes: List[Process]) -> None:
    for process in processes:
        # skip the process if it is still running
        if process.is_alive():
            continue
        # remove the process if it is not running
        if process.exitcode == 0:
            processes.remove(process)
            logging.info(f"Process {process.pid} exited with code {process.exitcode}")
        else:
            raise Exception(f"Process {process.pid} exited with code {process.exitcode}")
```
## Hardware Compatability
This project refactored the hardware API to support the asynchonous architecture of the system. By default, the ego agent uses the customed sensor API and therefore it can be transferred to the hardware QCar with little modification (replace the VirtualOptitrack with the UDPClient to get the ego state data from the Optitrack).