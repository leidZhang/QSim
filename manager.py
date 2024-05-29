import logging
from typing import List
from multiprocessing import Process, Event


class ExperimentManager: 
    def __init__(self) -> None:
        self.processes: List[Process] = []
        self.episode_events: list = [] # list of events

    def start_generator(self) -> None:
        pass

    def start_trainer(self) -> None:
        pass
    
    def check_processes(self) -> None:
        for process in self.processes:
            # skip the process if it is still running
            if process.is_alive():
                continue

            # remove the process if it is not running
            if process.exitcode == 0:
                self.processes.remove(process)
                logging.info(f"Process {process.pid} exited with code {process.exitcode}")
            else:
                raise Exception(f"Process {process.pid} exited with code {process.exitcode}")
    
    def start_experiment(self) -> None:
        pass