import os
import time
import logging
from multiprocessing import Process

import mlflow
from typing import List

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs

from core.roadmap import ACCRoadMap
from td3.generator import Generator
from td3.trainer import Trainer
from td3.constants import PREFILL, MAX_TRAINING_STEPS
from td3.exceptions import InsufficientDataException, StopTrainingException
from core.utils.tools import mlflow_init, configure_logging

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

def start_generator(
    run_id: str,
    mlruns_dir: str, 
    train_repo: str, 
    eval_repo: str,
    init_pos: list,
    waypoints: np.ndarray,
    resume: bool = False,
    privileged: bool = True
) -> None: 
    print("Creating generator instance...")
    generator: Generator = Generator(
        mlruns_dir=mlruns_dir,
        train_repo=train_repo,
        qcar_pos=init_pos,
        waypoints=waypoints,
        eval_repo=eval_repo,
        privileged=privileged
    )
    print("Preparing collection session...")
    generator.execute(run_id=run_id, resume=resume)

def start_trainer(
    run_id: str,
    mlruns_dir: str,
    device: str, 
    prefill_steps: int,
    init_pos: list,
    waypoints: np.ndarray,
    resume: bool = False
) -> None: 
    print("Creating trainer instance...")
    trainer: Trainer = Trainer(
        mlruns_dir=mlruns_dir, 
        run_id=run_id, 
        device=device,
        qcar_pos=init_pos,
        waypoints=waypoints,
        prefill_steps=prefill_steps
    )
    trainer.prepare_training(resume=resume)
    stop_training: bool = False
    while not stop_training: 
        try: 
            trainer.execute()
        except InsufficientDataException: 
            logging.info(f"Insufficient data sampled:[ {len(trainer.data)}/{PREFILL} ]")
            time.sleep(20)
        except StopTrainingException: 
            logging.info(f'Finished {MAX_TRAINING_STEPS} grad steps.')
            stop_training = True

def prepare_map_info(node_id: int = 24) -> tuple:
    roadmap: ACCRoadMap = ACCRoadMap()
    qlabs = QuanserInteractiveLabs()
    x_pos, y_pose, angle = roadmap.nodes[node_id].pose
    waypoint_sequence = roadmap.generate_path([10, 4, 14, 20, 22, 10])
    return [x_pos, y_pose, angle], waypoint_sequence

def start_system(resume_run_id: str, init_pos: list, waypoints: np.ndarray) -> None:
    # setup mlflow directory
    root_directory: str = os.getcwd() # get the absolute directory
    mlflow_directory: str = os.path.join(root_directory, 'mlruns')
    mlflow_directory = 'file:///' + mlflow_directory.replace('\\', '/')

    # configure logging
    configure_logging("[launcher]")
    # starting from existing checkpoint or not
    designated_run_id: str = None
    resume: bool = False
    if resume_run_id != '': # resume from existing checkpoint
        designated_run_id = resume_run_id
        resume = True
    else: # start from new checkpoint
        mlrun = mlflow_init(mlflow_directory)
        designated_run_id = mlrun.info.run_id

    print(f"Started MLFlow Run: {designated_run_id}")
    # start generator process
    artifact_uri: str = mlflow_directory + f'/0/{designated_run_id}/artifacts'
    train_repo: str = f'{artifact_uri}/episodes_train/0'
    eval_repo: str = f'{artifact_uri}/episodes_eval/0'
    logging.info(f"Started New MLFlow Run: {artifact_uri}")
    # initialize processes
    processes: List[Process] = []
    generator_process: Process = Process(
        target=start_generator, 
        daemon=True,
        kwargs=dict(
            mlruns_dir=mlflow_directory,
            run_id=designated_run_id,
            train_repo=train_repo,
            eval_repo=eval_repo,
            init_pos=init_pos,
            waypoints=waypoints,
            resume=resume,
            privileged=True
        )
    )
    processes.append(generator_process)
    trainer_process: Process = Process(
        target=start_trainer, 
        daemon=False, 
        kwargs=dict(
            mlruns_dir=mlflow_directory, 
            run_id=designated_run_id,
            init_pos=init_pos,
            waypoints=waypoints,
            device="cuda:0",
            prefill_steps=0,
        )
    )
    processes.append(trainer_process)
    # start process
    generator_process.start()
    trainer_process.start()
    # check process
    check_process(processes=processes)
    generator_process.join()
    trainer_process.join()

if __name__ == '__main__':
    resume_run_id = ''
    init_pos, waypoints = prepare_map_info(node_id=24)
    start_system(resume_run_id=resume_run_id, init_pos=init_pos, waypoints=waypoints)