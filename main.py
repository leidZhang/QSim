import os
import time
import logging
from typing import List, Dict
from multiprocessing import Process

import numpy as np

from core.roadmap import ACCRoadMap
from td3.generator import Generator
from td3.trainer import Trainer
from constants import PREFILL, MAX_TRAINING_STEPS, RUN_ID
from td3.exceptions import InsufficientDataException, StopTrainingException
from core.utils.tools import mlflow_init, configure_logging
import constants as C

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
    nodes: Dict[str, np.ndarray],
    waypoints: np.ndarray,
    resume: bool = False,
    privileged: bool = True
) -> None:
    generator: Generator = Generator(
        mlruns_dir=mlruns_dir,
        train_repo=train_repo,
        nodes=nodes,
        waypoints=waypoints,
        eval_repo=eval_repo,
        privileged=privileged
    )
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
    trainer: Trainer = Trainer(
        mlruns_dir=mlruns_dir,
        run_id=run_id,
        device=C.cuda,
        qcar_pos=init_pos,
        waypoints=waypoints,
        prefill_steps=prefill_steps
    )

    trainer.prepare_training(resume=resume)
    stop_training: bool = False
    counter = 0
    try:
        while not stop_training:
            try:
                trainer.execute()
            except InsufficientDataException:
                if counter % 4 == 0:
                    logging.info(f"Insufficient data sampled:[ {len(trainer.data)}/{PREFILL} ]")
                time.sleep(5)
                counter += 1
            except StopTrainingException:
                logging.info(f'Finished {MAX_TRAINING_STEPS} grad steps.')
                stop_training = True
    finally:
        if not stop_training:
            trainer.execute(True)

def prepare_map_info(node_sequence: list) -> tuple:
    roadmap: ACCRoadMap = ACCRoadMap()
    # x_pos, y_pose, angle = roadmap.nodes[node_id].pose
    # waypoint_sequence = roadmap.generate_path([10, 4, 14, 20, 22, 10])
    nodes: Dict[str, np.ndarray] = {}
    for node_id in node_sequence:
        pose: np.ndarray = roadmap.nodes[node_id].pose
        nodes[node_id] = pose # x, y, angle

    waypoint_sequence = roadmap.generate_path(node_sequence)
    return nodes, waypoint_sequence

def start_system(resume_run_id: str, nodes: Dict[str, np.ndarray], waypoints: np.ndarray) -> None:
    # setup mlflow directory
    root_directory: str = os.getcwd() # get the absolute directory
    mlflow_directory: str = os.path.join(root_directory, 'mlruns')
    mlflow_directory = 'file:///' + mlflow_directory.replace('\\', '/')
    # mlflow_directory: str = os.path.join(root_directory, 'mlruns.db')
    # mlflow_directory = "sqlite:///" + mlflow_directory.replace('\\', '/')

    # configure logging
    configure_logging("[LAUNCHER]")
    # starting from existing checkpoint or not
    designated_run_id: str = None
    resume: bool = False
    if resume_run_id != '': # resume from existing checkpoint
        designated_run_id = resume_run_id
        resume = True
    else: # start from new checkpoint
        mlrun = mlflow_init(mlflow_directory)
        designated_run_id = mlrun.info.run_id

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
            nodes=nodes,
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
            init_pos=None,
            waypoints=waypoints,
            resume=resume,
            device=C.cuda,
            prefill_steps=0,
        )
    )
    processes.append(trainer_process)
    # start process
    generator_process.start()
    trainer_process.start()
    # check process
    try:
        while True:
            check_process(processes=processes)
            time.sleep(5)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    # fill with file name of your experiment, set to '' to start new experiment
    resume_run_id = RUN_ID
    nodes, waypoints = prepare_map_info(node_sequence=[4, 14, 20])
    start_system(resume_run_id=resume_run_id, nodes=nodes, waypoints=waypoints)