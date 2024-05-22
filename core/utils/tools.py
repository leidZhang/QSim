import io
import os
import sys
import time
import torch
import mlflow
import logging
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List
from mlflow.store.artifact.artifact_repo import ArtifactRepository

class LogColorFormatter(logging.Formatter):
    GREY = '\033[90m'
    WHITE = '\033[37m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    RED_UNDERLINE = '\033[4;31m'

    def __init__(
        self,
        fmt,
        debug_color=GREY,
        info_color=None,
        warning_color=YELLOW,
        error_color=RED,
        critical_color=RED_UNDERLINE
    ):
        super().__init__(fmt)
        self.fmt = fmt
        self.debug_color = debug_color
        self.info_color = info_color
        self.warning_color = warning_color
        self.error_color = error_color
        self.critical_color = critical_color

    def format(self, record):
        RESET = '\033[0m'
        if record.levelno == logging.DEBUG:
            fmt = f'{self.debug_color or ""}{self.fmt}{RESET}'
        elif record.levelno == logging.INFO:
            fmt = f'{self.info_color or ""}{self.fmt}{RESET}'
        elif record.levelno == logging.WARNING:
            fmt = f'{self.warning_color or ""}{self.fmt}{RESET}'
        elif record.levelno == logging.ERROR:
            fmt = f'{self.error_color or ""}{self.fmt}{RESET}'
        elif record.levelno == logging.CRITICAL:
            fmt = f'{self.critical_color or ""}{self.fmt}{RESET}'
        else:
            fmt = self.fmt
        return logging.Formatter(fmt).format(record)

def configure_logging(prefix='[%(name)s]', level=logging.DEBUG, info_color=None):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        LogColorFormatter(
            f'{prefix}  %(message)s',
            info_color=info_color
        )
    )
    logging.root.setLevel(level)
    logging.root.handlers = [handler]
    for logname in ['urllib3', 'requests', 'mlflow', 'git', 'azure', 'PIL', 'numba', 'google.auth', 'rosout']:
        logging.getLogger(logname).setLevel(logging.WARNING)  # disable other loggers
    for logname in ['absl', 'minerl']:
        logging.getLogger(logname).setLevel(logging.INFO)

def mlflow_init(mlruns_dir):
    run_name = os.environ.get("MLFLOW_RUN_NAME")
    resume_id = os.environ.get("MLFLOW_RESUME_ID")
    uri = os.environ.get("MLFLOW_TRACKING_URI", mlruns_dir)
    mlflow.set_tracking_uri(uri)

    run = mlflow.active_run()
    if run:
        pass
    elif os.environ.get("MLFLOW_RUN_ID"):
        run = mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID"))
        logging.info(f'Reinitialized mlflow run {run.info.run_id} ({resume_id}) in {uri}/{run.info.experiment_id}')
    else:
        resume_run_id = None
        if resume_id:
            runs = mlflow.search_runs(filter_string=f'tags.resume_id={resume_id}')
            if len(runs) > 0:
                resume_run_id = runs.run_id.iloc[0]

        if resume_run_id:
            run = mlflow.start_run(run_id=resume_run_id)
            logging.info(f'Resumed mlflow run {run.info.run_id} ({resume_id}) in {uri}/{run.info.experiment_id}')
        else:
            run = mlflow.start_run(run_name=run_name, tags={"resume_id": resume_id or ""})
            logging.info(f'Started mlflow run {run.info.run_id} ({resume_id}) in {uri}/{run.info.experiment_id}')

    os.environ["MLFLOW_RUN_ID"] = run.info.run_id
    return run

def load_checkpoint(model, mlruns_dir, run_id, map_location="cpu"):
    path = Path(mlruns_dir[8:]) / "0" / run_id / "latest_checkpoint.pt"
    try:
        checkpoint = torch.load(path, map_location=map_location)
    except Exception as e:
        logging.exception('Error reading checkpoint')
        return None

    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint["epoch"]


# def load_checkpoint(model, mlruns_dir, run_id, map_location=None):
#     # 动态设置 map_location，如果没有指定，则根据 cuda 的可用性自动设置
#     if map_location is None:
#         map_location = "cuda" if torch.cuda.is_available() else "cpu"
#     path = Path(mlruns_dir[8:]) / "0" / run_id / "latest_checkpoint.pt"
#
#     try:
#         logging.info(f"Loading checkpoint from {path} with map_location set to {map_location}")
#         checkpoint = torch.load(path, map_location=map_location)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         logging.info("Checkpoint loaded successfully and model state dict has been set.")
#         return checkpoint["epoch"]
#     except Exception as e:
#         logging.exception('Error reading or setting the checkpoint.')
#         return None

def mlflow_load_checkpoint(model, run_id=None, optimizers=tuple(), artifact_path="checkpoints/latest.pt", map_location=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        run_id = run_id if run_id is not None else mlflow.active_run().info.run_id

        try:
            path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path, dst_path=tmpdir)
        except Exception as e:
            return None

        try:
            checkpoint = torch.load(path, map_location=map_location)
        except:
            logging.exception('Error reading checkpoint')
            return None

        model.load_state_dict(checkpoint['model_state_dict'])
        for i, opt in enumerate(optimizers):
            opt.load_state_dict(checkpoint[f'optimizer_{i}_state_dict'])

        return checkpoint["epoch"]

def mlflow_save_checkpoint(model, optimizers, steps, mlruns_dir=None, run_id=None):
    if mlruns_dir is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'latest.pt'
            path_step = Path(tmpdir) / 'step_{}.pt'.format(steps)
            checkpoint = {}
            checkpoint['epoch'] = steps
            checkpoint['model_state_dict'] = model.state_dict()
            for i, opt in enumerate(optimizers):
                checkpoint[f'optimizer_{i}_state_dict'] = opt.state_dict()
            torch.save(checkpoint, path)
            mlflow_log_artifact(path, subdir='checkpoints')

            if steps % 10000 == 0:
                torch.save(checkpoint, path_step)
                mlflow_log_artifact(path_step, subdir='checkpoints')
    else:
        assert run_id is not None, "Need to provide run_id if providing direct mlrun path"

        path = Path(mlruns_dir) / "0" / run_id / "latest.pt"
        checkpoint = {}
        checkpoint['epoch'] = steps
        checkpoint['model_state_dict'] = model.state_dict()
        for i, opt in enumerate(optimizers):
            checkpoint[f'optimizer_{i}_state_dict'] = opt.state_dict()
        torch.save(checkpoint, path)
        mlflow_log_artifact(mlflow_path, subdir='checkpoints')

def mlflow_log_metrics(metrics: dict, step: int, run_id: str = None):
    while True:
        try:
            mlflow.log_metrics(metrics, step=step, run_id=run_id)
            break
        except:
            logging.exception('Error logging metrics - will retry.')
            time.sleep(10)

def mlflow_log_npz(data: dict, name, subdir=None, verbose=False, repository: ArtifactRepository = None, mlruns_dir = None):
    if mlruns_dir is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = Path(tmpdir) / name
            save_npz(data, tmpfile)
            mlflow_log_artifact(tmpfile, subdir, verbose, repository)
    else:
        path = Path(mlruns_dir) / name
        save_npz(data, path)
        mlflow_log_artifact(path, subdir, verbose, repository)

def save_npz(data, path):
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **data)
        f1.seek(0)
        with path.open('wb') as f2:
            f2.write(f1.read())

def mlflow_log_artifact(path: Path, subdir=None, verbose=True, repository: ArtifactRepository = None):
    if verbose:
        logging.debug(f'Uploading artifact {subdir}/{path.name} size {path.stat().st_size/1024/1024:.2f} MB')
    while True:
        try:
            if repository:
                repository.log_artifact(str(path), artifact_path=subdir)
            else:
                mlflow.log_artifact(str(path), artifact_path=subdir)
            break
        except:
            logging.exception('Error saving artifact - will retry.')
            time.sleep(10)

def mlflow_load_npz(name, repository: ArtifactRepository):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / name
        repository._download_file(name, tmpfile)
        return load_npz(tmpfile)

def load_npz(path, keys=None) -> Dict[str, np.ndarray]:
    with path.open('rb') as f:
        fdata: Dict[str, np.ndarray] = np.load(f)

        if keys is None:
            data = {key: fdata[key] for key in fdata}
        else:
            data = {key: fdata[key] for key in keys}

        return data

def elapsed_time(start_time: float) -> float:
    return time.time() - start_time