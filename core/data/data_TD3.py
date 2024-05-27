import time
import logging
from queue import Queue
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, IterableDataset
from typing import Dict, List, Union, Optional
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

#project imports
from core.utils.tools import mlflow_log_npz, mlflow_load_npz
from core.utils.aggregation_utils import cat_structure_np, stack_structure_np, map_structure


@dataclass(frozen=True)
class FileInfo:
    path: str
    episode_from: int
    episode_to: int
    steps: int
    artifact_repo: ArtifactRepository

    def load_data(self) -> Dict[str, np.ndarray]:
        data = mlflow_load_npz(self.path, self.artifact_repo)
        return data

    def __repr__(self) -> str:
        return f'{self.path}'


class EpisodeRepository(ABC):
    @abstractmethod
    def save_data(self, data: Dict[str, np.ndarray], episode_from: int, episode_to: int):
        ...

    @abstractmethod
    def list_files(self) -> List[FileInfo]:
        ...


class DataRepository(ABC):
    @abstractmethod
    def save_data(self, data: Dict[str, np.ndarray], n_samples: int):
        ...

    @abstractmethod
    def list_files(self) -> List[FileInfo]:
        ...


class MlflowEpisodeRepository(EpisodeRepository):
    def __init__(self, artifcat_uris: Union[str, List[str]]):
        super().__init__()
        self.artifact_uris = [artifcat_uris] if isinstance(artifcat_uris, str) else artifcat_uris
        self.read_repos: List[ArtifactRepository] = [get_artifact_repository(uri) for uri in self.artifact_uris]
        self.write_repo = self.read_repos[0]

    def save_data(self, data: Dict[str, np.ndarray], episode_from: int, episode_to: int, chunk_seq: Optional[int] = None):
        n_episodes = data["reset"].sum()
        n_steps = len(data["reset"]) - n_episodes
        reward = data["reward"].sum()
        fname = self.build_episode_name(episode_from, episode_to, reward, n_steps, chunk_seq=chunk_seq)
        mlflow_log_npz(data, fname, repository=self.write_repo)

    def list_files(self) -> List[FileInfo]:
        while True:
            try:
                return self._list_files()
            except:
                logging.exception('Error listing files - will retry.')
                time.sleep(10)

    def _list_files(self) -> List[FileInfo]:
        files = []
        for repo in self.read_repos:
            for f in repo.list_artifacts(''):
                if f.path.endswith('.npz') and not f.is_dir:
                    (ep_from, ep_to, steps) = self.parse_episode_name(f.path)
                    files.append(FileInfo(
                        path=f.path,
                        episode_from=ep_from,
                        episode_to=ep_to,
                        steps=steps,
                        artifact_repo=repo
                    ))

        return files

    def count_steps(self):
        files = self.list_files()
        steps = sum(f.steps for f in files)
        episodes = (max(f.episode_to for f in files) + 1) if files else 0
        return len(files), steps, episodes

    def build_episode_name(self, episode_from, episode, reward, steps, chunk_seq=None):
        if chunk_seq is None:
            return f'ep{episode_from:06}_{episode:06}-r{reward:.0f}-{steps:04}.npz'
        else:
            return f'ep{episode_from:06}_{episode:06}-{chunk_seq}-r{reward:.0f}-{steps:04}.npz'

    def parse_episode_name(self, fname):
        fname = fname.split("/")[-1].split(".")[0]
        if fname.startswith('ep'):
            # fname = 'ep{epfrom}_{episode}-r{reward}-{steps}.npz'
            #       | 'ep{episode}-r{reward}-{steps}.npz'
            steps = fname.split('-')[-1]
            steps = int(steps) if steps.isnumeric() else 0
            ep_from = fname.split('ep')[1].split('-')[0].split('_')[0]
            ep_from = int(ep_from) if ep_from.isnumeric() else 0
            ep_to = fname.split('ep')[1].split('-')[0].split('_')[-1]
            ep_to = int(ep_to) if ep_to.isnumeric() else 0
            return (ep_from, ep_to, steps)
        else:
            # fname = '{timestamp}-{steps}
            steps = fname.split('-')[-1]
            steps = int(steps) if steps.isnumeric() else 0
            return (0, 0, steps)

    def __repr__(self) -> str:
        return f'{self.artifact_uris}'


# data repository
class MlflowDataRepository(DataRepository):
    def __init__(self, artifcat_uris: Union[str, List[str]]):
        super().__init__()
        self.artifact_uris = [artifcat_uris] if isinstance(artifcat_uris, str) else artifcat_uris
        self.read_repos: List[ArtifactRepository] = [get_artifact_repository(uri) for uri in self.artifact_uris]
        self.write_repo = self.read_repos[0]

        self.data_quantity = 0

    def save_data(self, data: Dict[str, np.ndarray], n_samples: int):
        for i in range(n_samples):
            sample = {}
            sample["image"] = data["image"][i]
            sample["waypoints"] = data["waypoints"][i]

            fname = str(self.data_quantity).zfill(6) + ".npz"
            mlflow_log_npz(sample, fname, repository=self.write_repo)
            self.data_quantity += 1

    def list_files(self) -> List[FileInfo]:
        while True:
            try:
                return self._list_files()
            except:
                logging.exception('Error listing files - will retry.')
                time.sleep(10)

    def _list_files(self) -> List[FileInfo]:
        files = []
        for repo in self.read_repos:
            for f in repo.list_artifacts(''):
                if f.path.endswith('.npz') and not f.is_dir:
                    #(ep_from, ep_to, steps) = self.parse_episode_name(f.path)
                    files.append(FileInfo(
                        path=f.path,
                        episode_from=0, #ep_from,
                        episode_to=0, #ep_to,
                        steps=0, #steps,
                        artifact_repo=repo
                    ))

        return files

    def count_data(self):
        files = self.list_files()
        return len(files)

    def build_episode_name(self, episode_from, episode, reward, steps, chunk_seq=None):
        if chunk_seq is None:
            return f'ep{episode_from:06}_{episode:06}-r{reward:.0f}-{steps:04}.npz'
        else:
            return f'ep{episode_from:06}_{episode:06}-{chunk_seq}-r{reward:.0f}-{steps:04}.npz'

    def parse_episode_name(self, fname):
        fname = fname.split("/")[-1].split(".")[0]
        if fname.startswith('ep'):
            # fname = 'ep{epfrom}_{episode}-r{reward}-{steps}.npz'
            #       | 'ep{episode}-r{reward}-{steps}.npz'
            steps = fname.split('-')[-1]
            steps = int(steps) if steps.isnumeric() else 0
            ep_from = fname.split('ep')[1].split('-')[0].split('_')[0]
            ep_from = int(ep_from) if ep_from.isnumeric() else 0
            ep_to = fname.split('ep')[1].split('-')[0].split('_')[-1]
            ep_to = int(ep_to) if ep_to.isnumeric() else 0
            return (ep_from, ep_to, steps)
        else:
            # fname = '{timestamp}-{steps}
            steps = fname.split('-')[-1]
            steps = int(steps) if steps.isnumeric() else 0
            return (0, 0, steps)

    def __repr__(self) -> str:
        return f'{self.artifact_uris}'

class OfflineDataset(Dataset):
    def __init__(
        self,
        repository: MlflowDataRepository,
        batch_size: int = 64
    ):
        super().__init__()
        self.repository = repository
        self.batch_size = batch_size

        self.files = self.repository.list_files()

    def __len__(self):
        return self.repository.count_data()

    def __getitem__(self, idx):
        try:
            data = self.files[idx].load_data()
        except Exception as e:
            print("Error loading file!")
            print(e)
            print(idx)

        return data

class QLabsDataset(IterableDataset):
    def __init__(
        self,
        repository: MlflowDataRepository,
        batch_length: int,
        batch_size: int,
        buffer_size: int = 0,
        reload_interval: int = 120,
        allow_mid_reset: bool = True
    ):
        super().__init__()
        self.repository = repository
        self.batch_length = batch_length
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.reload_interval = reload_interval
        self.allow_mid_reset = allow_mid_reset
        self.reload_files()

    def reload_files(self):
        files_all = self.repository.list_files()
        files_all.sort(key = lambda e: -e.episode_to)

        files = []
        steps_total = 0
        steps_filtered = 0
        for f in files_all:
            steps_total += f.steps
            if steps_total < self.buffer_size or not self.buffer_size:
                files.append(f)
                steps_filtered += f.steps

        self.files: List[FileInfo] = files
        self.last_reload = time.time()
        self.stats_steps = steps_total

    def should_reload_files(self):
        return self.reload_interval and (time.time() - self.last_reload > self.reload_interval)

    #iterate dataset to get batch
    def __iter__(self):
        iters = [self.iter_single(batch_idx) for batch_idx in range(self.batch_size)]

        for batches in zip(*iters):
            batch = stack_structure_np(batches)
            batch = map_structure(batch, lambda d: d.swapaxes(0, 1))
            yield batch

    #TODO: Finish implementing this function
    def iter_single(self, batch_idx):
        last_partial_batch = None

        for file in self.iter_shuffled_files():
            if last_partial_batch:
                first_shorter_length = self.batch_length - last_partial_batch["reward"].shape[0]
            else:
                first_shorter_length = None

            it = self.iter_file(file, self.batch_length, first_shorter_length)

            #concatenate end of partial batch with beginning of next batch to create a full sequence
            if last_partial_batch is not None:
                for batch, partial in it:
                    assert not partial, 'First batch must be full. Is episode_length < batch_size?'
                    batch = cat_structure_np([last_partial_batch, batch])
                    assert batch["reward"].shape[0] == self.batch_length
                    last_partial_batch = None
                    yield batch
                    break

            for batch, partial in it:
                if partial:
                    if self.allow_mid_reset:
                        last_partial_batch = batch
                    else:
                        # If no mid-reset, just ignore the partial batch, and next batch will be from beginning
                        last_partial_batch = None
                    break  # partial will always be last
                yield batch

    def iter_file(self, file: FileInfo, batch_length, first_shorter_length=False):
        #load data from file
        try:
            data = file.load_data()
        except Exception as e:
            print("error reading file")
            print(e)
            return

        #undo image transformation from generator
        if "image" not in data and "image_t" in data:
            data["image"] = data["image_t"].transpose(3, 0, 1, 2) #HWCT => THWC
            del data["image_t"]

        #if file doesn't contain enough timesteps to create a batch, then skip this file
        n = data["reward"].shape[0]
        if n < batch_length:
            logging.info(f"Skipping too short file: {file}, len={n}")
            return

        #File must start with reset and have 0 reward
        data["reset"][0] = True
        data["reward"][0] = 0.0

        i = 0
        l = first_shorter_length or batch_length
        while i < n:
            batch = {key: data[key][i:i + l] for key in data}
            is_partial = batch["reward"].shape[0] < l
            i += l
            l = batch_length

            yield batch, is_partial

    #keep iterating files and randomly selecting one
    def iter_shuffled_files(self):
        while True:
            try:
                if self.should_reload_files():
                    self.reload_files()

                f = np.random.choice(self.files)
                yield f
            except Exception as e:
                print("CANT LOAD FILE {}".format(e))
                time.sleep(1)

# class SequenceRolloutBuffer(IterableDataset):
class SequenceRolloutBuffer:
# 用于存储和采样一系列轨迹（rollouts）的 replay buffer
    def __init__(self,
                 repository: MlflowEpisodeRepository,
                 batch_size: int,
                 update_rate: int,
                 observation_shape: tuple,
                 action_dim: int):

        self.repository = repository  # 仓库
        self.batch_size = batch_size  # 批次大小
        self.update_rate = update_rate  # 更新频率
        self.obs_shape = observation_shape
        self.action_dim = action_dim
        self.last_load_time = time.time()  # 上次加载时间
        self.end = 0  # mark
        self.buffer_size = 1000000
        self.full = False
        self.buffer_set: set = set()
        self.current_file = None
        self.pos = 0
        self.stats_steps = 0
        self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, *self.obs_shape), dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=bool)

    def __len__(self):
        return self.buffer_size if self.full else self.pos

    # TODO: Improve this function
    def reload_files(self):
    # file --> list
    # 从一个存储库（repository）中加载, 降序排列以及更新文件
    # 并将它们添加到 self.files 列表中，直到累积的step数达到上限 (self.buffer_size)
        files_all = self.repository.list_files()
        files_all.sort(key = lambda e: -e.episode_to)
        self.files: List[FileInfo] = files_all
        # print(self.files)
        self.last_reload = time.time()
        # self.stats_steps = steps_total

    def file_to_batch(self):
    # file --> list --> buffer --> batch
    # 按频率，更新文件列表，并加载列表文件数据到buffer
        # print(f"timer: {time.time() - self.last_load_time}")
        if time.time() - self.last_load_time > self.update_rate:
            # print("Loading file to batch...")
            self.reload_files()  # file --> list
            self.parse_and_load_buffer()  # list --> buffer
            self.last_load_time = time.time()

        return self.sample(self.batch_size)  # buffer --> batch

    def load_file(self, index):
    # 根据索引加载单个文件，返回一个包含轨迹数据的字典
        episode = self.files[index].load_data()
        return episode

    def parse_and_load_buffer(self):
        total_steps: int = 0
        buffer_queue: Queue = Queue()
        waste_queue: Queue = Queue()
        for i in range(len(self.files) - 1, -1, -1):
            total_steps += self.files[i].steps
            file = self.files[i] # file, index
            buffer_queue.put(file)
            if total_steps > self.buffer_size:
                pop_file = buffer_queue.get()
                waste_queue.put(pop_file)
                total_steps -= pop_file.steps

        while not buffer_queue.empty():
            file = buffer_queue.get()
            if file in self.buffer_set:
                continue # skip if is duplicate

            # self.pos = 0 # reset pos???
            episode = file.load_data()
            self.buffer_set.add(file)
            for t in range(episode['state'].shape[0] - 1):
                state = episode["state"][t]
                action = episode["action"][t + 1]
                next_state = episode["state"][t + 1]
                reward = episode["reward"][t + 1]
                done = episode["terminal"][t + 1]
                self.add(state, action, reward, next_state, done)

        while not waste_queue.empty(): # for resume
            waste = waste_queue.get()
            if waste in self.buffer_set:
                self.buffer_set.remove(waste)

    def add(self, state, action, reward, next_state, done):
        # add a new step data to buffer
        # update index and pos
        # after full, replace buffer data from begin
        # print(f"state: {state}, next_state: {next_state}")
        index = self.pos % self.buffer_size  # 取余数
        self.observations[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True

    def sample(self, batch_size: int):
    # buffer --> batch
        max_ind = self.buffer_size if self.full else self.pos  # 同 __len__
        if max_ind >= batch_size:  # 确保有数据可供采样
            indices = np.random.choice(max_ind, size=batch_size, replace=False)  # max里随机选size个不同的数字 组成一个数组
            return {
                'states': self.observations[indices],  # 获取一组数据 存到对应张量里 再赋值给返回字典里对应的键
                'actions': self.actions[indices],
                'rewards': self.rewards[indices],
                'next_states': self.next_states[indices],
                'dones': self.dones[indices]
            }
        else:
            # 处理无数据可采样的情况，返回空数据或合适的默认值
            return {
                'states': np.array([]),
                'actions': np.array([]),
                'rewards': np.array([]),
                'next_states': np.array([]),
                'dones': np.array([])
            }
