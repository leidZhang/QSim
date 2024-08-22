# We skip the tests for some corner cases and focus on the main functionality of the function.
import numpy as np

from reinformer.dataset import CustomDataSet


def build_mock_dataset() -> CustomDataSet:
    mock_path: str = ""
    mock_context_len: int = 10
    mock_device: str = "cpu"

    return CustomDataSet(
        dataset_path=mock_path,
        context_len=mock_context_len,
        device=mock_device
    )


def online_mean_and_std_process(dataset: CustomDataSet, test_observations: np.ndarray) -> None:
    concatenated_observations = np.concatenate(test_observations, axis=0)
    expected_mean: np.ndarray = np.mean(concatenated_observations, axis=0)
    expected_std: np.ndarray = np.std(concatenated_observations, axis=0) + 1e-6

    count, m2 = 0, np.zeros(5)
    dataset.state_mean = np.zeros(5)
    for i in range(len(test_observations)):
        episode_data: dict = test_observations[i]
        dataset.state_mean, m2, count = dataset._online_mean_and_std(episode_data, m2, count)
    variance: np.ndarray = m2 / count
    state_std = np.sqrt(variance) + 1e-6

    assert np.array_equal(expected_mean, dataset.state_mean), \
        f"Expected mean is {expected_mean} but got {dataset.state_mean}"
    assert np.array_equal(expected_std, state_std), \
        f"Expected mean is {expected_std} but got {state_std}"    


def test_online_mean_and_std_1() -> None:
    dataset: CustomDataSet = build_mock_dataset()
    test_observations: np.ndarray = np.array([
        [[1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3]],
        [[1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3]],
    ])
    online_mean_and_std_process(dataset, test_observations)


def test_online_mean_and_std_2() -> None:
    dataset: CustomDataSet = build_mock_dataset()
    test_observations: np.ndarray = np.array([
        [[1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]],
    ])
    online_mean_and_std_process(dataset, test_observations)