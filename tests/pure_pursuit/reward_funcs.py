def reward_based_on_indices_and_noise(
    prev_pos: int, 
    cur_pos: int, 
    noise: list, 
    last_task_len: int
) -> float:
    reward: float = 0.0
    # progress based on indices
    if cur_pos < 0 and prev_pos > 0: # if the car is on the overlapped path
        progress: int = (cur_pos + last_task_len) - prev_pos
    else: # normal case
        progress: int = cur_pos - prev_pos
    # cal reward for the progress
    reward += progress * 0.125
    # add panelty for the noise
    reward += -abs(noise[1]) * progress * 0.104

    return reward