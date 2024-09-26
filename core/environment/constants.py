from typing import Dict, Set, Any

import numpy as np

DEFAULT_CAR_CONFIG: Dict[str, Any] = {i: None for i in range(1)}

STOP_SIGN_CONFIG: Dict[str, Any] = {
    0: {
        "pose": np.array([2.46, 3.20, -np.pi/2]),
        "random": False,
        "group_id": 3
    },
    1: {
        "pose": np.array([-1.17, 4.66, 0.0]),
        "random": False,
        "group_id": 4
    }
}

TRAFFIC_LIGHT_CONFIG: Dict[str, Any] = {
    0: {
        "ip": "",
        "pose": np.array([-2.20, 0.11, np.pi]),
        "scale": 1.0,
        "random": False,
        "group_id": 0,
        "pair_id": 0,
    },
    1: {
        "ip": "",
        "pose": np.array([2.46, 1.75, 0.0]),
        "scale": 1.0,
        "random": False,
        "group_id": 1,
        "pair_id": 0,
    },
    2: {
        "ip": "",
        "pose": np.array([1.70, 0.11, np.pi]),
        "scale": 1.0,
        "random": False,
        "group_id": 1,
        "pair_id": 0,
    },
    3: {
        "ip": "",
        "pose": np.array([2.46, 0.56, -np.pi/2]),
        "scale": 0.75,
        "random": False,
        "group_id": 1,
        "pair_id": 1,
    },
    4: {
        "ip": "",
        "pose": np.array([0.51, 1.75, 0.0]),
        "scale": 1.0,
        "random": False,
        "group_id": 2,
        "pair_id": 0,
    },
    5: {
        "ip": "",
        "pose": np.array([0.95, 0.56, -np.pi/2]),
        "scale": 1.0,
        "random": False,
        "group_id": 2,
        "pair_id": 1,
    },
    6: {
        "ip": "",
        "pose": np.array([-0.25, 0.11, np.pi]),
        "scale": 1.0,
        "random": False,
        "group_id": 2,
        "pair_id": 0,
    },
    7: {
        "ip": "",
        "pose": np.array([-0.70, 1.32, np.pi/2]),
        "scale": 1.0,
        "random": False,
        "group_id": 2,
        "pair_id": 1,
    }
}

FULL_CONFIG: Dict[str, Any] = {
    'cars': {i: None for i in range(1)},
    'stop_signs': STOP_SIGN_CONFIG,
    'traffic_lights': TRAFFIC_LIGHT_CONFIG
}