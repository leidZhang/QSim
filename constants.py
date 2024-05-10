print_QCar_info = False  # print "QCar configured successfully." in qcar.py or not
max_action = 1
discount = 0.99
state_dim = 8
action_dim = 2
lr = 3e-4
tau = 0.005
observation_shape = (8,)
buffer_update_rate = 2
policy_freq = 2
# generator constants
METRIC_PREFIX: str = 'agent'
COOL_DOWN_TIME: float = 3.0
# trainer constants
MAX_TRAINING_STEPS = 100_000_000
SAVE_INTERVAL = 30
LOGBATCH_INTERVAL = 1000
LOG_INTERVAL = 10
QCAR_POS = []
# env constants
MAX_LOOKAHEAD_INDICES: int = 200
GOAL_THRESHOLD: int = 2


# settings
DEFAULT_MAX_STEPS: int = 240
batch_size = 512
PREFILL = 512
RUN_ID = '02e6737a267d4d19ad84cea9fe8c63d3'
# RUN_ID = 'ca3395cb35e54b79bfce05109d0f65f4'