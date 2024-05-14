print_QCar_info = False  # print "QCar configured successfully." in qcar.py or not
max_action = 0.5
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
MAX_TRAINING_STEPS = 100_000_000000
SAVE_INTERVAL = 30
LOGBATCH_INTERVAL = 400
LOG_INTERVAL = 10
QCAR_POS = []
# env constants
MAX_LOOKAHEAD_INDICES: int = 200
GOAL_THRESHOLD: int = 0.05


# settings
DEFAULT_MAX_STEPS: int = 300
batch_size = 16
PREFILL = 16
# RUN_ID = 'dac197849d4048a4ba97d658c4a9d5d8'
RUN_ID = ''
action_v = 0.08