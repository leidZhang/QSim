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
LOGBATCH_INTERVAL = 1000
LOG_INTERVAL = 10
QCAR_POS = []
# env constants
MAX_LOOKAHEAD_INDICES: int = 200
GOAL_THRESHOLD: int = 0.05


# settings
DEFAULT_MAX_STEPS: int = 1000
batch_size = 1024
PREFILL = 4000
# RUN_ID = '2a9e92b00a834f5ba87c3447bcaa81f6'
# RUN_ID = 'b513bb8928e24055940f655935299620'
RUN_ID = '7da7f4aa6cb14269b0b515acd942f24d'
action_v = 0.08