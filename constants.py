print_QCar_info = False  # print "QCar configured successfully." in qcar.py or not
max_action = 0.5
discount = 0.99
state_dim = 10
action_dim = 2
lr = 4e-4
tau = 0.006
observation_shape = (10,)
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
DEFAULT_MAX_STEPS: int = 800
batch_size = 4096
<<<<<<< HEAD
PREFILL = 100
RUN_ID = ''
# RUN_ID = '4d2e229373de4830a5efc1bc79573e36'
action_v = 0.08
=======
PREFILL = 20000
# RUN_ID = 'ec568c877dfc450a9379359cc314f466'
RUN_ID = ''
action_v = 0.08
cuda = "cuda:1"
>>>>>>> Yida
