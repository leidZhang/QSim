batch_size = 512
print_QCar_info = False  # print "QCar configured successfully." in qcar.py or not
max_action = 1
discount = 0.99
state_dim = 6
action_dim = 2
lr = 3e-4
tau = 0.005
observation_shape = (6,)
buffer_update_rate = 2
policy_noise = 0.02,
noise_clip = 0.05,
policy_freq = 2
run_id = ''

PREFILL = 100
# generator constants
METRIC_PREFIX: str = 'agent'

# trainer constants
MAX_TRAINING_STEPS = 1_000_000
SAVE_INTERVAL = 200
LOGBATCH_INTERVAL = 1000
LOG_INTERVAL = 10
QCAR_POS = []