from Environment import Environment
from Parameter import Parameter
from scipy.io import savemat
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import os
import time
import argparse
import sys
import time
import pickle
import pdb
import time
import psutil
from memory_profiler import profile
from memory_profiler import memory_usage
# +
#################### params ###########################
parser = argparse.ArgumentParser(description='Hyper_params')
parser.add_argument('--Info', default='', type=str)  # information added to log dir name

parser.add_argument('--Seed', default=41, type=int)
parser.add_argument('--Units', default=128, type=int)  # hidden units num of critic NN
parser.add_argument('--Lr', default=0.001, type=float)  # learning rate
parser.add_argument('--omega', default=0.005, type=float)  # used to update target networks
parser.add_argument('--Max_Epsilon', default=0.0, type=float)
parser.add_argument('--Min_Epsilon', default=0.0, type=float)
parser.add_argument('--Epsilon_Decay', default=1.0, type=float)
parser.add_argument('--Batch_Size', default=256, type=int)
parser.add_argument('--Memory_Size', default=200000, type=int) # buffer size
parser.add_argument('--Start_Size', default=0, type=int)  # random action before start_size
parser.add_argument('--Update_After', default=0, type=int)
parser.add_argument('--Train_Interval', default=1, type=int)
parser.add_argument('--load_weights', default=False, type=bool)
parser.add_argument('--Alg', default='dpds', type=str)
parser.add_argument('--Gpu_Id', default="0", type=str) # -1 means CPU
parser.add_argument('--E_max', default=0.1, type=float)
# parser.add_argument('--N', default=15, type=int)  # number of WDs
parser.add_argument('--N', default=52, type=int)  # number of WDs
parser.add_argument('--T', default=100000, type=int)  # number of simulated slots
# parser.add_argument('--T', default=4000, type=int)  # number of simulated slots -- amin test
parser.add_argument('--batch_norm', default=True, type=bool)

args = parser.parse_args()
# -

#################### seed ###########################
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = args.Gpu_Id
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(tf.config.list_physical_devices())
tf.random.set_seed(args.Seed)
np.random.seed(args.Seed)
random.seed(args.Seed)

param = Parameter(args.N, args.T)
param.E_max = [args.E_max*param.d_t for _ in range(param.N)]
if args.Alg == 'lpo':
    # for local processing only, we set the wireless bandwidth to zero
    param.W_max = 0.0
    param.E_max = [3*args.E_max*param.d_t for _ in range(param.N)]
    #param.lam_init = np.ones(param.N) * 1000
    #param.theta = lambda t: 0
if args.Alg == 'coo':
    # for computation offloading only, we set the local CPU frequency to zero
    param.f_max = [0.0 for _ in range(param.N)]
    param.E_max = [2*args.E_max*param.d_t for _ in range(param.N)]
env = Environment(param)
if args.load_weights:
    with open('models/v.pickle', 'rb') as f:
        Initial_v = pickle.load(f)
else:
    Initial_v = 0  # initial average reward

#################### log ###########################
# create log file
time_str = time.strftime("%m-%d_%H-%M", time.localtime())
alg = args.Alg
log_dir_name = 'logs/' + time_str + '_' + alg + args.Info + '_n' + \
               str(args.N) + '_seed' + str(args.Seed) + '_lam' + str(int(param.lam_init[0]))
data_dir_name = 'data/' + alg + '_' + args.Info + '_n' + \
               str(args.N) + '_lam' + str(int(param.lam_init[0]))
fw = tf.summary.create_file_writer(log_dir_name)  # log file witer

# create dir to save model
if not os.path.exists(log_dir_name + '/models'):
    os.makedirs(log_dir_name + '/models')

# save params to a .txt file
prams_file = open(log_dir_name + '/prams_table.txt', 'w')
prams_file.writelines(f'{i:50} {v}\n' for i, v in args.__dict__.items())
prams_file.close()

###################### others ###############################
W_Initializer = tf.initializers.he_normal(args.Seed)  # NN initializer
Epsilon_Decay_Rate = (args.Min_Epsilon - args.Max_Epsilon) / (args.T) * args.Epsilon_Decay # factor of decay
TENSOR_FLOAT_TYPE = tf.dtypes.float32
TENSOR_INT_TYPE = tf.dtypes.int32

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# +
class ReplayBuffer:
    def __init__(self, buffer_capacity = 100000):
        self.buffer_capacity = buffer_capacity
        self.buffer_counter = 0

        # dim(action) = N * 3
        # dim(state) = N * 4
        # dim(pds) = N * 4
        buffer_a_dim = (buffer_capacity, param.N, 3)
        buffer_s_dim = (buffer_capacity, param.N, 4)

        self.s_buffer = np.empty(buffer_s_dim, dtype=np.float32)
        self.a_buffer = np.empty(buffer_a_dim, dtype=np.float32)
        self.r_buffer = np.empty((buffer_capacity,), dtype=np.float32)
        self.next_s_buffer = np.empty(buffer_s_dim, dtype=np.float32)

    def store(self, exp):
        index = self.buffer_counter % self.buffer_capacity

        s, a, r, next_s = exp
        self.s_buffer[index] = s
        self.a_buffer[index] = a
        self.r_buffer[index] = r
        self.next_s_buffer[index] = next_s

        self.buffer_counter += 1

    def sample(self, batch_size):
        sampling_range = min(self.buffer_counter, self.buffer_capacity)
        idx = np.random.randint(0, sampling_range, batch_size)

        batch_s = tf.convert_to_tensor(self.s_buffer[idx])
        batch_a = tf.convert_to_tensor(self.a_buffer[idx])
        batch_r = tf.convert_to_tensor(self.r_buffer[idx])
        batch_next_s = tf.convert_to_tensor(self.next_s_buffer[idx])

        return batch_s, batch_a, batch_r, batch_next_s

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
# -


### added by amin
class ConvergenceDetector:
    def __init__(self, window_size=1000, threshold=1e-2):
        self.window_size = window_size
        self.threshold = threshold
        self.values = []
        self.converged = False
        self.std_on_mean = 0.0

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

        if len(self.values) == self.window_size:
            mean = np.mean(self.values)
            std = np.std(self.values)
            self.std_on_mean = abs(std / mean)
            if abs(std / mean) < self.threshold:
                print(f"std / mean: {abs(std / mean)} --- th {self.threshold}")
                self.converged = True

    def is_converged(self):
        return self.converged


class DPDS:
    def __init__(self, batch_size, memory_size, max_epsilon):

        def build_actor():
            print("hi")
            inputs = keras.Input(shape=(param.N, 4))
            x = keras.layers.Flatten()(inputs)
            x = keras.layers.Dense(args.Units, activation='relu', kernel_initializer=W_Initializer)(x)
            if args.batch_norm:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dense(args.Units, activation='relu', kernel_initializer=W_Initializer)(x)
            if args.batch_norm:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dense(param.N*3, activation='sigmoid', kernel_initializer=W_Initializer)(x)
            outputs = keras.layers.Reshape((param.N, 3))(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            return model

        def build_value():
            inputs = keras.Input(shape=(param.N, 4))
            x = keras.layers.Flatten()(inputs)
            x = keras.layers.Dense(args.Units, activation='relu', kernel_initializer=W_Initializer)(x)
            if args.batch_norm:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dense(args.Units, activation='relu', kernel_initializer=W_Initializer)(x)
            if args.batch_norm:
                x = keras.layers.BatchNormalization()(x)
            outputs = keras.layers.Dense(1)(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            return model

        self.actor = build_actor()
        self.value = build_value()
        self.target_actor = build_actor()
        self.target_value = build_value()
        if args.load_weights:
            self.value.load_weights("models/value")
            self.actor.load_weights("models/actor")
            print("load weight")
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_value.set_weights(self.value.get_weights())

        self.actor_optimizer = tf.optimizers.Adam(args.Lr)
        self.value_optimizer = tf.optimizers.Adam(args.Lr*2)
        self.epsilon = max_epsilon
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(memory_size)
        self.alg = alg
        self.v = Initial_v  # average reward
        self.target_v = Initial_v
        self.lam = param.lam_init

        # transform some parameters to tensors
        self.f_max = tf.convert_to_tensor(param.f_max)
        self.P_max = tf.convert_to_tensor(param.P_max)
        self.W_max = tf.convert_to_tensor(param.W_max)
        self.E_max = tf.convert_to_tensor(param.E_max)
        self.d_t = tf.convert_to_tensor(param.d_t)
        self.kappa = tf.convert_to_tensor(param.kappa)
        self.sigma2 = tf.convert_to_tensor(param.sigma2)
        self.gamma = tf.convert_to_tensor(param.gamma)

    def random_action(self, s):
        action = np.random.rand(param.N, 3)
        # apply softmax to W so that its sum equals 1
        action[:,2] = softmax(action[:,2])
        return action
    
    def _choose_action(self, s):
        action = self.actor(s[None, :])[0].numpy()
        action[:,2] = softmax(action[:,2])
        return action

    def choose_action(self, s, noise_object, epsilon):
        action = self.actor(s[None, :])[0].numpy()
        noise = noise_object()
        # print(f"noise: {noise}")
        # Adding noise to action
        action = action + epsilon * noise

        # We make sure action is within bounds
        legal_action = np.clip(action, 0, 1)
        legal_action[:,2] = softmax(legal_action[:,2])
        # print(f"legal_action: {legal_action}")
        return legal_action

    def choose_alternative_action(self, s, std_mean, epsilon):
        # Generate base action
        action = self.actor(s[None, :])[0].numpy()

        # Add noise with increasing magnitude
        # action = action * (np.random.rand())
        action = action + epsilon * (random.uniform(epsilon, std_mean))
        # We make sure action is within bounds
        legal_action = np.clip(action, 0, 1)
        legal_action[:, 2] = softmax(legal_action[:, 2])
        # print(f"try action: {legal_action}")
        # sys.exit(0)
        return legal_action


    @tf.function(jit_compile=True)
    def f_k(self, batch_state, batch_action):
        f = batch_action[:, :, 0] * self.f_max
        P = batch_action[:, :, 1] * self.P_max
        W = tf.nn.softmax(batch_action[:, :, 2]) * self.W_max
        d_r = batch_state[:, :, 0]
        a = batch_state[:, :, 1]
        q = batch_state[:, :, 2]
        h = batch_state[:, :, 3]

        d = f * self.d_t / self.kappa + \
                self.d_t * W * (tf.math.log(1+P*h/self.sigma2)/tf.math.log(2.))
        b = ((d > 0) & (d_r > 0) & (d >= d_r))
        b = tf.where(b, 1.0, 0.0)
        pds_q = q - b
        pds_h = h
        pds_d_r = tf.maximum(tf.constant(0, dtype=tf.float32), d_r - d)
        # explanation of the following update formula can be found here: https://github.com/XingqiuHe/DPDS/issues/2
        pds_a = a - 3.3 * b 
        
        s = tf.stack([pds_d_r, pds_a, pds_q, pds_h], axis=2)
        return s

    @tf.function(jit_compile=True)
    def cost(self, batch_state, batch_action):
        f = batch_action[:, :, 0] * self.f_max
        P = batch_action[:, :, 1] * self.P_max
        W = tf.nn.softmax(batch_action[:, :, 2]) * self.W_max
        E = self.gamma * f**3 * self.d_t + P * self.d_t
        h = batch_state[:, :, 3]
        d = f * self.d_t / self.kappa + \
                self.d_t * W * (tf.math.log(1+P*h/self.sigma2)/tf.math.log(2.))
        # in expectation, completing a task reduces aoi by 1/p_g
        aoi_per_bit = 1 / param.p_g / ((param.d_lb+param.d_ub)/2)
        if args.Alg == 'dpl':
            # for delay-based algorithm, the total delay equals the sum of length of queues
            cost = tf.reduce_sum(batch_state[:, :, 2] - d/((param.d_lb+param.d_ub)/2) + self.lam * tf.math.maximum(0.0, E - self.E_max), axis=1)
        else:
            cost = tf.reduce_sum(batch_state[:, :, 1] - d*aoi_per_bit + self.lam * tf.math.maximum(0.0, E - self.E_max), axis=1)
            #cost = tf.reduce_sum(batch_state[:, :, 1] + self.lam * tf.math.maximum(0.0, E - self.E_max), axis=1)
            #cost = tf.reduce_sum(batch_state[:, :, 1] + self.lam * (E - self.E_max), axis=1)
        return cost


    @tf.function(jit_compile=True)
    def train(self, s, a, r, s_next):
        # update value network
        with tf.GradientTape() as tape:
            # calculate target y
            target_a_next = self.target_actor(s_next, training=True)
            target_pds_next = self.f_k(s_next, target_a_next)
            target_y = self.cost(s_next, target_a_next) + \
                self.target_value(target_pds_next, training=True) - self.target_v
            pds = self.f_k(s,a) ### To improve learning efficiency, paper introduce Post-Decision States (PDSs) to split the known & unknown system dynamics so that we only need to learn the unknown part.
            pds_value = self.value(pds, training=True)
            td = pds_value - target_y
            value_loss = tf.math.reduce_mean(tf.math.abs(td))

        value_grad = tape.gradient(value_loss, self.value.trainable_variables)
        #value_grad = [tf.clip_by_norm(grad, 10.0) for grad in value_grad]
        self.value_optimizer.apply_gradients( zip(value_grad, self.value.trainable_variables) )

        # update actor network
        with tf.GradientTape() as tape:
            actions = self.actor(s, training=True)
            pds = self.f_k(s, actions) ### To improve learning efficiency, paper introduce Post-Decision States (PDSs) to split the known & unknown system dynamics so that we only need to learn the unknown part.
            #critic_value = cost(s, actions) + self.value(pds, training=True) - self.v
            c = self.cost(s, actions)
            value = self.value(pds, training=True)
            critic_value = c + value
            actor_loss = tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        #actor_grad = [tf.clip_by_norm(grad, 10.0) for grad in actor_grad]
        self.actor_optimizer.apply_gradients( zip(actor_grad, self.actor.trainable_variables) )
        # td is returned to update self.v
        # we do not update self.v in this function because it leaks the local tensor 'td', which is prohibited by tensorflow
        return (td, value_loss, actor_loss, c, value)

    def save_model(self, dir=log_dir_name + '/models'):
        self.actor.save_weights(dir + '/' + self.alg + '_actor')
        self.value.save_weights(dir + '/' + self.alg + '_value')
        self.actor.save_weights('models/actor')
        self.value.save_weights('models/value')
        with open(dir + '/' + self.alg + '_v.pickle', 'wb') as f:
            pickle.dump(self.v, f)
        with open('models/v.pickle', 'wb') as f:
            pickle.dump(self.v, f)

    
@tf.function(jit_compile=True)
def update_target(target_weights, weights, omega):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * omega + a * (1 - omega))


def train(T):  #### T: number of simulated slots
    # Start the timer
    start_time_dpds = time.time()
    agent = DPDS(args.Batch_Size, args.Memory_Size, args.Max_Epsilon)
    print("============" + agent.alg + "============")

    state = env.reset()
    std_dev = 0.01
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
    acc_E = np.zeros(param.N)
    acc_A = np.zeros(param.N)
    stat_W = []
    timer = 1
    
    acc_interaction_time = 0
    acc_inference_time = 0
    acc_training_time = 0

    # Initialize convergence detectors ## amin
    cost_detector = ConvergenceDetector()  ## amin
    value_loss_detector = ConvergenceDetector() ## amin
    actor_loss_detector = ConvergenceDetector() ## amin

    ### added by amin for memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    max_memory = initial_memory
    total_memory_used = 0
    memory_usage_log = []

    old_E = np.zeros(52)
    old_h = 0
    step_after_prune = 0
    cost_list = []
    old_cost = 0.0
    count_cost = 0
    np_sum_E_old = 1.1

    start_time = time.time()
    start_cpu_time = time.process_time()
    ## T: number of simulated slots
    while timer <= T:
        # if timer % 10000 == 0:  ## was
        if timer % 100 == 0:  ## is
            print(timer)
            # print(f"buffer_size: {agent.buffer.buffer_counter}")  ## about same to value of timer - 1
            # print(f"step_after_prune: {step_after_prune}")
            # print(f"count_cost: {count_cost}")
            step_after_prune = 0
            # print(
            #     f"CONV_cost_detector: {(np.std(cost_detector.values) / np.mean(cost_detector.values))} ...{cost_detector.threshold}")
            # print(
            #     f"CONV_value_loss_detector: {(np.std(value_loss_detector.values) / np.mean(value_loss_detector.values))} ...{value_loss_detector.threshold}")
            print(f"CONV_actor_loss_detector: {(abs(np.std(actor_loss_detector.values) / np.mean(actor_loss_detector.values)))} ||| {actor_loss_detector.threshold}")
            # print("Average number of tasks:", env.nTask / timer / args.N)  ## 0.28 - 0.30
            # print(f"aoi: {(np.sum(state[:, 1]) / args.N)}")
            # print(f"average aoi: {np.sum(acc_A) / timer / args.N}")
            # print(f"agent.target_v: {agent.target_v}")
            # print(f"agent.v: {agent.v}")
            # print(f"value_loss: {value_loss}")
            # print(f"actor_loss: {actor_loss}")
            # print(f'cost1: {tf.math.reduce_mean(c)}')
            # print(f'value: {tf.math.reduce_mean(v)}')
            print()

        if timer <= args.Start_Size:
            action = agent.random_action(state)
        else:
            inference_begin = time.time()
            action = agent.choose_action(state, ou_noise, agent.epsilon)
            inference_end = time.time()
            acc_inference_time += inference_end - inference_begin
        interaction_begin = time.time()
        next_state, E = env.step(action)
        # print(f"E: {E}")
        # print(f"next_state: {next_state}")
        # print(f"abs((np.sum(E)/52)-(np.sum(old_E)/52)): {abs((np.sum(E)/52)-(np.sum(old_E)/52))}")
        ################
        zero_first_element_count = np.sum(next_state[:, 0] == 0)
        # print(f"number of completed tasks: {zero_first_element_count}")
        ### if (abs((np.sum(E) / 52) - (np.sum(old_E) / 52)) > 1e-3):
        #############################################################

        while True:
            ## when d_r (first item in 'next_state') is '0' means that the task is completed while we want step that be informative rather
            # print(f"(np.sum(E)/52)_UP: {(np.sum(E)/52)}")
            print(f"current cost {old_cost} + std {np.std(cost_list)} = {(old_cost + np.mean(cost_list))}")
            print(f"next cost: {(np.sum(next_state[:,1] + agent.lam * (E - param.E_max)))}")
            if (np.sum(next_state[:,1] + agent.lam * (E - param.E_max))) < (old_cost + np.mean(cost_list)) or (timer < 2):
                # print("pass")
            # if (zero_first_element_count > 26):
            # if ((np_sum_E_old - (np.sum(E)/52)) > 0) or (zero_first_element_count < 20):
                # print(f"np_sum_E_old: {np_sum_E_old}")
                # np_sum_E_old = np.sum(E)/52
                # print(f"new E: {np.sum(E)/52}")
                # (abs((np.sum(E) / 52) - (np.sum(old_E) / 52)) > 1e-3):
                # or (abs((old_h) - (np.sum(next_state[:, 3]))) > 1e-4)
                # old_E = E
                # old_h = np.sum(next_state[:, 3])
                # step_after_prune += 1
                break  # Exit the loop when conditions are met
            else:
                # action = agent.random_action(state)
                ###
                # inference_begin = time.time()
                action = agent.choose_alternative_action(state, actor_loss_detector.std_on_mean, agent.epsilon)
                print(f"change")
                # inference_end = time.time()
                # acc_inference_time += inference_end - inference_begin
                ###
                interaction_begin = time.time()
                next_state, E = env.step(action)
                # print(f"(np.sum(E)/52)_DOWN: {(np.sum(E) / 52)}")
                zero_first_element_count = np.sum(next_state[:, 0] == 0)
                # print(f"next_state_again is asked")
                # print()
        ################################################################
        # if (abs((old_h) - (np.sum(next_state[:, 3]))) > 1e-4):
        #     print(f"h: {np.sum(next_state[:, 3])}")

        # print(f"len(E): {len(E)}") ### 52  -- number of users
        # print(f"len(next_state): {len(next_state)}") ### 52 -- number of users
        # print(f"next_state: {next_state}")
        # zero_first_element_count = np.sum(next_state[:, 0] == 0)
        # if (zero_first_element_count >= 26):
        #     print(f"Number of items with first element equal to zero: {zero_first_element_count}")
        # print()
        interaction_end = time.time()
        acc_interaction_time += interaction_end - interaction_begin
        cost = np.sum(state[:,1] + agent.lam * (E - param.E_max))
        old_cost = cost
        # print(f"state[:,1]: {state[:,1]}")
        # print(f"agent.lam: {agent.lam}")
        # print(f"(E - param.E_max): {(E - param.E_max)}")

        cost_list.append(cost)
        # print(f"cost: {cost}")
        # print(f"avg of cost_list: {np.mean(cost_list)}")
        ###########################################################################
        ## we can say the cost that is worth it to move as agent...maybe 10% of std
        # if ((abs(old_cost - cost)) > ((np.std(cost_list) * 25) / 100)):
        #     old_cost = cost
        #     print(f"((np.std(cost_list) * 25) / 100): {((np.std(cost_list) * 25) / 100)}")
        #     print(f"cost: {cost}")
        #     count_cost += 1
        #     agent.buffer.store((state, action, cost, next_state))  ## is
        #     print()
        ###########################################################################
        agent.buffer.store((state, action, cost, next_state))  ## was


        # print("before train")

        ###### train ###############
        if timer > args.Update_After and timer % args.Train_Interval == 0:
            ### memory -- added amin
            pre_train_memory = process.memory_info().rss

            training_begin = time.time()
            # sample from buffer
            s, a, r, s_next = agent.buffer.sample(args.Batch_Size)
            # print(f"s_next: {s_next}")
            # are_equal = np.array_equal(s_next, next_state)  ## amin
            # print(f"The arrays are {'equal' if are_equal else 'not equal'}") ## amin
            td, value_loss, actor_loss, c, v = agent.train(s, a, r, s_next)
            agent.v = agent.v - param.beta(timer) * tf.reduce_mean(td)
            training_end = time.time()
            acc_training_time += training_end - training_begin

            # Measure memory usage after training step -- amin
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            max_memory = max(max_memory, current_memory)
            total_memory_used += memory_increase
            memory_usage_log.append(memory_increase)
            # Measure memory usage after training step -- amin

            # update target networks
            update_target(agent.target_actor.variables, agent.actor.variables, args.omega)
            update_target(agent.target_value.variables, agent.value.variables, args.omega)
            agent.target_v = args.omega * agent.v + (1 - args.omega) * agent.target_v

            # update lambda
            agent.lam = agent.lam + param.theta(timer) * (E - param.E_max)
            agent.lam = np.maximum(0, agent.lam)
            agent.lam = np.minimum(param.lam_max, agent.lam)

            # Check for convergence ## amin
            cost_detector.update(cost)   ## amin
            value_loss_detector.update(value_loss)   ## amin
            actor_loss_detector.update(actor_loss)   ## amin
            ## amin
            # if cost_detector.is_converged() and value_loss_detector.is_converged() and actor_loss_detector.is_converged():   ## amin
            if actor_loss_detector.is_converged():  ## amin
                print(f"DPDS has converged at iteration {timer}")   ## amin
                agent.save_model()
                print("Average interaction time:", acc_interaction_time / args.T)
                print("Average inference time:", acc_inference_time / (args.T - args.Start_Size))
                print("Average training time:", acc_training_time / (args.T - args.Update_After))
                print("Average number of tasks:", env.nTask / timer / args.N)
                data = {'N': param.N, 'locations': param.WD_loc_list, 'distances': param.distance, 'E': env.E_stat,
                        'A': env.A_stat}
                savemat(data_dir_name + '.mat', data)
                # End the timer and calculate the duration
                end_time_dpds = time.time()
                duration_dpds = end_time_dpds - start_time_dpds
                print(f"Execution time: {duration_dpds:.2f} seconds")

                ### memory print -- amin
                print(f"Total memory usage of training: {total_memory_used / (1024 * 1024):.2f} MB")
                print(f"Average memory usage per training step: {(sum(memory_usage_log) / len(memory_usage_log)) / (1024 * 1024):.2f} MB")
                ### memory print -- amin

                sys.exit(0)  # Terminate the program immediately

            with fw.as_default():
                tf.summary.scalar('value_loss', value_loss, step = timer)
                tf.summary.scalar('actor_loss', actor_loss, step = timer)
                tf.summary.scalar('cost1', tf.math.reduce_mean(c), step = timer)
                tf.summary.scalar('value', tf.math.reduce_mean(v), step = timer)
            
        # epsilon decay
        agent.epsilon = max(Epsilon_Decay_Rate * timer + args.Max_Epsilon, args.Min_Epsilon)

        timer += 1

        """
        TODO
            we need to explain that in the paper that we can guarantee constraint (10b) by 
            applying softmax on the W
            also need to specify we use values between (0,1) to represent the control variables
            also need to discuss we do not need to ensure constraint (7)
        """



        # log
        acc_E += E
        acc_A += state[:,1]
        stat_W.append(action[:,2])
        with fw.as_default():
            tf.summary.scalar('cost', cost, step=timer)
            tf.summary.scalar('aoi', np.sum(state[:,1])/args.N, step=timer)
            tf.summary.scalar('average aoi', np.sum(acc_A)/timer/args.N, step=timer)
            tf.summary.scalar('energy', np.sum(E)/args.N, step=timer)
            tf.summary.scalar('v', agent.v, step=timer)
            tf.summary.scalar('epsilon', agent.epsilon, step=timer)
            tf.summary.scalar('lambda', np.sum(agent.lam)/args.N, step=timer)
            tf.summary.scalar('average energy', np.sum(acc_E)/timer/args.N, step=timer)

        state = next_state

    end_time = time.time()
    end_cpu_time = time.process_time()

    #### ### memory -- amin
    #### final_memory = process.memory_info().rss

    agent.save_model()
    print("Average interaction time:", acc_interaction_time / args.T)
    print("Average inference time:", acc_inference_time / (args.T - args.Start_Size))
    print("Average training time:", acc_training_time / (args.T - args.Update_After))
    print("Average number of tasks:", env.nTask / timer /args.N)

    ### memory print -- amin
    print(f"Total memory usage of training: {total_memory_used / (1024 * 1024):.2f} MB")
    print(f"Average memory usage per training step: {(sum(memory_usage_log) / len(memory_usage_log)) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    train(args.T)

print("DPDS is done")


