import tensorflow as tf
import tensorflow_probability as tfp
import tf_agents
from tf_agents.networks import network
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from mcts import MCTS  # Custom MCTS implementation
from gym_data.envs import CarcassoneEnv
from tf_agents.environments import FlattenObservationsWrapper
from gym.envs.registration import register
from tf_agents.environments import PyEnvironmentBaseWrapper
from tf_agents.agents import PPOAgent


register(id='Carcassone-v0',entry_point='gym_data.envs:CarcassoneEnv',)

env = suite_gym.wrap_env(FlattenObservationsWrapper(PyEnvironmentBaseWrapper(suite_gym.load("Carcassone-v0"))))
env2 = suite_gym.wrap_env(FlattenObservationsWrapper(PyEnvironmentBaseWrapper(suite_gym.load("Carcassone-v0"))))

# Create a TF-Agents environment
env_name = 'Carcassone-v0'  # Replace with your custom environment
train_env = tf_py_environment.TFPyEnvironment(env)
test_env = tf_py_environment.TFPyEnvironment(env2)

import tensorflow as tf
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.networks import encoding_network
from tf_agents.networks import normal_projection_network
from tf_agents.networks import categorical_projection_network

# Define actor network
class ActorNetwork(network.Network):
    def __init__(self, input_tensor_spec, output_tensor_spec, name="ActorNetwork"):
        super(ActorNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name
        )

        # Separate the observations
        # self._observation_encoders = {}
        # for modality, spec in input_tensor_spec.items():
        #     print(spec)
        #     self._observation_encoders[modality] = encoding_network.EncodingNetwork(
        #         input_tensor_spec=spec,
        #         fc_layer_params=(128, 64),
        #         preprocessing_combiner=None,
        #         activation_fn=tf.nn.relu,
        #         name=f"{modality}_encoder"
        #     )

        self._action_spec = output_tensor_spec
        self._num_actions_per_dimension = output_tensor_spec.maximum - output_tensor_spec.minimum + 1

        self._projection_layers = []
        for num_actions in self._num_actions_per_dimension:
            self._projection_layers.append(
                tf.keras.layers.Dense(num_actions, activation=tf.nn.softmax)
            )



    def call(self, observations, step_type, network_state):
        # encoded_modalities = []
        # for modality, observation in observations.items():
        #     encoding, _ = self._observation_encoders[modality](
        #         observation, step_type=step_type, network_state=network_state)
        #     encoded_modalities.append(encoding)
        #
        # # Concatenate the encoded modalities
        # concatenated_encodings = tf.concat(encoded_modalities, axis=-1)

        action_distribution_list = []

        for i, projection_layer in enumerate(self._projection_layers):
            logits = projection_layer(observations)
            action_distribution = tfp.distributions.Categorical(logits=logits)
            action_distribution_list.append(action_distribution)

        return action_distribution_list

# Define value network
class ValueNetwork(network.Network):
    def __init__(self, input_tensor_spec, name="ValueNetwork"):
        super(ValueNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name
        )

        # Separate the observations
        self._observation_encoders = {}
        for modality, spec in input_tensor_spec.items():
            self._observation_encoders[modality] = encoding_network.EncodingNetwork(
                input_tensor_spec=spec,
                fc_layer_params=(128, 64),
                activation_fn=tf.nn.relu,
                name=f"{modality}_encoder"
            )

        self._output_layer = tf.keras.layers.Dense(
                units=1,
                activation=None,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=2.0, mode="fan_in", distribution="truncated_normal"),
                name="value"
            )

    def call(self, observations, step_type, network_state):
        encoded_modalities = []
        for modality, observation in observations.items():
            encoding, _ = self._observation_encoders[modality](
                observation, step_type=step_type, network_state=network_state)
            encoded_modalities.append(encoding)

        # Concatenate the encoded modalities
        concatenated_encodings = tf.concat(encoded_modalities, axis=-1)
        return self._output_layer(concatenated_encodings, step_type=step_type)

input_tensor_spec = train_env.observation_spec()
output_tensor_spec = train_env.action_spec()
# print()
# print(input_tensor_spec)
# print()

actor_network = ActorNetwork(input_tensor_spec, output_tensor_spec)
value_network = ValueNetwork(input_tensor_spec)

# Define a custom DRL agent (e.g., PPO) using TF-Agents
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = common.create_variable('train_step')
agent = PPOAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    optimizer=optimizer,
    actor_net=actor_network,
    value_net=value_network,
    num_epochs=10,
    train_step_counter=train_step_counter)

# Initialize the MCTS search
mcts = MCTS(train_env, num_simulations=1000)
num_iterations = 1_000
# Training loop
for _ in range(num_iterations):
    # MCTS search
    train_env.reset()
    action = mcts.search(train_env.game)

    # Execute the action in the environment
    train_env.step(action)

    # Update MCTS tree with the result of the action
    mcts.update_tree(action, train_env.game)

    # Collect data for training the DRL agent (PPO)
    trajectories = agent.collect_data(
        train_env,
        policy=random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec()),
        num_episodes=1)

    # Train the DRL agent with the collected data (use your DRL agent's training routine)
    agent.train(trajectories)

# Use the trained DRL agent to play the game

num_episodes = 10  # Adjust the number of episodes as needed
total_reward = 0

for _ in range(num_episodes):
    time_step = test_env.reset()
    episode_reward = 0

    while not time_step.is_last():
        action_step = trained_agent.policy.action(time_step)
        time_step = test_env.step(action_step.action)
        episode_reward += time_step.reward

    total_reward += episode_reward

average_reward = total_reward / num_episodes
print(f"Average reward over {num_episodes} episodes: {average_reward}")

