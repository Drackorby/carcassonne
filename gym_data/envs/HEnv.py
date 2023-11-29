import numpy as np

import gymnasium as gym

from gymnasium import spaces

import random
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet
from wingedsheep.carcassonne.utils.action_util import ActionUtil
from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame  
from wingedsheep.carcassonne.utils.points_collector import PointsCollector

def find(valid_actions, list):
    found = False
    
    for a in valid_actions:
        if all(a == list):
            found = True
            break;

    return found



class CarcassoneEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array",None], "render_fps": 4}
    possible_actions = []

    def __init__(self, render_mode=None):

        

        n_players = 2

        board_size = 30
        
        self.max_steps = 1000
        self.curr_step = 0


        self.game = CarcassonneGame(  
            players=n_players,  board_size = (board_size, board_size),
            tile_sets=[TileSet.BASE],  
            supplementary_rules=[], visualize = False
        )         
        # self.game.state.get_obs()

        other_properties_space = np.ones(16)*2
        
        other_properties_space[-4] = 250
        other_properties_space[-3] = 250
        other_properties_space[-2] = n_players
        other_properties_space[-1] = 3
        other_properties_space[0] = 15
        other_properties_space[1] = 10
        other_properties_space[5] = 59
        other_properties_space[6] = 8 # Meeples
        other_properties_space[7] = 8

        self.observation_space = spaces.Dict(
        {
            "city_planes": spaces.MultiBinary([15, board_size, board_size]),
            "road_planes": spaces.MultiBinary([10, board_size, board_size]),
            "chapel_plane": spaces.MultiBinary([board_size, board_size]),
            "shield_plane": spaces.MultiBinary([board_size, board_size]),
            "flowers_plane": spaces.MultiBinary([board_size, board_size]),
            "field_planes": spaces.MultiBinary([59, board_size, board_size]),
            "meeple_planes": spaces.MultiBinary([5 * n_players, board_size, board_size]),
            "abbot_planes": spaces.MultiBinary([n_players, board_size, board_size]),
            "farmer_planes": spaces.MultiBinary([9 * n_players, board_size, board_size]),
            "big_farmer_planes": spaces.MultiBinary([9 * n_players, board_size, board_size]),
            "big_meeples_planes": spaces.MultiBinary([5 * n_players, board_size, board_size]),
            "other_properties_plane": spaces.MultiDiscrete(other_properties_space)}
        )
        

        # self.action_space = spaces.MultiDiscrete([3, board_size, board_size, 4, 9, 5])
        self.action_space = spaces.Discrete(3 * board_size * board_size * 4 * 9 * 5)

        for a in range(3):
            for row in range(board_size):
                for col in range(board_size):
                    for pos in range(4):
                        for rot in range(9):
                            for b in range(5):
                                CarcassoneEnv.possible_actions.append([a,row,col,pos,rot,b])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


        self.multi_discrete_action_space = [3, board_size, board_size, 4, 9, 5]

        # Calculate the maximum number of actions
        self.max_num_actions = np.prod(self.multi_discrete_action_space)

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.valid_total_moves = 0
        self.total_tried_moves = 0
        self.past_score = 0
      


    def reset(self, seed=None, options=None):

        self.game.reset()
        self.curr_step = 0
        observation = self.game.get_obs()
        player = self.game.get_current_player()
        while player != 1:
            valid_actions = self.game.get_possible_actions()  
            action = random.choice(valid_actions)
            if action is not None:
                self.game.step(player, action) 
            player = self.game.get_current_player() 
        self.valid_total_moves = 0
        self.total_tried_moves = 0
        self.past_score = 0
        return observation, {}

    def test_step(self,action,player_id):

        action = ActionUtil.create_action(action,self.game.state.next_tile,self.game.state.board,self.game.state,player_id)
        valid_actions = self.game.get_possible_actions() 

        reward = -1000

        if action in valid_actions:
            self.game.step(player_id,action)
        
            scores = PointsCollector.count_score(self.game.state)
            my_score = scores[player_id]
            a.pop(player_id)    
            reward = my_score # - np.max(scores)

        observation = self.game.get_obs()
        terminated = self.game.is_finished()
        info = {}

        return observation, reward, terminated, info
    

    def get_current_player(self):
        return self.game.get_current_player()  
    
    def get_random_action_coded(self):
        actions = self.game.get_possible_actions()
        actions = [ActionUtil.code_action(actions[i], self.game.state.board) for i in range(len(actions))]
        
        return random.choice(actions)

    def action_masks(self):
        return self.get_action_masks()

    def discrete_to_multi(self, discrete_action):
        # Ensure that discrete_action is within a valid range
        discrete_action %= self.max_num_actions 
        
        # Map the discrete action to values for each dimension
        multi_action = np.unravel_index(discrete_action, self.multi_discrete_action_space)
        
        return list(multi_action)

    def multi_to_discrete(self,multi_action):
        # Calculate a unique integer index from the MultiDiscrete actions
        # You can use various methods to do this, like flattening the multi_action into a single array
        flat_action = np.ravel_multi_index(multi_action, self.multi_discrete_action_space)
        
        # Map the flattened action to a Discrete action
        discrete_action = flat_action % self.max_num_actions   # Assuming num_discrete_actions is predefined
        
        return discrete_action
        
    def convert_action_to_indices(self, action, dim_size):
        newSet= { int(element[dim_size]) for element in  action } 
        return list(newSet)
        
    def get_action_masks(self):
        # Determine valid actions based on the current state
        # Your logic for determining valid actions goes here
        actions = self.game.get_possible_actions()
        valid_actions = [self.multi_to_discrete(ActionUtil.code_action(actions[i], self.game.state.board)) for i in range(len(actions))]
    
        # Create an action mask for each dimension of the action space
        action_masks = np.zeros(self.max_num_actions, dtype=bool)
        board_size = len(self.game.state.board)
        # Loop through each dimension of the action space
        for action in valid_actions:
            action_masks[action] = True
        
        return action_masks

 
    def valid_action_mask(self):
        actions = self.game.get_possible_actions()
        actions = [ActionUtil.code_action(actions[i], self.game.state.board) if i < len(actions) else np.zeros(6) for i in range(10)]
        
        return actions

    def step(self,action,player_id = 1):
        action = self.discrete_to_multi(action)
        # valid_actions = self.game.get_possible_actions()
        player_id = self.game.get_current_player()

        if player_id != 1:
            print("You shouldn't be playing...")
        action_obj = ActionUtil.create_action(action, self.game.state.next_tile, self.game.state.board, self.game.state, player_id)
        reward = -1
        valid = False

        if action_obj is not None:
            valid = True
            
        if valid:
            print(action_obj)
            reward = 0
            # print("Valid")
            self.game.step(player_id, action_obj)
            scores = PointsCollector.count_score(self.game.state).tolist()
            # my_score = scores[player_id]
            reward += scores[player_id] - self.past_score
            self.past_score = scores[player_id]
        # else:
        #     print(action)
        #     print(self.valid_action_mask())

        observation = self.game.get_obs()
        terminated = self.game.is_finished() # or len(valid_actions) == 0

        
                
        info = {}

        player = self.game.get_current_player()
        i = 0

        while player != 1 and not terminated:
            if i > 10:
                print("something is wrong")
            valid_actions = self.game.get_possible_actions()  
            action = random.choice(valid_actions)

            if action is not None:  
                self.game.step(player, action)

            player = self.game.get_current_player()
            observation = self.game.get_obs()
            terminated = self.game.is_finished()
            i += 1

        truncated = True if self.curr_step >= self.max_steps else False
        self.curr_step += 1


        if terminated:
            PointsCollector.count_final_scores(self.game.state)
            if self.game.state.get_winner() == 1:
                reward += 100
            else:
                reward -= 100

                
        if valid:
            self.valid_total_moves += 1
        self.total_tried_moves += 1

        if truncated or terminated:
            print("Game finished. % of valid moves: ", self.valid_total_moves/self.total_tried_moves)
        # elif self.curr_step % 50 ==0:
        #     print("Reward: ", reward)
        #     print("Current step: ", self.curr_step)
        return observation, reward, terminated, truncated, info

    def render(self,mode,**kwargs):
        # if self.render_mode == "human":
        self.game.render()

    def close(self):
        pass




# Define the grid world environment
class GridWorldEnvironment(tf_agents.environments.py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.float32, minimum=[0, 0, 0], maximum=[1, 1, 1], name="observation"
        )
        self._state = np.zeros(3, dtype=np.float32)
        self._current_agent = 0  # 0: Parent, 1: Agent A, 2: Agent B

        n_players = 2
        board_size = 30
        
        self.max_steps = 1000
        self.curr_step = 0

        self.game = CarcassonneGame(  
            players=n_players,  board_size = (board_size, board_size),
            tile_sets=[TileSet.BASE],  
            supplementary_rules=[], visualize = False
        )         

        # Complete observation space
        other_properties_space = np.ones(16)*2
        other_properties_space[-4] = 250
        other_properties_space[-3] = 250
        other_properties_space[-2] = n_players
        other_properties_space[-1] = 3
        other_properties_space[0] = 15
        other_properties_space[1] = 10
        other_properties_space[5] = 59
        other_properties_space[6] = 8 # Meeples
        other_properties_space[7] = 8

        self.observation_space = spaces.Dict(
        {
            "city_planes": spaces.MultiBinary([15, board_size, board_size]),
            "road_planes": spaces.MultiBinary([10, board_size, board_size]),
            "chapel_plane": spaces.MultiBinary([board_size, board_size]),
            "shield_plane": spaces.MultiBinary([board_size, board_size]),
            "flowers_plane": spaces.MultiBinary([board_size, board_size]),
            "field_planes": spaces.MultiBinary([59, board_size, board_size]),
            "meeple_planes": spaces.MultiBinary([5 * n_players, board_size, board_size]),
            "abbot_planes": spaces.MultiBinary([n_players, board_size, board_size]),
            "farmer_planes": spaces.MultiBinary([9 * n_players, board_size, board_size]),
            "big_farmer_planes": spaces.MultiBinary([9 * n_players, board_size, board_size]),
            "big_meeples_planes": spaces.MultiBinary([5 * n_players, board_size, board_size]),
            "other_properties_plane": spaces.MultiDiscrete(other_properties_space)}
        )
        
        # Tile agent observation space
        other_properties_space = np.ones(6)*2
        other_properties_space[0] = 15
        other_properties_space[1] = 10
        other_properties_space[5] = 59
        
        self.tite_observation_space = spaces.Dict({
            "city_planes": spaces.MultiBinary([15, board_size, board_size]),
            "road_planes": spaces.MultiBinary([10, board_size, board_size]),
            "chapel_plane": spaces.MultiBinary([board_size, board_size]),
            "shield_plane": spaces.MultiBinary([board_size, board_size]),
            "flowers_plane": spaces.MultiBinary([board_size, board_size]),
            "field_planes": spaces.MultiBinary([59, board_size, board_size]),
            "other_properties_plane": spaces.MultiDiscrete(other_properties_space)}
        )
        
        
        # Meeple agent observation space

        other_properties_space = np.ones(4)*2
        other_properties_space[0] = 8 # Meeples
        other_properties_space[1] = 8

        self.meeple_observation_space = spaces.Dict(
        {
            "city_planes": spaces.MultiBinary([15, board_size, board_size]),
            "road_planes": spaces.MultiBinary([10, board_size, board_size]),
            "chapel_plane": spaces.MultiBinary([board_size, board_size]),
            "shield_plane": spaces.MultiBinary([board_size, board_size]),
            "flowers_plane": spaces.MultiBinary([board_size, board_size]),
            "field_planes": spaces.MultiBinary([59, board_size, board_size]),
            "meeple_planes": spaces.MultiBinary([5 * n_players, board_size, board_size]),
            "abbot_planes": spaces.MultiBinary([n_players, board_size, board_size]),
            "farmer_planes": spaces.MultiBinary([9 * n_players, board_size, board_size]),
            "big_farmer_planes": spaces.MultiBinary([9 * n_players, board_size, board_size]),
            "big_meeples_planes": spaces.MultiBinary([5 * n_players, board_size, board_size]),
            "other_properties_plane": spaces.MultiDiscrete(other_properties_space)}
        )

        self.action_space = spaces.Discrete(3 * board_size * board_size * 4 * 9 * 5)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


        self.multi_discrete_action_space = [3, board_size, board_size, 4, 9, 5]

        # Calculate the maximum number of actions
        self.max_num_actions = np.prod(self.multi_discrete_action_space)

        self.window = None
        self.clock = None

        self.valid_total_moves = 0
        self.total_tried_moves = 0
        self.past_score = 0
        
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._current_agent = 1  # Start with Agent A
        self._state = np.zeros(3, dtype=np.float32)
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):
        if self._current_agent == 1:
            self._state[0] += 0.1  # Move Agent A
        elif self._current_agent == 2:
            self._state[1] += 0.1  # Move Agent B

        if action == 1:
            self._current_agent = 1  # Switch to Agent A
        else:
            self._current_agent = 2  # Switch to Agent B

        if self._state[0] >= 1.0 and self._state[1] >= 1.0:
            return ts.termination(np.array(self._state, dtype=np.float32), reward=1.0)
        else:
            return ts.transition(np.array(self._state, dtype=np.float32), reward=0.0, discount=1.0)

# Create the environment
grid_world_env = GridWorldEnvironment()

# Wrap the environment in a TF-Agents environment
tf_env = tf_py_environment.TFPyEnvironment(grid_world_env)

# Define a simple Q-Network
q_net = tf.keras.layers.Dense(2, activation=None)

# Define the sub-agents' policies
agent_a = tfa.agents.dqn.dqn_agent.DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
)
agent_b = tfa.agents.dqn.dqn_agent.DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
)

# Define the parent agent's policy
parent_agent = tfa.agents.dqn.dqn_agent.DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
)

# Define the overall policy that switches between sub-agents
class HierarchicalPolicy(tf.Module):
    def __init__(self, sub_agents, parent_agent):
        self.sub_agents = sub_agents
        self.parent_agent = parent_agent

    def select_sub_agent(self, observation):
        # Implement logic to select the active sub-agent based on the observation
        # For simplicity, switch between sub-agents every step
        return self.sub_agents[self.parent_agent.active_sub_agent]

# Create the hierarchical policy
hierarchical_policy = HierarchicalPolicy([agent_a, agent_b], parent_agent)

# Define a function to evaluate the hierarchical policy
def evaluate_hierarchical_policy(policy, num_steps=100):
    total_reward = 0.0
    for _ in range(num_steps):
        time_step = tf_env.reset()
        episode_reward = 0.0
        while not time_step.is_last():
            action = policy.select_sub_agent(time_step.observation)
            time_step = tf_env.step(action)
            episode_reward += time_step.reward
        total_reward += episode_reward
    return total_reward / num_steps

# Evaluate the hierarchical policy
avg_reward = evaluate_hierarchical_policy(hierarchical_policy)
print("Average Reward:", avg_reward)