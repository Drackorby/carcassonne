import numpy as np

import gymnasium as gym

from gymnasium import spaces
import time
import random

from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState
from wingedsheep.carcassonne.carcassonne_visualiser import CarcassonneVisualiser
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet
from wingedsheep.carcassonne.utils.action_util import ActionUtil
from wingedsheep.carcassonne.utils.state_updater import StateUpdater
from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame
from wingedsheep.carcassonne.utils.points_collector import PointsCollector
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction
from wingedsheep.carcassonne.objects.actions.meeple_action import MeepleAction
from wingedsheep.carcassonne.objects.game_phase import GamePhase

import requests
import websockets
import json
import asyncio


class MyWebSocketClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.ws = None

    async def connect(self):
        self.ws = await websockets.connect(self.server_url)
        print("WebSocket connected")

    async def send(self, data):
        if self.ws:
            await self.ws.send(json.dumps(data))
            response = await self.ws.recv()
            response = json.loads(response)
            return response["action"]
        else:
            return False

    async def close(self):
        if self.ws:
            await self.ws.close()


def numpy_array_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [numpy_array_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_array_to_list(value) for key, value in obj.items()}
    else:
        return obj


class CarcassoneEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 4}
    possible_actions = []

    def __init__(self, render_mode=None):

        self.verbose = True
        n_players = 2

        board_size = 10

        self.max_steps = 101
        self.curr_step = 0

        self.game = CarcassonneGame(
            players=n_players, board_size=(board_size, board_size),
            tile_sets=[TileSet.BASE],
            supplementary_rules=[], visualize=False
        )

        other_properties_space = np.ones(12) * 2

        other_properties_space[-4] = 250
        other_properties_space[-3] = 250
        other_properties_space[-2] = n_players
        other_properties_space[-1] = 3
        other_properties_space[0] = 15
        other_properties_space[1] = 10
        other_properties_space[5] = 66
        other_properties_space[6] = 8
        other_properties_space[7] = 8

        self.observation_space = spaces.Dict(
            {
                "tile_planes": spaces.MultiBinary([94, board_size, board_size]),
                "chars_planes": spaces.MultiBinary([42, board_size, board_size]),
                "other_properties_plane": spaces.MultiDiscrete(other_properties_space)}
        )

        self.action_space = spaces.MultiDiscrete([3, board_size, board_size, 4, 9, 5])
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.multi_discrete_action_space = [3, board_size, board_size, 4, 9, 5]

        # Calculate the maximum number of actions
        self.max_num_actions = np.prod(self.multi_discrete_action_space)
        self.action_mask = np.zeros((self.max_num_actions, 6))

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
        self.progress = 1
        self.self_play = False
        self.ws = None
        self.id = -1

    def connect_ws(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                for key1, value1 in value.items():
                    if key1 == "i":
                        self.id = value1
        self.ws = MyWebSocketClient('ws://127.0.0.1:8000/action/' + str(self.id) + "/")

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.ws.connect())

    def generate_trajectory(self, trajectory_length):
        """
        Generate a trajectory of a specified length.

        Args:
            trajectory_length (int): The length of the trajectory to generate.

        Returns:
            list of tuples: A list of (state, action, reward, next_state, done) tuples.
        """
        prev_verbose = self.verbose
        self.verbose = False

        trajectory = []

        for _ in range(trajectory_length):
            valid_actions = self.game.get_possible_actions()
            action = ActionUtil.code_action(random.choice(valid_actions))

            state = self.game.get_obs()
            next_state, reward, terminated, truncated, info = self.step(action)
            done = terminated
            operation = {
                'state': state,
                'action': action,
                'done': done,
            }
            trajectory.append(operation)

            if done:
                self.reset()

        self.verbose = prev_verbose

        return trajectory

    def reset(self, seed=None, options=None):

        rand_seed = self.set_random()
        self.game.reset(rand_seed)
        self.curr_step = 0

        if self.self_play and self.ws is None and self.id != -1:
            self.ws = MyWebSocketClient('ws://127.0.0.1:8000/action/' + str(self.id) + "/")

            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.ws.connect())

        observation, terminated = self.other_player_move()

        self.valid_total_moves = 0
        self.total_tried_moves = 0

        self.past_score = 0

        return observation, {}

    def get_current_player(self):
        return self.game.get_current_player()

    def is_action_valid(self, phase, valid_actions, action ):
        valid = False
        i = 0

        if phase == GamePhase.TILES:
            action_to_validate = np.array(action[0:4])
        else:
            action_to_validate = np.concatenate((action[0:3], action[4:]))

        while i < len(valid_actions):
            coded_valid_action = ActionUtil.code_action(valid_actions[i])
            if phase == GamePhase.TILES:
                coded_valid_action = np.array(coded_valid_action[0:4])
            else:
                coded_valid_action = np.concatenate((coded_valid_action[0:3], coded_valid_action[4:]))

            error = abs(np.linalg.norm(action_to_validate - coded_valid_action))
            if error == 0:
                valid = True
                break
            i += 1
        return valid

    def set_random(self, random_num=None):
        if random_num is None:
            random_num = self.progress
        np.random.seed(int(time.time()))
        random_num = max(random_num, 2)
        rand_num = np.random.randint(0, random_num)
        np.random.seed(rand_num)

        random.seed(rand_num)
        return rand_num

    def step(self, action):
        valid_actions = self.game.get_possible_actions()
        player_id = self.game.get_current_player()

        action_obj = ActionUtil.create_action(action, self.game.state.next_tile,
                                              self.game.state.board, self.game.state, player_id)
        reward = -1
        valid = self.is_action_valid(self.game.state.phase, valid_actions, action)

        if valid:
            reward = 1
            if isinstance(action_obj, TileAction):
                action_obj.tile.turn(action[3])

            self.game.step(player_id, action_obj)

            my_score = self.game.state.scores[player_id]
            reward += max(my_score - self.past_score, 0)

            self.past_score = my_score
            self.valid_total_moves += 1

        info = {}

        observation, terminated = self.other_player_move()

        self.curr_step += 1
        self.total_tried_moves += 1

        truncated = True if self.curr_step >= self.max_steps or len(valid_actions) == 0 else False

        if terminated:
            reward += 100
            PointsCollector.count_final_scores(self.game.state)
            if self.game.state.get_winner() == 1:
                reward += 100
            else:
                reward -= 10

        return observation, reward, terminated, truncated, info

    def other_player_move(self):
        player = self.game.get_current_player()

        observation = self.game.get_obs(True if player != 1 else False)
        terminated = self.game.is_finished()
        i = 0
        while player != 1 and not terminated:
            valid_actions = self.game.get_possible_actions()
            valid = False
            if self.self_play and i < 3:
                loop = asyncio.get_event_loop()
                action = loop.run_until_complete(self.self_play_get_action(observation))
                if action:
                    valid = self.is_action_valid(self.game.state.phase, valid_actions, action)

                if valid:
                    action = ActionUtil.create_action(action, self.game.state.next_tile, self.game.state.board,
                                                      self.game.state, player)
            else:
                valid_actions = self.game.get_possible_actions()
                action = random.choice(valid_actions)
                valid = True

            if valid:
                self.game.step(player, action)

            player = self.game.get_current_player()

            observation = self.game.get_obs(True if player != 1 else False)
            terminated = self.game.is_finished()
            i += 1
        return observation, terminated

    async def self_play_get_action(self, obs):
        try:
            if self.ws:
                return await self.ws.send(numpy_array_to_list(obs))
            else:
                return False
        except websockets.exceptions.ConnectionClosed as e:
            print(f"WebSocket connection closed unexpectedly with code {e.code}: {e.reason}")
            self.ws = None

        finally:
            return False


    def updateProgress(self, progress):
        progress = int(progress * 100)
        rand_num = self.set_random(progress)
        self.game.update_random_num(rand_num)

    def render(self, mode, **kwargs):
        self.game.render()

    def close(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.ws.close())
