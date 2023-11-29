import json
from stable_baselines3 import PPO
from gym_data.envs import CarcassoneEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from channels.generic.websocket import AsyncJsonWebsocketConsumer


def numpy_array_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [numpy_array_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_array_to_list(value) for key, value in obj.items()}
    else:
        return obj


env = make_vec_env(CarcassoneEnv, seed=42, n_envs=1, vec_env_cls=SubprocVecEnv)
model = PPO.load("../../carcassone_new/model_carcassone_curr_lern3", env, device="cpu")


class ModelConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        self.model_id = self.scope["url_route"]["kwargs"]["id"]
        self.model_group_id = 'id_%s' % self.model_id

        print("Model : ", self.model_id, " connected successfully")
        await self.channel_layer.group_add(
            self.model_group_id,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        print(self.model_group_id, " disconnected with code: ", close_code)
        await self.channel_layer.group_discard(
            self.model_group_id,
            self.channel_name
        )

    async def receive(self, text_data):
        json_data = json.loads(text_data)
        action = model.predict(json_data)
        await self.channel_layer.group_send(self.model_group_id, {'type': 'send_message', "action": action[0].tolist()})

    async def send_message(self, res):
        await self.send(text_data=json.dumps({
            "payload": res,
        }))
