# # from django.shortcuts import render
# import json
# from django.http import JsonResponse, HttpResponseBadRequest
# from stable_baselines3 import PPO
# from gym_data.envs import CarcassoneEnv
# from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3.common.env_util import make_vec_env
# from django.views.decorators.csrf import csrf_exempt
# import numpy as np
#
# def numpy_array_to_list(obj):
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, list):
#         return [numpy_array_to_list(item) for item in obj]
#     elif isinstance(obj, dict):
#         return {key: numpy_array_to_list(value) for key, value in obj.items()}
#     else:
#         return obj
#
# env = make_vec_env(CarcassoneEnv, seed=42,  n_envs=1, vec_env_cls=SubprocVecEnv)
# model = PPO.load("../../carcassone_new/model_carcassone_curr_lern3", env, device="cpu")
#
# # Create your views here.
# @csrf_exempt
# def get_action(request):
#     global model
#     if request.method == 'POST':
#         json_data = json.loads(request.body)
#         action = model.predict(json_data)
#         print(action)
#         return JsonResponse(action[0].tolist(), safe=False)
#     return HttpResponseBadRequest()
