import numpy as np
import json

board_size = 10
n_players = 2

other_properties_space = np.ones(12) * 2

other_properties_space[-4] = 250
other_properties_space[-3] = 250
other_properties_space[-2] = n_players
other_properties_space[-1] = 3
other_properties_space[0] = 15
other_properties_space[1] = 10
other_properties_space[5] = 59
other_properties_space[6] = 8  # Meeples
other_properties_space[7] = 8


obs_spaces = {
            "city_planes": [15, board_size, board_size],
            "road_planes": [10, board_size, board_size],
            "chapel_plane": [board_size, board_size],
            "shield_plane": [board_size, board_size],
            "flowers_plane": [board_size, board_size],
            "field_planes": [66, board_size, board_size],
            "meeple_planes": [5 * n_players, board_size, board_size],
            "abbot_planes": [n_players, board_size, board_size],
            "farmer_planes": [5 * n_players, board_size, board_size],
            "big_farmer_planes": [5 * n_players, board_size, board_size],
            "big_meeples_planes": [5 * n_players, board_size, board_size],
            "other_properties_plane": other_properties_space
}

with open("trajectories.json", "r") as json_file:
    data = json.load(json_file)
    for item in data:
        state = item.get("state")
        action = item.get("action")  

        if state is not None and action is not None:
            for key, value in state.items():
                state[key] = np.array(value)

                if key == "other_properties_plane":
                    curr_state = state[key]
                    space = obs_spaces[key]
                    for i  in range(len(curr_state)):
                        if curr_state[i] >= space[i]:
                            print("Curr_state: ", curr_state, " space: ", space)


                
                
            


