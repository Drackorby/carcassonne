
# This code snippet allows us to count the unique farm connections in the current tile set

from wingedsheep.carcassonne.tile_sets.base_deck import base_tiles
from wingedsheep.carcassonne.tile_sets.the_river_deck import the_river_tiles
from wingedsheep.carcassonne.tile_sets.inns_and_cathedrals_deck import inns_and_cathedrals_tiles
from wingedsheep.carcassonne.utils.side_modification_util import SideModificationUtil

def get_unique_farm_connections(tiles):
    unique_farm_connections = []
    for tile in tiles:
        if len(tiles[tile].farms) > 0:
            for farm in tiles[tile].farms:
                for i in range(4):
                    farm = SideModificationUtil.turn_farmer_connection(farm, i)
                    if len(farm.tile_connections) > 0:
                        tile_connections = farm.tile_connections
                        # Check if the farm connections are already in the list of unique farm connections independently of the order
                        tile_connections.sort()
                        if not tile_connections in unique_farm_connections:
                            unique_farm_connections.append(tile_connections)
                
    return unique_farm_connections

# Concatenate the dicts of tiles
complete_set = {**base_tiles, **the_river_tiles, **inns_and_cathedrals_tiles}

a = get_unique_farm_connections(complete_set)
for i in a:
    print(i)

print(len(a))