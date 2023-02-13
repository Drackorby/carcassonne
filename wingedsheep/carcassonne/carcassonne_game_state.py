import random
from typing import Optional
import numpy as np

from wingedsheep.carcassonne.objects.meeple_type import MeepleType
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.objects.coordinate import Coordinate
from wingedsheep.carcassonne.objects.game_phase import GamePhase
from wingedsheep.carcassonne.objects.rotation import Rotation
from wingedsheep.carcassonne.objects.tile import Tile
from wingedsheep.carcassonne.tile_sets.base_deck import base_tile_counts, base_tiles
from wingedsheep.carcassonne.tile_sets.inns_and_cathedrals_deck import inns_and_cathedrals_tiles, \
    inns_and_cathedrals_tile_counts
from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule
from wingedsheep.carcassonne.tile_sets.the_river_deck import the_river_tiles, the_river_tile_counts
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet
from wingedsheep.carcassonne.objects.side import Side
from wingedsheep.carcassonne.objects.farmer_side import FarmerSide


class CarcassonneGameState:

    def __init__(
            self,
            tile_sets: [TileSet] = (TileSet.BASE, TileSet.THE_RIVER, TileSet.INNS_AND_CATHEDRALS),
            supplementary_rules: [SupplementaryRule] = (SupplementaryRule.FARMERS, SupplementaryRule.ABBOTS),
            players: int = 2,
            board_size: (int, int) = (30, 30),
            starting_position: Coordinate = Coordinate(6, 15)
    ):
        self.deck = self.initialize_deck(tile_sets=tile_sets)
        self.supplementary_rules: [SupplementaryRule] = supplementary_rules
        self.board: [[Tile]] = [[None for column in range(board_size[1])] for row in range(board_size[0])]
        self.starting_position: Coordinate = starting_position
        self.next_tile = self.deck.pop(0)
        self.players = players
        self.meeples = [7 for _ in range(players)]
        self.abbots = [1 if SupplementaryRule.ABBOTS in supplementary_rules else 0 for _ in range(players)]
        self.big_meeples = [1 if TileSet.INNS_AND_CATHEDRALS in tile_sets else 0 for _ in range(players)]
        self.placed_meeples = [[] for _ in range(players)]
        self.scores: [int] = [0 for _ in range(players)]
        self.current_player = 0
        self.phase = GamePhase.TILES
        self.last_tile_action: Optional[TileAction] = None
        self.last_river_rotation: Rotation = Rotation.NONE

    def get_tile(self, row: int, column: int):
        if row < 0 or column < 0:
            return None
        elif row >= len(self.board) or column >= len(self.board[0]):
            return None
        else:
            return self.board[row][column]

    def get_obs_city(self,city):
        if len(city) <= 0:
            return -1
        elif len(city) == 1:
            if city[0] == Side.TOP:
                return 0
            if city[0] == Side.RIGHT:
                return 4
            if city[0] == Side.BOTTOM:
                return 7
            if city[0] == Side.LEFT:
                return 9
        elif len(city) == 2:
            if city[0] == Side.TOP:
                if city[1] == Side.RIGHT:
                    return 1
                if city[1] == Side.BOTTOM:
                    return 2
                if city[1] == Side.LEFT:
                    return 3
            if city[0] == Side.RIGHT:
                if city[1] == Side.BOTTOM:
                    return 5
                if city[1] == Side.LEFT:
                    return 6
                if city[1] == Side.TOP:
                    return 1
            if city[0] == Side.BOTTOM:
                if city[1] == Side.LEFT:
                    return 8
                if city[1] == Side.TOP:
                    return 2
                if city[1] == Side.RIGHT:
                    return 5
            if city[0] == Side.LEFT:
                if city[1] == Side.TOP:
                    return 3
                if city[1] == Side.RIGHT:
                    return 6
                if city[1] == Side.BOTTOM:
                    return 8     
        elif len(city) == 3:
            if Side.TOP not in city:
                return 10
            if Side.RIGHT not in city:
                return 11
            if Side.BOTTOM not in city:
                return 12
            if Side.LEFT not in city:
                return 13
        elif len(city) == 4:
            return 14
               
        return -2

    def get_obs_road(self,road):
        if road.a == Side.TOP:
            if road.b == Side.RIGHT:
                return 0
            if road.b == Side.BOTTOM:
                return 1
            if road.b == Side.LEFT:
                return 2
            if road.b == Side.CENTER:
                return 3
        if road.a == Side.RIGHT:
            if road.b == Side.BOTTOM:
                return 4
            if road.b == Side.LEFT:
                return 5
            if road.b == Side.CENTER:
                return 6
            if road.b == Side.TOP:
                return 0
        if road.a == Side.BOTTOM:
            if road.b == Side.LEFT:
                return 7
            if road.b == Side.CENTER:
                return 8
            if road.b == Side.TOP:
                return 1
            if road.b == Side.RIGHT:
                return 4
        if road.a == Side.LEFT:
            if road.b == Side.CENTER:
                return 9
            if road.b == Side.TOP:
                return 2
            if road.b == Side.RIGHT:
                return 5
            if road.b == Side.BOTTOM:
                return 7
        if road.a == Side.CENTER:
            if road.b == Side.TOP:
                return 3
            if road.b == Side.RIGHT:
                return 6
            if road.b == Side.BOTTOM:
                return 8
            if road.b == Side.LEFT:
                return 9
               
        return -2

    def get_meeple_plane_obs(self,player, side, farmer):

        multiplier = 5

        if farmer:
            multiplier = 9
        if side == Side.TOP:
            return 0 + player * multiplier
        if side == Side.RIGHT:
            return 1 + player * multiplier
        if side == Side.BOTTOM:
            return 2 + player * multiplier
        if side == Side.LEFT:
            return 3 + player * multiplier
        if side == Side.CENTER:
            return 4 + player * multiplier
        
        if farmer:
            if side == Side.TOP_LEFT:
                return 5 + player * multiplier
            if side == Side.TOP_RIGHT:
                return 6 + player * multiplier
            if side == Side.BOTTOM_LEFT:
                return 7 + player * multiplier
            if side == Side.BOTTOM_RIGHT:
                return 8 + player * multiplier

        
        return -1


    def get_farm_connections(self, farm):
        connections = farm.tile_connections

        if len(connections) == 0:
            return 0
        if len(connections) == 1:
            if connections[0] == FarmerSide.TLT:
                return 1
            if connections[0] == FarmerSide.TRT:
                return 2
            if connections[0] == FarmerSide.TRR:
                return 3
            if connections[0] == FarmerSide.BRR:
                return 4
            if connections[0] == FarmerSide.BRB:
                return 5
            if connections[0] == FarmerSide.BLB:
                return 6
            if connections[0] == FarmerSide.BLL:
                return 7
            if connections[0] == FarmerSide.TLL:
                return 8
        if len(connections) == 2:
            if FarmerSide.TLT in connections and FarmerSide.TRT in connections :
                return 9
            if FarmerSide.TLT in connections and FarmerSide.BRR in connections :
                return 10
            
            if FarmerSide.TLT in connections and FarmerSide.BLB in connections :
                return 11
            if FarmerSide.TLT in connections and FarmerSide.TLL in connections :
                return 12
            if FarmerSide.TRT in connections and FarmerSide.TRR in connections :
                return 13
            if FarmerSide.TRT in connections and FarmerSide.BRB in connections :
                return 14
            
            if FarmerSide.TRT in connections and FarmerSide.BLL in connections :
                return 15
            if FarmerSide.TRR in connections and FarmerSide.BRR in connections :
                return 16
            if FarmerSide.TRR in connections and FarmerSide.BLB in connections :
                return 17
            
            if FarmerSide.TRR in connections and FarmerSide.TLL in connections :
                return 18
            if FarmerSide.BRR in connections and FarmerSide.BRB in connections :
                return 19
            if FarmerSide.BRR in connections and FarmerSide.BLL in connections :
                return 20
            
            if FarmerSide.BRB in connections and FarmerSide.BLB in connections :
                return 21
            if FarmerSide.BRB in connections and FarmerSide.TLL in connections :
                return 22
            if FarmerSide.BLB in connections and FarmerSide.BLL in connections :
                return 23
            if FarmerSide.BLL in connections and FarmerSide.TLL in connections :
                return 24
            
        if len(connections) == 3:

            if FarmerSide.TLT in connections and FarmerSide.TRT in connections and FarmerSide.TRR:
                return 25
            if FarmerSide.TRR in connections and FarmerSide.BRR in connections and FarmerSide.BRB:
                return 26
            if FarmerSide.BRB in connections and FarmerSide.BLB in connections and FarmerSide.BLL:
                return 27
            if FarmerSide.TLT in connections and FarmerSide.BLL in connections and FarmerSide.TLL:
                return 28
            
            if FarmerSide.BLB in connections and FarmerSide.BLL in connections and FarmerSide.TLL:
                return 29
            if FarmerSide.TLT in connections and FarmerSide.TRT in connections and FarmerSide.TLL:
                return 30
            if FarmerSide.TRT in connections and FarmerSide.TRR in connections and FarmerSide.BRR:
                return 31
            if FarmerSide.BRR in connections and FarmerSide.BRB in connections and FarmerSide.BLB:
                return 32
            

        if len(connections) == 4:
            if FarmerSide.TLT in connections and FarmerSide.BLB in connections and FarmerSide.BLL in connections and FarmerSide.TLL in connections :
                return 33
            if FarmerSide.TLT in connections and FarmerSide.TRT in connections and FarmerSide.TRR in connections and FarmerSide.TLL in connections :
                return 34
            if FarmerSide.TRT in connections and FarmerSide.TRR in connections and FarmerSide.BRR in connections and FarmerSide.BRB in connections :
                return 35
            if FarmerSide.BRR in connections and FarmerSide.BRB in connections and FarmerSide.BLB in connections and FarmerSide.BLL in connections :
                return 36
            if FarmerSide.TLT in connections and FarmerSide.TRT in connections and FarmerSide.BRB in connections and FarmerSide.BLB in connections :
                return 37
            if FarmerSide.TRR in connections and FarmerSide.BRR in connections and FarmerSide.BLL in connections and FarmerSide.TLL in connections :
                return 38
            if FarmerSide.TRT in connections and FarmerSide.TRR in connections and FarmerSide.BLL in connections and FarmerSide.BLB in connections :
                return 39
            if FarmerSide.TLT in connections and FarmerSide.BRR in connections and FarmerSide.BRB in connections and FarmerSide.TLL in connections :
                return 40
            if FarmerSide.TLT in connections and FarmerSide.TRT in connections and FarmerSide.TRR in connections and FarmerSide.BLB in connections :
                return 41
            if FarmerSide.TRR in connections and FarmerSide.BRR in connections and FarmerSide.BRB in connections and FarmerSide.TLL in connections :
                return 42
            if FarmerSide.TRT in connections and FarmerSide.BRB in connections and FarmerSide.BLB in connections and FarmerSide.BLL in connections :
                return 43
            if FarmerSide.TLT in connections and FarmerSide.BRR in connections and FarmerSide.BLL in connections and FarmerSide.TLL in connections :
                return 44
            if FarmerSide.TLT in connections and FarmerSide.TRT in connections and FarmerSide.TRR in connections and FarmerSide.BRR in connections :
                return 45
            if FarmerSide.TRR in connections and FarmerSide.BRR in connections and FarmerSide.BRB in connections and FarmerSide.BLB in connections :
                return 46
            if FarmerSide.BRB in connections and FarmerSide.BLB in connections and FarmerSide.BLL in connections and FarmerSide.TLL in connections :
                return 47
            if FarmerSide.TLT in connections and FarmerSide.TRT in connections and FarmerSide.BLL in connections and FarmerSide.TLL in connections :
                return 48
            if FarmerSide.TLT in connections and FarmerSide.TRT in connections and FarmerSide.BRB in connections and FarmerSide.TLL in connections :
                return 49
            if FarmerSide.TRR in connections and FarmerSide.BLB in connections and FarmerSide.BLL in connections and FarmerSide.TLL in connections :
                return 50
            
        if len(connections) == 6:
            if FarmerSide.TLT in connections and FarmerSide.TRT in connections and FarmerSide.TRR in connections and FarmerSide.BRR in connections and FarmerSide.BRB in connections and FarmerSide.TLL in connections:
                return 50
            if FarmerSide.TRT in connections and FarmerSide.TRR in connections and FarmerSide.BRR in connections and FarmerSide.BRB in connections and FarmerSide.BLB in connections and FarmerSide.BLL in connections:
                return 51
            if FarmerSide.TLT in connections and FarmerSide.BRR in connections and FarmerSide.BRB in connections and FarmerSide.BLB in connections and FarmerSide.BLL in connections and FarmerSide.TLL in connections:
                return 52
            if FarmerSide.TLT in connections and FarmerSide.TRT in connections and FarmerSide.TRR in connections and FarmerSide.BLB in connections and FarmerSide.BLL in connections and FarmerSide.TLL in connections:
                return 53
            if FarmerSide.TRR in connections and FarmerSide.BRR in connections and FarmerSide.BRB in connections and FarmerSide.BLB in connections and FarmerSide.BLL in connections and FarmerSide.TLL in connections:
                return 54
            if FarmerSide.TLT in connections and FarmerSide.TRT in connections and FarmerSide.BRB in connections and FarmerSide.BLB in connections and FarmerSide.BLL in connections and FarmerSide.TLL in connections:
                return 55
            if FarmerSide.TLT in connections and FarmerSide.TRT in connections and FarmerSide.TRR in connections and FarmerSide.BRR in connections and FarmerSide.BLL in connections and FarmerSide.TLL in connections:
                return 56
            if FarmerSide.TLT in connections and FarmerSide.TRT in connections and FarmerSide.TRR in connections and FarmerSide.BRB in connections and FarmerSide.BRB in connections and FarmerSide.BLB in connections:
                return 57
        if len(connections) == 8:
            return 58
        return -1
            

    def get_obs(self):

        
        city_planes = np.zeros((15,len(self.board),len(self.board)),dtype=np.intc)
        road_planes = np.zeros((10,len(self.board),len(self.board)),dtype=np.intc)
        chapel_plane = np.zeros((len(self.board),len(self.board)),dtype=np.intc)
        shield_plane = np.zeros((len(self.board),len(self.board)),dtype=np.intc)
        flowers_plane = np.zeros((len(self.board),len(self.board)),dtype=np.intc)
        field_planes = np.zeros((59,len(self.board),len(self.board)),dtype=np.intc)
        # Next we will have n planes per player and most of them will need 5 planes to indicate the position in a tile
        meeple_planes = np.zeros((5*self.players , len(self.board),len(self.board)),dtype=np.intc)
        abbot_planes = np.zeros( (self.players ,len(self.board),len(self.board)),dtype=np.intc) # This only needs one plane per player
        farmer_planes = np.zeros((9*self.players , len(self.board),len(self.board)),dtype=np.intc)
        big_farmer_planes = np.zeros((9*self.players , len(self.board),len(self.board)),dtype=np.intc)
        big_meeples_planes =np.zeros((5*self.players , len(self.board),len(self.board)),dtype=np.intc)
        
        
        for player_id in range(len(self.placed_meeples)):
            player_meeples = self.placed_meeples[player_id]
            for meeple in player_meeples:
                coordinate = meeple.coordinate_with_side.coordinate
                side = meeple.coordinate_with_side.side
                                   
                if meeple.meeple_type == MeepleType.NORMAL:
                    meeple_planes[self.get_meeple_plane_obs(player_id, side,False),coordinate.row, coordinate.column] = 1
                if meeple.meeple_type == MeepleType.ABBOT:
                    abbot_planes[player_id,coordinate.row, coordinate.column] = 1
                if meeple.meeple_type == MeepleType.FARMER:
                    farmer_planes[self.get_meeple_plane_obs(player_id, side,True),coordinate.row, coordinate.column] = 1
                if meeple.meeple_type == MeepleType.BIG:
                    big_meeples_planes[self.get_meeple_plane_obs(player_id, side,False),coordinate.row, coordinate.column] = 1
                if meeple.meeple_type == MeepleType.BIG_FARMER:
                    big_farmer_planes[self.get_meeple_plane_obs(player_id, side, True),coordinate.row, coordinate.column] = 1
        
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                tile = self.get_tile(i,j)
                if tile:
                    for city in tile.city:
                        result = self.get_obs_city(city)
                        if result >= 0 :
                            city_planes[result,i,j] = 1
                    
                    for road in tile.road:
                        result = self.get_obs_road(road)
                        if result >= 0 :
                            road_planes[result,i,j] = 1
                    if tile.shield:
                        shield_plane[i,j] = 1
                    if tile.chapel:
                        chapel_plane[i,j] = 1
                    if tile.flowers:
                        flowers_plane[i,j] = 1
                    if len(tile.farms) >0:
                        for farm in tile.farms:
                            result = self.get_farm_connections(farm)
                            if result >= 0:
                                field_planes[result,i,j] = 1
                                    
        next_tile_description = np.zeros((6),dtype=np.intc)
        tile = self.next_tile
        if tile:
            for city in tile.city:
                result = self.get_obs_city(city)
                if result >= 0 :
                    next_tile_description[0] = result
            
            for road in tile.road:
                result = self.get_obs_road(road)
                if result >= 0 :
                    next_tile_description[1] = result
            if tile.shield:
                next_tile_description[2] = 1
            if tile.chapel:
                next_tile_description[3] = 1
            if tile.flowers:
                next_tile_description[4] = 1
            if len(tile.farms) >0:
                for farm in tile.farms:
                    result = self.get_farm_connections(farm)
                    if result >= 0:
                        next_tile_description[5] = result
                            

        phase = 0
        if self.phase == GamePhase.MEEPLES:
            phase = 1
        
        other_properties_plane = np.concatenate((
        next_tile_description,
        np.array(self.meeples,dtype=np.intc),
        np.array(self.abbots,dtype=np.intc),
        np.array(self.big_meeples,dtype=np.intc),
        np.array(self.placed_meeples,dtype=np.intc).flatten(),
        np.array(self.scores,dtype=np.intc),
        [int(self.current_player)],
        [int(phase)]),dtype=np.intc)


        return { 
        "city_planes": city_planes.flatten(),
        "road_planes": road_planes.flatten(),
        "chapel_plane": chapel_plane.flatten(),
        "shield_plane": shield_plane.flatten(),
        "flowers_plane": flowers_plane.flatten(),
        "field_planes": field_planes.flatten(),
        "meeple_planes": meeple_planes.flatten(),
        "abbot_planes": abbot_planes.flatten(),
        "farmer_planes": farmer_planes.flatten(),
        "big_farmer_planes": big_farmer_planes.flatten(),
        "big_meeples_planes": big_meeples_planes.flatten(),
        "other_properties_plane": other_properties_plane.flatten()
     }
        

    def empty_board(self):
        for row in self.board:
            for column in row:
                if column is not None:
                    return False
        return True

    def is_terminated(self) -> bool:
        self.get_obs()
        return self.next_tile is None

    def initialize_deck(self, tile_sets: [TileSet]):
        deck: [Tile] = []

        # The river
        if TileSet.THE_RIVER in tile_sets:
            deck.append(the_river_tiles["river_start"])

            new_tiles = []
            for card_name, count in the_river_tile_counts.items():
                if card_name == "river_start":
                    continue
                if card_name == "river_end":
                    continue

                for i in range(count):
                    new_tiles.append(the_river_tiles[card_name])

            random.shuffle(new_tiles)
            for tile in new_tiles:
                deck.append(tile)

            deck.append(the_river_tiles["river_end"])

        new_tiles = []

        if TileSet.BASE in tile_sets:
            for card_name, count in base_tile_counts.items():
                for i in range(count):
                    new_tiles.append(base_tiles[card_name])

        if TileSet.INNS_AND_CATHEDRALS in tile_sets:
            for card_name, count in inns_and_cathedrals_tile_counts.items():
                for i in range(count):
                    new_tiles.append(inns_and_cathedrals_tiles[card_name])

        random.shuffle(new_tiles)
        for tile in new_tiles:
            deck.append(tile)

        return deck
