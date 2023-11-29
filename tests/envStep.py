
import unittest


from wingedsheep.carcassonne.utils.action_util import ActionUtil
from wingedsheep.carcassonne.objects.game_phase import GamePhase
import numpy as np

def is_valid_action( action, valid_actions, phase):
    action_to_validate = []
    if phase == GamePhase.TILES:
        action_to_validate = np.array(action[0:4])
    else:
        action_to_validate = np.concatenate((action[0:3],action[4:]))
    
    for valid_action in valid_actions:
        coded_valid_action = valid_action
        if phase == GamePhase.TILES:
            coded_valid_action = np.array(coded_valid_action[0:4])
        else:
            coded_valid_action = np.concatenate((coded_valid_action[0:3],coded_valid_action[4:]))
            
        error = abs(np.linalg.norm(action_to_validate - coded_valid_action)) 
        if  error == 0:
            return True
    
    return False


class TestCarcassoneEnv(unittest.TestCase):
    
    
            
    
    def test_valid_action(self):
        # Test a valid action for the TILES phase
        action = [0, 0, 0, 0, 0, 0]
        valid_actions = [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], [2, 2, 2, 2, 2, 0]]
        phase = GamePhase.TILES
        self.assertTrue(is_valid_action(action, valid_actions, phase))
        
        # Test an invalid action for the TILES phase
        action = [0, 0, 0, 0, 1, 0]
        valid_actions = [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], [2, 2, 2, 2, 2, 0]]
        phase = GamePhase.TILES
        self.assertTrue(is_valid_action(action, valid_actions, phase))
        
        # Test a valid action for the MEEPLES phase
        action = [0, 0, 0, 1, 0, 0]
        valid_actions = [[0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]]
        phase = GamePhase.MEEPLES
        self.assertTrue(is_valid_action(action, valid_actions, phase))
        
        # Test an invalid action for the MEEPLES phase
        action = [0, 0, 0, 1, 0, 1]
        valid_actions = [[0, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]]
        phase = GamePhase.MEEPLES
        self.assertFalse(is_valid_action(action, valid_actions, phase))
        

       