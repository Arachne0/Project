import random


class RandomAction(object):

    def __init__(self):
        pass

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, env):
        available = [i for i in range(36) if env.state_[3][i // 4][i % 4] != 1]
        sensible_moves = available

        if len(sensible_moves) > 0:
            move = random.choice(sensible_moves)
            return move
        else:
            print("WARNING: the board is full")
