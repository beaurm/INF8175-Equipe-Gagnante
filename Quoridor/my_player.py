from player_quoridor import PlayerQuoridor
from seahorse.game.action import Action
from game_state_quoridor import GameStateQuoridor
from seahorse.utils.custom_exceptions import MethodNotImplementedError

class MyPlayer(PlayerQuoridor):
    """
    Player class for Quoridor game

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, goal_row: int=0, name: str = "bob", *args, **kwargs) -> None:
        """
        Initialize the PlayerQuoridor instance.

        Args:
            piece_type (str): Type of the player's game piece
            goal_row (int): The row the player wants to reach
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, goal_row, name)

    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameStateQuoridor): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        #Greedy algorithm for now, with slightly different heuristic.
        actions = tuple(current_state.generate_possible_stateless_actions())

        if not actions:
            raise RuntimeError("No legal action available.")

        best_action = None
        best_cost = 100

        my_id = self.get_id()
        opponent = current_state._opponent(self)
        opponent_id = opponent.get_id()

        wall_penalty = 0.1 # Penalty for having fewer walls than the opponent
        slowing_incentive = 1.1 # Incentive to place walls / slow the enemy
        for action in actions:
            temp_state = current_state.apply_action(action)

            my_dist = temp_state._shortest_path(self)
            opp_dist = temp_state._shortest_path(opponent)

            my_walls = temp_state.rep.remaining_walls[my_id]
            opp_walls = temp_state.rep.remaining_walls[opponent_id]

            cost = my_dist - slowing_incentive * opp_dist + (opp_walls - my_walls) * wall_penalty
                
            if cost < best_cost:
                best_cost = cost
                best_action = action

        return best_action