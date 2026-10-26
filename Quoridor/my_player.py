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

    def _filtered_actions(self, state: GameStateQuoridor) -> list:
        """
        Returns every legal moving action, but only the wall placements that
        improve the mover's own (opponent_distance  own_distance) heuristic,
        the same one used as the depth cutoff heuristic. This cuts branching factor.
        """
        active = state.active_player
        other = state._opponent(active)

        baseline = state._shortest_path(other) - state._shortest_path(active)

        actions = []
        for action in state.generate_possible_stateless_actions():
            if action.data["type"] == "move":
                actions.append(action)
                continue

            temp_state = state.apply_action(action)
            new_metric = temp_state._shortest_path(other) - temp_state._shortest_path(active)
            if new_metric > baseline:
                actions.append(action)

        return actions

    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use the minimax algorithm  to choose the best action.
        Added alpha beta pruning + cutoff
        Args:
            current_state (GameStateQuoridor): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        depth = 2 # 3 moves each: me, opponent, me, opponent, me, opponent

        actions = tuple(self._filtered_actions(current_state))

        if not actions:
            raise RuntimeError("No legal action available.")

        best_action = None
        best_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for action in actions:
            child_state = current_state.apply_action(action)

            #We maximize here over value, depth bellow minimizes
            value = self.minimax(child_state, depth - 1, alpha, beta, maximizing=False)

            if value > best_value:
                best_value = value
                best_action = action

            alpha = best_value

        return best_action

    def minimax(self, state: GameStateQuoridor, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """
        Returns the minimax value of `state` from this player's point of view.

        Args:
            state (GameStateQuoridor): The current game state.
            depth (int): Remaining depth to seatch.
            alpha (float): Best value the maximizer can already guarantee.
            beta (float): Best value the minimizer can already guarantee.
            maximizing (bool): True: maximizing, false: minimizing.
        """

        #means one of the 2 players reached the goal
        if state.is_done():
            return float("inf") if state.scores[self.get_id()] == 1.0 else float("-inf")

        if depth == 0:
            #heuristic evaluating the
            return self.evaluate_heuristic(state=state)

        actions = tuple(self._filtered_actions(state))

        #maxValue in the slides
        if maximizing:
            best_val = float("-inf")
            for action in actions:
                child_state = state.apply_action(action)
                value = self.minimax(child_state, depth - 1, alpha, beta, maximizing=False)
                if value > best_val:
                    best_val = value
                    alpha = max(alpha, best_val)
                if best_val >= beta:
                    break  # beta cutoff
            return best_val
        
        #minValue in the slides
        else:
            best_val = float("inf")
            for action in actions:
                child_state = state.apply_action(action)
                value = self.minimax(child_state, depth - 1, alpha, beta, maximizing=True)
                if value < best_val:
                    best_val = value
                    beta = min(beta, best_val)
                if best_val <= alpha:
                    break  # alpha cutoff
            return best_val
    
    
    def evaluate_heuristic(self, state: GameStateQuoridor) -> float:
        """
            Evaluate the given game state using a weighted linear heuristic

            Args:
                state (GameStateQuoridor): The game state to evaluate.
                
            Returns:
                float: the heuristic score of the state from this player's perspective.
                Higher scores indicate more favorable position.
        """
            
        opponent = state._opponent(self)
        my_id = self.get_id()
        opponent_id = opponent.get_id()
        
        
        # H1: Shortest-path distance advantage
        my_dist = state._shortest_path(self)
        opp_dist = state._shortest_path(opponent)
        h1_distance = opp_dist - my_dist
        
        # H2: Remaining wall advantage
        my_walls = state.rep.remaining_walls[my_id]
        opp_walls = state.rep.remaining_walls[opponent_id]
        h2_wall_remaining = opp_walls - my_walls
        
        # H3: Central-column advantage
        center_col = (state.rep.dimension - 1) / 2
        my_position = state.rep.pawn_positions[my_id]
        opp_position = state.rep.pawn_positions[opponent_id]
        
        my_to_center = abs(my_position[1] - center_col)
        opp_to_center = abs(opp_position[1] - center_col)

        h3_position_to_center = opp_to_center - my_to_center

        # Linear heuristic with weight chosen manually
        h = (
            0.8 * h1_distance
            + 0.35 * h2_wall_remaining
            + 0.2 * h3_position_to_center
        )
        
        return h