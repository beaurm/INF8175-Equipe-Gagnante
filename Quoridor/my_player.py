from player_quoridor import PlayerQuoridor
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction
from game_state_quoridor import GameStateQuoridor
from actions_quoridor import Wall, Orientation
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

    def _nearby_walls(self, state: GameStateQuoridor, around_position: tuple, margin: int = 2) -> list:
        """
        Returns the wall slots within `margin` cells of `around_position`,
        instead of every slot on the board. This misses a wall that would
        matter far from that position, trading completeness for fewer
        legality checks per node.
        """
        dimension = state.rep.dimension
        row, col = around_position

        walls = []
        for r in range(max(0, row - margin), min(dimension - 1, row + margin) + 1):
            for c in range(max(0, col - margin), min(dimension - 1, col + margin) + 1):
                walls.append(Wall(row=r, col=c, orientation=Orientation.HORIZONTAL))
                walls.append(Wall(row=r, col=c, orientation=Orientation.VERTICAL))

        return walls

    def _filtered_actions(self, state: GameStateQuoridor) -> list:
        """
        Returns every legal moving action, but only the wall placements near
        the opponent that improve the mover's own (opponent_distance -
        own_distance) heuristic, the same one used as the depth cutoff
        heuristic. Only scanning wall slots near the opponent, instead of
        every slot on the board, cuts both the branching factor and the
        number of legality checks paid per node.
        """
        active = state.active_player
        other = state._opponent(active)

        baseline = state._shortest_path(other) - state._shortest_path(active)

        actions = list(state._legal_moves())

        opponent_position = state.rep.pawn_positions[other.get_id()]
        for wall in self._nearby_walls(state, opponent_position):
            if not state._is_wall_legal(wall):
                continue

            action = StatelessAction({
                "type": "vertical" if wall.orientation == Orientation.VERTICAL else "horizontal",
                "destination": (wall.row, wall.col),
            })

            temp_state = state.apply_action(action)
            new_metric = temp_state._shortest_path(other) - temp_state._shortest_path(active)
            if new_metric > baseline:
                actions.append(action)

        return actions

    def _ordered_children(self, state: GameStateQuoridor, actions: list, maximizing: bool) -> list:
        """
        Pairs each action with its resulting state, sorted by evaluate_heuristic
        so the most promising actions are tried first. Best-for-me first when
        maximizing, worst-for-me first when minimizing, since that's the order
        the mover at this node would actually prefer, so alpha-beta tightens its
        bound sooner and prunes more of the remaining siblings.
        """
        children = [(action, state.apply_action(action)) for action in actions]
        children.sort(key=lambda pair: self.evaluate_heuristic(pair[1]), reverse=maximizing)
        return children

    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use the minimax algorithm  to choose the best action.
        Added alpha beta pruning + cutoff
        Args:
            current_state (GameStateQuoridor): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        actions = tuple(self._filtered_actions(current_state))

        if not actions:
            raise RuntimeError("No legal action available.")

        opponent_id = current_state._opponent(self).get_id()
        walls_left = current_state.rep.remaining_walls[self.get_id()] + current_state.rep.remaining_walls[opponent_id]

        depth = 6 if walls_left <= 3 and len(actions) <= 6 else 4 # 3 moves each: me, opponent, me, opponent, me, opponent

        best_action = None
        best_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for action, child_state in self._ordered_children(current_state, actions, True):

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
            for action, child_state in self._ordered_children(state, actions, True):
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
            for action, child_state in self._ordered_children(state, actions, False):
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