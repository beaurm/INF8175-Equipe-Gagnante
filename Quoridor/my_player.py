import time
from collections import deque
 
from player_quoridor import PlayerQuoridor
from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction
from game_state_quoridor import GameStateQuoridor
from actions_quoridor import Wall, Orientation
from seahorse.utils.custom_exceptions import MethodNotImplementedError
 
 
class TimeUp(Exception):
    """Raised inside the search when the per-move deadline is reached."""
 
 
class MyPlayer(PlayerQuoridor):
    """
    Player class for Quoridor game
 
    Attributes:
        piece_type (str): piece type of the player
    """
 
    MAX_WALL_CANDIDATES = 6      # best wall placements kept per node
    REPEAT_PENALTY = 1.5         # root penalty for stepping back onto a recent square
    HISTORY_LENGTH = 6           # how many of our own recent positions are remembered
 
    def __init__(self, piece_type: str, goal_row: int = 0, name: str = "bob", *args, **kwargs) -> None:
        super().__init__(piece_type, goal_row, name)
        self._deadline = None
        self._recent_positions = deque(maxlen=self.HISTORY_LENGTH)
 
    def _nearby_walls(self, state: GameStateQuoridor, around_position: tuple, margin: int = 1) -> list:
        """
        Returns the wall slots within `margin` cells of `around_position`
        (margin reduced from 2 to 1 to cut legality checks per node).
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
        Returns every legal move plus only the MAX_WALL_CANDIDATES wall
        placements near the opponent that improve the mover's
        (opponent_distance - own_distance) the most.
        """
        active = state.active_player
        other = state._opponent(active)
 
        baseline = state._shortest_path(other) - state._shortest_path(active)
 
        actions = list(state._legal_moves())
 
        opponent_position = state.rep.pawn_positions[other.get_id()]
        scored_walls = []
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
                scored_walls.append((new_metric, action))
 
        scored_walls.sort(key=lambda pair: pair[0], reverse=True)
        actions.extend(action for _, action in scored_walls[:self.MAX_WALL_CANDIDATES])
 
        return actions
 
    def _ordered_children(self, state: GameStateQuoridor, actions: list, maximizing: bool) -> list:
        """
        Pairs each action with its resulting state, sorted by evaluate_heuristic
        (best-for-me first when maximizing, worst-for-me first when minimizing).
        """
        children = [(action, state.apply_action(action)) for action in actions]
        children.sort(key=lambda pair: self.evaluate_heuristic(pair[1]), reverse=maximizing)
        return children
 
    def _time_budget(self, remaining_time, step=0):
        moves_left = max(15, 60 - step // 2)
        if remaining_time < 30:
            return 0.3
        return min(remaining_time / moves_left, 25.0)
 
    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Iterative-deepening minimax with alpha-beta pruning, a per-move time
        budget, and a penalty for returning to recently visited squares.
        Always returns the best action of the last fully completed depth.
        """
        my_id = self.get_id()
        actions = tuple(self._filtered_actions(current_state))
 
        if not actions:
            raise RuntimeError("No legal action available.")
 
        opponent_id = current_state._opponent(self).get_id()
        walls_left = current_state.rep.remaining_walls[my_id] + current_state.rep.remaining_walls[opponent_id]
        max_depth = 8 if walls_left <= 3 and len(actions) <= 6 else 6
 
        start = time.time()
        deadline = start + self._time_budget(remaining_time, current_state.get_step())
        current_position = tuple(current_state.rep.pawn_positions[my_id])
 
        ordered = self._ordered_children(current_state, actions, True)
        best_action = ordered[0][0]
 
        for depth in range(1, max_depth + 1):
            # depth 1 always completes so we always have a valid answer
            self._deadline = None if depth == 1 else deadline
            depth_best_action, depth_best_value = None, float("-inf")
            alpha, beta = float("-inf"), float("inf")
 
            try:
                for action, child_state in ordered:
                    value = self.minimax(child_state, depth - 1, alpha, beta, maximizing=False)
 
                    child_position = tuple(child_state.rep.pawn_positions[my_id])
                    if child_position != current_position and child_position in self._recent_positions:
                        value -= self.REPEAT_PENALTY
 
                    if value > depth_best_value or depth_best_action is None:
                        depth_best_value = value
                        depth_best_action = action
                    alpha = max(alpha, depth_best_value)
            except TimeUp:
                break
 
            best_action = depth_best_action
            # search the previous best first at the next depth
            ordered.sort(key=lambda pair: pair[0] is not best_action)
 
            if depth_best_value in (float("inf"), float("-inf")):
                break  # forced win/loss found, deeper search will not change it
            if time.time() >= deadline:
                break
 
        self._deadline = None
 
        self._recent_positions.append(current_position)
        return best_action
 
    def minimax(self, state: GameStateQuoridor, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """
        Returns the minimax value of `state` from this player's point of view.
        Raises TimeUp if the per-move deadline is reached.
        """
        if self._deadline is not None and time.time() > self._deadline:
            raise TimeUp()
 
        #means one of the 2 players reached the goal
        if state.is_done():
            return float("inf") if state.scores[self.get_id()] == 1.0 else float("-inf")
 
        if depth == 0:
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
        
    def evaluate_heuristic(self, state):
        opponent = state._opponent(self)
        my_id, opp_id = self.get_id(), opponent.get_id()
        my_dist = state._shortest_path(self)
        opp_dist = state._shortest_path(opponent)
        my_walls = state.rep.remaining_walls[my_id]
        opp_walls = state.rep.remaining_walls[opp_id]
        i_move = state.active_player.get_id() == my_id

        # Course pure : plus aucun mur, le résultat est quasi déterministe
        if my_walls == 0 and opp_walls == 0:
            i_win = my_dist <= opp_dist if i_move else my_dist < opp_dist
            return 100 - my_dist if i_win else -100 + opp_dist

        tempo = 0.5 if i_move else -0.5
        return 0.9 * (opp_dist - my_dist) + 0.25 * (my_walls - opp_walls) + tempo