from gymnasium_2048.agents.expectimax.board import (
    ACTION_NAMES,
    NUM_ACTIONS,
    board_from_observation,
    legal_action_mask,
    masked_softmax,
    normalize_legal_mask,
)
from gymnasium_2048.agents.expectimax.data import (
    generate_expectimax_dataset,
    load_expectimax_dataset,
    save_expectimax_dataset,
)
from gymnasium_2048.agents.expectimax.heuristic import (
    DEFAULT_WEIGHTS,
    ExpectimaxHeuristicWeights,
    evaluate_board,
)
from gymnasium_2048.agents.expectimax.fast_move import fast_apply_action
from gymnasium_2048.agents.expectimax.policy import ExpectimaxPolicy, ExpectimaxResult
