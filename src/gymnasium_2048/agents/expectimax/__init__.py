from gymnasium_2048.agents.expectimax.board import (
    ACTION_NAMES,
    NUM_ACTIONS,
    board_from_observation,
    legal_action_mask,
    masked_softmax,
    normalize_legal_mask,
)
from gymnasium_2048.agents.expectimax.data import (
    augment_afterstate_samples,
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
from gymnasium_2048.agents.expectimax.search import (
    AfterstateEvaluator,
    ChanceOutcome,
    RootActionValues,
    apply_player_action,
    evaluate_root_actions,
    expand_chance_node,
    expectimax_afterstate_value,
    player_state_value,
    select_best_action,
)
from gymnasium_2048.agents.expectimax.symmetry import (
    NUM_SYMMETRIES,
    all_symmetries,
    apply_symmetry,
    random_symmetry,
    transform_action,
)
