from gymnasium_2048.agents.SL1.data import (
    ExpectimaxDataset,
    WeightConfig,
    compute_sample_weights,
)
from gymnasium_2048.agents.SL1.config import (
    SUPERVISED_RUN_ROOT,
    SupervisedTrainingYamlConfig,
    load_supervised_training_config,
    resolve_supervised_output_dir,
    train_supervised_from_yaml,
)
from gymnasium_2048.agents.SL1.encoding import (
    encode_board,
    encode_boards,
    encode_boards_torch,
)
from gymnasium_2048.agents.SL1.loss import masked_soft_cross_entropy
from gymnasium_2048.agents.SL1.model import CNNConfig, SupervisedCNN
from gymnasium_2048.agents.SL1.policy import (
    SupervisedCNNPolicy,
    SupervisedCNNResult,
)
from gymnasium_2048.agents.SL1.train import (
    SupervisedTrainingConfig,
    resolve_device,
    train_supervised_cnn,
)

