from .basic_training import BasicTraining
from .training import Training
from .training_with_pretrain import TrainingWithPretrain

TRAININGS = {
    "BasicTraining": BasicTraining,
    "TrainingWithPretrain": TrainingWithPretrain,
}
