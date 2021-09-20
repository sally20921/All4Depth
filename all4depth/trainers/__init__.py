"""
Trainers
========

Trainer classes providing an easy way to train and evaluate SfM models
when wrapped in a ModelWrapper.

Inspired by pytorch-lightning.

"""

from all4depth.trainers.horovod_trainer import HorovodTrainer

__all__ = ["HorovodTrainer"]