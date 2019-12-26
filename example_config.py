import hashlib

from applications import cyclegan_disc, cyclegan_gen_9
from configs.abstract_config import AbstractConfig
from CycleGAN import CycleGANBase


class CustomConfig(AbstractConfig):

    def __init__(self, name):
        super().__init__(name)

        # Edit from here
        # Run metadata
        self.gpus = 1  # Up to 2
        self.name = name
        self.seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % (
            10 ** 8)

        # Dataset size
        self.dataA_channels = 3
        self.dataB_channels = 3
        self.dataset_size = 1500  # Same size for the 2 datasets
        self.resize_size = 500  # The images sizes are first standardized with a resizing
        self.image_size = 200  # Then patches are randomly cropped at training time

        # Training settings
        # These are the standard CycleGAN parameters

        self.cyc_factor = 10  # The Lambda hyperparameter, controls the importance of the reconstruction term
        self.pool_size = 50  # Size of the generated image pool

        self.initial_learning_rate = 0.0002
        self.final_learning_rate = 0.000002  # From mid-training on, learning rate decays linearly

        self.batch_size = 1
        self.epochs = 200

        # Networks setup
        self.genA = cyclegan_gen_9.create_network
        self.genA_args = {"channels_out": 3, "name": "GenA"}

        self.genB = cyclegan_gen_9.create_network
        self.genB_args = {"channels_out": 3, "name": "GenB"}

        self.discA = cyclegan_disc.create_network
        self.discA_args = {"channels": 3, "name": "DiscA"}

        self.discB = cyclegan_disc.create_network
        self.discB_args = {"channels": 3, "name": "DiscB"}
