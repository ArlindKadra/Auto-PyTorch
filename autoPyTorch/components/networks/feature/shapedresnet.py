#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNets in fancy shapes.
"""

from copy import deepcopy

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter, get_hyperparameter
from autoPyTorch.components.networks.feature.resnet import ResNet
from autoPyTorch.components.networks.feature.shapedmlpnet import get_shaped_neuron_counts

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

class ShapedResNet(ResNet):
    def __init__(self, config, in_features, out_features, *args, **kwargs):
        augmented_config = deepcopy(config)

        neuron_counts = get_shaped_neuron_counts(config['resnet_shape'],
                                                 in_features,
                                                 out_features,
                                                 config['max_units'],
                                                 config['num_groups']+2)[:-1]
        augmented_config.update(
                {"num_units_%d" % (i) : num for i, num in enumerate(neuron_counts)})
        

        if (config['use_dropout']):
            dropout_shape = get_shaped_neuron_counts(config['dropout_shape'], 0, 0, 1000, config['num_groups'])
            
            dropout_shape = [dropout / 1000 * config["max_dropout"] for dropout in dropout_shape]
        
            augmented_config.update(
                    {"dropout_%d" % (i+1) : dropout for i, dropout in enumerate(dropout_shape)})    

        super(ShapedResNet, self).__init__(augmented_config, in_features, out_features, *args, **kwargs)


    @staticmethod
    def get_config_space(
        num_groups=(1, 9),
        blocks_per_group=(1, 4),
        max_units=((10, 1024), True),
        activation=('sigmoid', 'tanh', 'relu'),
        max_shake_drop_probability=(0, 1),
        max_dropout=(0, 0.8),
        resnet_shape=('funnel', 'long_funnel', 'diamond', 'hexagon', 'brick', 'triangle', 'stairs'),
        dropout_shape=('funnel', 'long_funnel', 'diamond', 'hexagon', 'brick', 'triangle', 'stairs'),
        use_dropout=(True, False),
        use_shake_shake=(True, False),
        use_batch_normalization=(True, False),
        use_shake_drop=(True, False),
        use_skip_connection=(True, False),
    ):
        cs = CS.ConfigurationSpace()
        
        num_groups_hp = get_hyperparameter(CS.UniformIntegerHyperparameter, "num_groups", num_groups)
        cs.add_hyperparameter(num_groups_hp)
        blocks_per_group_hp = get_hyperparameter(CS.UniformIntegerHyperparameter, "blocks_per_group", blocks_per_group)
        cs.add_hyperparameter(blocks_per_group_hp)
        add_hyperparameter(cs, CS.CategoricalHyperparameter, "activation", activation)
        use_dropout_hp = add_hyperparameter(cs, CS.CategoricalHyperparameter, "use_dropout", use_dropout)
        add_hyperparameter(cs, CS.CategoricalHyperparameter, "use_shake_shake", use_shake_shake)
        add_hyperparameter(cs, CS.CategoricalHyperparameter, "use_batch_normalization", use_batch_normalization)
        add_hyperparameter(cs, CS.CategoricalHyperparameter, "use_skip_connection", use_skip_connection)

        shake_drop_hp = add_hyperparameter(cs, CS.CategoricalHyperparameter, "use_shake_drop", use_shake_drop)

        validate_if_activated = False
        if isinstance(use_shake_drop, tuple):
            if isinstance(use_shake_drop[0], list):
                value_to_check = use_shake_drop[0]

            else:
                value_to_check = use_shake_drop
                validate_if_activated = True
        else:
            if isinstance(use_shake_drop, bool):
                value_to_check = use_shake_drop

        if True in value_to_check:
            shake_drop_prob_hp = add_hyperparameter(
                cs,
                CS.UniformFloatHyperparameter,
                "max_shake_drop_probability",
                max_shake_drop_probability,
            )
            if validate_if_activated:
                cs.add_condition(CS.EqualsCondition(shake_drop_prob_hp, shake_drop_hp, True))
        
        add_hyperparameter(cs, CSH.CategoricalHyperparameter, 'resnet_shape', resnet_shape)
        add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, "max_units", max_units)

        validate_if_activated = False
        if isinstance(use_dropout, tuple):
            if isinstance(use_dropout[0], list):
                value_to_check = use_dropout[0]

            else:
                value_to_check = use_dropout
                validate_if_activated = True
        else:
            if isinstance(use_dropout, bool):
                value_to_check = use_dropout

        if True in value_to_check:
            dropout_shape_hp = add_hyperparameter(
                cs,
                CSH.CategoricalHyperparameter,
                'dropout_shape',
                dropout_shape
            )
            max_dropout_hp = add_hyperparameter(
                cs,
                CSH.UniformFloatHyperparameter,
                'max_dropout',
                max_dropout
            )
            if validate_if_activated:
                cs.add_condition(CS.EqualsCondition(dropout_shape_hp, use_dropout_hp, True))
                cs.add_condition(CS.EqualsCondition(max_dropout_hp, use_dropout_hp, True))

        return cs
