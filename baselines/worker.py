from copy import deepcopy
from functools import partial
from typing import Dict, Tuple, Union

import ConfigSpace as cs
from hpbandster.core.worker import Worker
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import balanced_accuracy_score
import torch
import xgboost as xgb


def balanced_error(
    threshold_predictions: bool,
    predt: np.ndarray,
    dtrain: xgb.DMatrix,
) -> Tuple[str, float]:
    """Calculate the balanced error for the predictions.

    Calculate the balanced error. Used as an evaluation metric for
    the xgboost algorithm.

    Parameters:
    -----------
    threshold_predictions: bool
        If the predictions should be threshold to 0 or 1. Should only be used for
        binary classification.
    predt: np.ndarray
        The predictions of the algorithm.
    dtrain: float
        The real values for the set.

    Returns:
    --------
    str, float - The name of the evaluation metric and its value on the arguments.
    """

    if threshold_predictions:
        predt = np.array(predt)
        predt = predt > 0.5
        predt = predt.astype(int)
    else:
        predt = np.argmax(predt, axis=1)
    y_train = dtrain.get_label()
    accuracy_score = balanced_accuracy_score(y_train, predt)

    return 'Balanced_error', 1 - accuracy_score


class XGBoostWorker(Worker):

    def __init__(self, *args, param=None, splits=None, categorical_information=None, **kwargs):

        super().__init__(*args, **kwargs)
        self.param = param
        self.splits = splits
        self.categorical_ind = categorical_information

        if self.param['objective'] == 'binary:logistic':
            self.threshold_predictions = True
        else:
            self.threshold_predictions = False

    def compute(self, config, budget, **kwargs):
        """What should be computed for one XGBoost worker.

        The function takes a configuration and a budget, it
        then uses the xgboost algorithm to generate a loss
        and other information.

        Parameters:
        -----------
            config: dict
                dictionary containing the sampled configurations by the optimizer
            budget: float
                amount of time/epochs/etc. the model can use to train
        Returns:
        --------
            dict:
                With the following mandatory arguments:
                'loss' (scalar)
                'info' (dict)
        """
        xgboost_config = deepcopy(self.param)
        xgboost_config.update(config)
        num_rounds = xgboost_config['num_round']
        del xgboost_config['num_round']
        X_train = self.splits['X_train']
        X_val = self.splits['X_val']
        X_test = self.splits['X_test']
        y_train = self.splits['y_train']
        y_val = self.splits['y_val']
        y_test = self.splits['y_test']

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_val = xgb.DMatrix(X_val, label=y_val)
        d_test = xgb.DMatrix(X_test, label=y_test)

        eval_results = {}
        gb_model = xgb.train(
            xgboost_config,
            d_train,
            num_rounds,
            feval=partial(balanced_error, self.threshold_predictions),
            evals=[(d_train, 'd_train'), (d_val, 'd_val')],
            evals_result=eval_results,
        )
        # TODO Do something with eval_results in the future
        # print(eval_results)
        # make prediction
        y_train_preds = gb_model.predict(d_train)
        y_val_preds = gb_model.predict(d_val)
        y_test_preds = gb_model.predict(d_test)

        if self.threshold_predictions:
            y_train_preds = np.array(y_train_preds)
            y_train_preds = y_train_preds > 0.5
            y_train_preds = y_train_preds.astype(int)

            y_val_preds = np.array(y_val_preds)
            y_val_preds = y_val_preds > 0.5
            y_val_preds = y_val_preds.astype(int)

            y_test_preds = np.array(y_test_preds)
            y_test_preds = y_test_preds > 0.5
            y_test_preds = y_test_preds.astype(int)

        train_performance = balanced_accuracy_score(y_train, y_train_preds)
        val_performance = balanced_accuracy_score(y_val, y_val_preds)
        test_performance = balanced_accuracy_score(y_test, y_test_preds)

        if val_performance is None or val_performance is np.inf:
            val_error_rate = 1
        else:
            val_error_rate = 1 - val_performance

        res = {
            'train_accuracy': float(train_performance),
            'val_accuracy': float(val_performance),
            'test_accuracy': float(test_performance),
        }

        return ({
            'loss': float(val_error_rate),  # this is the a mandatory field to run hyperband
            'info': res  # can be used for any user-defined information - also mandatory
        })

    def refit(self, config):
        """Runs refit on the best configuration.

        The function refits on the best configuration. It then
        proceeds to train and test the network, this time combining
        the train and validation set together for training. Probably,
        in the future, a budget should be added too as an argument to
        the parameter.

        Parameters:
        -----------
            config: dict
                dictionary containing the sampled configurations by the optimizer
        Returns:
        --------
            res: dict
                Dictionary with the train and test accuracy.
        """
        xgboost_config = deepcopy(self.param)
        xgboost_config.update(config)
        num_rounds = xgboost_config['num_round']
        del xgboost_config['num_round']
        X_train = self.splits['X_train']
        X_test = self.splits['X_test']
        y_train = self.splits['y_train']
        y_test = self.splits['y_test']

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_test = xgb.DMatrix(X_test, label=y_test)

        eval_results = {}
        gb_model = xgb.train(
            xgboost_config,
            d_train,
            num_rounds,
            feval=partial(balanced_error, self.threshold_predictions),
            evals=[(d_train, 'd_train'), (d_test, 'd_test')],
            evals_result=eval_results,
        )
        # TODO do something with eval_results
        # print(eval_results)
        # make prediction
        y_train_preds = gb_model.predict(d_train)
        y_test_preds = gb_model.predict(d_test)

        if self.threshold_predictions:
            y_train_preds = np.array(y_train_preds)
            y_train_preds = y_train_preds > 0.5
            y_train_preds = y_train_preds.astype(int)

            y_test_preds = np.array(y_test_preds)
            y_test_preds = y_test_preds > 0.5
            y_test_preds = y_test_preds.astype(int)

        train_performance = balanced_accuracy_score(y_train, y_train_preds)
        test_performance = balanced_accuracy_score(y_test, y_test_preds)

        if test_performance is None or test_performance is np.inf:
            test_performance = 0

        res = {
            'train_accuracy': float(train_performance),
            'test_accuracy': float(test_performance),
        }

        return res

    @staticmethod
    def get_default_configspace(
            seed: int = 11,
    ) -> cs.ConfigurationSpace:
        """Get the hyperparameter search space.

        The function provides the configuration space that is
        used to generate the algorithm specific hyperparameter
        search space.

        Parameters:
        -----------
        seed: int
            The seed used to build the configuration space.
        Returns:
        --------
        config_space: cs.ConfigurationSpace
            Configuration space for XGBoost.
        """
        config_space = cs.ConfigurationSpace(seed=seed)
        # learning rate
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'eta',
                lower=0.001,
                upper=1,
                log=True,
            )
        )
        # l2 regularization
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'lambda',
                lower=1E-10,
                upper=1,
                log=True,
            )
        )
        # l1 regularization
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'alpha',
                lower=1E-10,
                upper=1,
                log=True,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformIntegerHyperparameter(
                'num_round',
                lower=1,
                upper=1000,
            )
        )
        booster = cs.CategoricalHyperparameter(
            'booster',
            choices=['gbtree', 'dart'],
        )
        config_space.add_hyperparameter(
            booster,
        )
        rate_drop = cs.UniformFloatHyperparameter(
            'rate_drop',
            1e-10,
            1-(1e-10),
            default_value=0.5,
        )
        config_space.add_hyperparameter(
            rate_drop,
        )
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'gamma',
                lower=0.1,
                upper=1,
                log=True,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'colsample_bylevel',
                lower=0.1,
                upper=1,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'colsample_bynode',
                lower=0.1,
                upper=1,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'colsample_bytree',
                lower=0.5,
                upper=1,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformIntegerHyperparameter(
                'max_depth',
                lower=1,
                upper=20,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformIntegerHyperparameter(
                'max_delta_step',
                lower=0,
                upper=10,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'min_child_weight',
                lower=0.1,
                upper=20,
                log=True,
            )
        )
        config_space.add_hyperparameter(
            cs.UniformFloatHyperparameter(
                'subsample',
                lower=0.01,
                upper=1,
            )
        )

        config_space.add_condition(
            cs.EqualsCondition(
                rate_drop,
                booster,
                'dart',
            )
        )

        return config_space

    @staticmethod
    def get_parameters(
            nr_classes: int,
            seed: int = 11,
            nr_threads: int = 1
    ) -> Dict[str, Union[int, str]]:
        """Get the parameters of the method.

        Get a dictionary based on the arguments given to the
        function, which will be used to as the initial configuration
        for the algorithm.

        Parameters:
        -----------
        nr_classes: int
            The number of classes in the dataset that will be used
            to train the model.
        seed: int
            The seed that will be used for the model.
        nr_threads: int
            The number of parallel threads that will be used for
            the model.

        Returns:
        --------
        param: dict
            A dictionary that will be used as a configuration for the
            algorithm.
        """
        param = {
            'disable_default_eval_metric': 1,
            'seed': seed,
            'nthread': nr_threads,
        }
        if nr_classes != 2:
            param.update(
                {
                    'objective': 'multi:softmax',
                    'num_class': nr_classes + 1,
                }
            )
        else:
            param.update(
                {
                    'objective': 'binary:logistic',

                }
            )

        return param


class TabNetWorker(Worker):

    def __init__(
            self,
            *args,
            param: dict,
            splits: dict,
            categorical_information: dict,
            **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.param = param
        self.splits = splits
        if categorical_information is not None:
            self.categorical_ind = categorical_information['categorical_ind']
            self.categorical_columns = categorical_information['categorical_columns']
            self.categorical_dimensions = categorical_information['categorical_dimensions']
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(self.param['seed'])

    def compute(self, config: dict, budget: float, **kwargs) -> Dict:
        """What should be computed for one TabNet worker.

        The function takes a configuration and a budget, it
        then uses the tabnet algorithm to generate a loss
        and other information.

        Parameters:
        -----------
            config: dict
                dictionary containing the sampled configurations by the optimizer
            budget: float
                amount of time/epochs/etc. the model can use to train

        Returns:
        --------
            dict:
                With the following mandatory arguments:
                'loss' (scalar)
                'info' (dict)
        """
        X_train = self.splits['X_train']
        X_val = self.splits['X_val']
        X_test = self.splits['X_test']
        y_train = self.splits['y_train']
        y_val = self.splits['y_val']
        y_test = self.splits['y_test']

        clf = TabNetClassifier(
            n_a=config['na'],
            n_d=config['na'],
            n_steps=config['nsteps'],
            gamma=config['gamma'],
            lambda_sparse=config['lambda_sparse'],
            momentum=config['mb'],
            cat_idxs=self.categorical_columns,
            cat_dims=self.categorical_dimensions,
            seed=self.param['seed'],
            optimizer_params={
                'lr': config['learning_rate'],
            },
            scheduler_params={
                'step_size': config['decay_iterations'],
                'gamma': config['decay_rate'],
            },
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
        )
        batch_size = config['batch_size']
        if batch_size == 32768:
            vbatch_size = config['vbatch_size1']
        elif batch_size == 16384:
            vbatch_size = config['vbatch_size2']
        elif batch_size == 8192:
            vbatch_size = config['vbatch_size3']
        elif batch_size == 4096:
            vbatch_size = config['vbatch_size4']
        elif batch_size == 2048:
            vbatch_size = config['vbatch_size5']
        elif batch_size == 1024:
            vbatch_size = config['vbatch_size6']
        elif batch_size == 512:
            vbatch_size = config['vbatch_size7']
        elif batch_size == 256:
            vbatch_size = config['vbatch_size8']
        else:
            raise ValueError('Illegal batch size given')

        clf.fit(
            X_train=X_train,
            y_train=y_train,
            batch_size=batch_size,
            virtual_batch_size=vbatch_size,
            eval_set=[(X_val, y_val)],
            eval_name=['Validation'],
            eval_metric=['balanced_accuracy'],
            max_epochs=200,
            patience=0,
        )

        y_train_preds = clf.predict(X_train)
        y_val_preds = clf.predict(X_val)
        y_test_preds = clf.predict(X_test)

        train_performance = balanced_accuracy_score(y_train, y_train_preds)
        val_performance = balanced_accuracy_score(y_val, y_val_preds)
        test_performance = balanced_accuracy_score(y_test, y_test_preds)

        if val_performance is None or val_performance is np.inf:
            val_error_rate = 1
        else:
            val_error_rate = 1 - val_performance

        res = {
            'train_accuracy': float(train_performance),
            'val_accuracy': float(val_performance),
            'test_accuracy': float(test_performance),
        }

        return ({
            'loss': float(val_error_rate),  # this is the a mandatory field to run hyperband
            'info': res  # can be used for any user-defined information - also mandatory
        })

    def refit(self, config: dict) -> Dict:
        """Runs refit on the best configuration.

        The function refits on the best configuration. It then
        proceeds to train and test the network, this time combining
        the train and validation set together for training. Probably,
        in the future, a budget should be added too as an argument to
        the parameter.

        Parameters:
        -----------
            config: dict
                dictionary containing the sampled configurations by the optimizer
        Returns:
        --------
            res: dict
                Dictionary with the train and test accuracy.
        """
        X_train = self.splits['X_train']
        X_test = self.splits['X_test']
        y_train = self.splits['y_train']
        y_test = self.splits['y_test']

        categorical_columns = []
        categorical_dimensions = []

        for index, categorical_column in enumerate(self.categorical_ind):
            if categorical_column:
                column_unique_values = len(set(X_train[:, index]))
                column_max_index = int(max(X_train[:, index]))
                # categorical columns with only one unique value
                # do not need an embedding.
                if column_unique_values == 1:
                    continue
                categorical_columns.append(index)
                categorical_dimensions.append(column_max_index + 1)

        clf = TabNetClassifier(
            n_a=config['na'],
            n_d=config['na'],
            n_steps=config['nsteps'],
            gamma=config['gamma'],
            lambda_sparse=config['lambda_sparse'],
            momentum=config['mb'],
            cat_idxs=self.categorical_columns,
            cat_dims=self.categorical_dimensions,
            seed=self.param['seed'],
            optimizer_params={
                'lr': config['learning_rate'],
            },
            scheduler_params={
                'step_size': config['decay_iterations'],
                'gamma': config['decay_rate'],
            },
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
        )
        batch_size = config['batch_size']
        if batch_size == 32768:
            vbatch_size = config['vbatch_size1']
        elif batch_size == 16384:
            vbatch_size = config['vbatch_size2']
        elif batch_size == 8192:
            vbatch_size = config['vbatch_size3']
        elif batch_size == 4096:
            vbatch_size = config['vbatch_size4']
        elif batch_size == 2048:
            vbatch_size = config['vbatch_size5']
        elif batch_size == 1024:
            vbatch_size = config['vbatch_size6']
        elif batch_size == 512:
            vbatch_size = config['vbatch_size7']
        elif batch_size == 256:
            vbatch_size = config['vbatch_size8']
        else:
            raise ValueError('Illegal batch size given')

        clf.fit(
            X_train=X_train, y_train=y_train,
            batch_size=batch_size,
            virtual_batch_size=vbatch_size,
            eval_metric=['balanced_accuracy'],
            max_epochs=200,
        )

        y_train_preds = clf.predict(X_train)
        y_test_preds = clf.predict(X_test)

        train_performance = balanced_accuracy_score(y_train, y_train_preds)
        test_performance = balanced_accuracy_score(y_test, y_test_preds)

        if test_performance is None or test_performance is np.inf:
            test_performance = 0

        res = {
            'train_accuracy': float(train_performance),
            'test_accuracy': float(test_performance),
        }

        return res

    @staticmethod
    def get_default_configspace(
            seed: int = 11,
    ) -> cs.ConfigurationSpace:
        """Get the hyperparameter search space.

        The function provides the configuration space that is
        used to generate the algorithm specific hyperparameter
        search space.

        Parameters:
        -----------
        seed: int
            The seed used to build the configuration space.
        Returns:
        --------
        config_space: cs.ConfigurationSpace
            Configuration space for XGBoost.
        """
        config_space = cs.ConfigurationSpace(seed=seed)
        # learning rate
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'na',
                choices=[8, 16, 24, 32, 64, 128],
            )
        )
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'learning_rate',
                choices=[0.005, 0.01, 0.02, 0.025],
            )
        )
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'gamma',
                choices=[1.0, 1.2, 1.5, 2.0],
            )
        )
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'nsteps',
                choices=[3, 4, 5, 6, 7, 8, 9, 10],
            )
        )
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'lambda_sparse',
                choices=[0, 0.000001, 0.0001, 0.001, 0.01, 0.1],
            )
        )
        batch_size = cs.CategoricalHyperparameter(
            'batch_size',
            choices=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        )
        vbatch_size1 = cs.CategoricalHyperparameter(
            'vbatch_size1',
            choices=[256, 512, 1024, 2048, 4096],
        )
        vbatch_size2 = cs.CategoricalHyperparameter(
            'vbatch_size2',
            choices=[256, 512, 1024, 2048, 4096],
        )
        vbatch_size3 = cs.CategoricalHyperparameter(
            'vbatch_size3',
            choices=[256, 512, 1024, 2048, 4096],
        )
        vbatch_size4 = cs.CategoricalHyperparameter(
            'vbatch_size4',
            choices=[256, 512, 1024, 2048],
        )
        vbatch_size5 = cs.CategoricalHyperparameter(
            'vbatch_size5',
            choices=[256, 512, 1024],
        )
        vbatch_size6 = cs.CategoricalHyperparameter(
            'vbatch_size6',
            choices=[256, 512],
        )
        vbatch_size7 = cs.Constant(
            'vbatch_size7',
            256
        )
        vbatch_size8 = cs.Constant(
            'vbatch_size8',
            256
        )
        config_space.add_hyperparameter(
            batch_size
        )
        config_space.add_hyperparameters(
            [
                vbatch_size1,
                vbatch_size2,
                vbatch_size3,
                vbatch_size4,
                vbatch_size5,
                vbatch_size6,
                vbatch_size7,
                vbatch_size8,
            ]
        )
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'decay_rate',
                choices=[0.4, 0.8, 0.9, 0.95],
            )
        )
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'decay_iterations',
                choices=[500, 2000, 8000, 10000, 20000],
            )
        )
        config_space.add_hyperparameter(
            cs.CategoricalHyperparameter(
                'mb',
                choices=[0.6, 0.7, 0.8, 0.9, 0.95, 0.98],
            )
        )

        config_space.add_condition(
            cs.EqualsCondition(
                vbatch_size1,
                batch_size,
                32768,
            )
        )
        config_space.add_condition(
            cs.EqualsCondition(
                vbatch_size2,
                batch_size,
                16384,
            )
        )
        config_space.add_condition(
            cs.EqualsCondition(
                vbatch_size3,
                batch_size,
                8192,
            )
        )
        config_space.add_condition(
            cs.EqualsCondition(
                vbatch_size4,
                batch_size,
                4096,
            )
        )
        config_space.add_condition(
            cs.EqualsCondition(
                vbatch_size5,
                batch_size,
                2048,
            )
        )
        config_space.add_condition(
            cs.EqualsCondition(
                vbatch_size6,
                batch_size,
                1024,
            )
        )
        config_space.add_condition(
            cs.EqualsCondition(
                vbatch_size7,
                batch_size,
                512,
            )
        )
        config_space.add_condition(
            cs.EqualsCondition(
                vbatch_size8,
                batch_size,
                256,
            )
        )

        return config_space

    @staticmethod
    def get_parameters(
            seed: int = 11,
    ) -> Dict[str, Union[int, str]]:
        """Get the parameters of the method.

        Get a dictionary based on the arguments given to the
        function, which will be used to as the initial configuration
        for the algorithm.

        Parameters:
        -----------
        seed: int
            The seed that will be used for the model.

        Returns:
        --------
        param: dict
            A dictionary that will be used as a configuration for the
            algorithm.
        """
        param = {
            'seed': seed,
        }

        return param