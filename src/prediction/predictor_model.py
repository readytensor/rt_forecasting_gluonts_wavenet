import os
import warnings
import joblib
import torch
import numpy as np
import pandas as pd
from typing import Optional
from gluonts.torch.model.wavenet import WaveNetEstimator
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError
from gluonts.dataset.common import ListDataset
from pytorch_lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping


warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Forecaster:
    """A wrapper class for the WaveNet Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "WaveNet Forecaster"
    made_up_frequency = "S"  # by seconds
    made_up_start_dt = "2000-01-01 00:00:00"

    def __init__(
        self,
        data_schema: ForecastingSchema,
        history_forecast_ratio: int = None,
        num_bins: int = 1024,
        num_residual_channels: int = 24,
        num_skip_channels: int = 32,
        dilation_depth: Optional[int] = None,
        num_stacks: int = 1,
        lr: float = 0.001,
        weight_decay: float = 1e-08,
        use_log_scale_feature: bool = True,
        temperature: float = 1.0,
        batch_size: int = 32,
        early_stopping: bool = True,
        early_stopping_patience: int = 20,
        min_delta: float = 0.01,
        trainer_kwargs: dict = {},
        use_exogenous: bool = True,
        random_state: int = 0,
    ):
        """Construct a new WaveNet Forecaster

        Args:

            data_schema (ForecastingSchema): Schema of the data used for training.

            history_forecast_ratio (int):
                Sets the history length depending on the forecast horizon.
                For example, if the forecast horizon is 20 and the history_forecast_ratio is 10,
                history length will be 20*10 = 200 samples.

            num_residual_channels (int): Number of residual channels.

            num_skip_channels (int): Number of skip channels.

            dilation_depth (Optional[int]): The depth of the dilated convolution.

            num_stacks (int): The number of dilation stacks.

            lr (float): Learning rate.

            weight_decay (float): Weight decay regularization parameter (default: 1e-8).

            temperature (float): Temparature used for sampling from the output softmax distribution, by default 1.0

            batch_size (int) The size of the batches to be used for training (default: 32).

            early_stopping (bool): If true, use early stopping.

            early_stopping_patience (int): Patience used by early stopper.

            min_delta (float): Minimum imporovement required by early stopper.

            use_exogenous (bool): If true, uses covariates in training.

            random_state (int): Sets the underlying random seed at model initialization time.
        """

        self.data_schema = data_schema
        self.num_bins = num_bins
        self.num_residual_channels = num_residual_channels
        self.num_skip_channels = num_skip_channels
        self.dilation_depth = dilation_depth
        self.num_stacks = num_stacks
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_log_scale_feature = use_log_scale_feature
        self.temperature = temperature
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.trainer_kwargs = trainer_kwargs
        self.random_state = random_state
        self.use_exogenous = use_exogenous and data_schema.future_covariates
        self._is_trained = False
        self.freq = self.map_frequency(data_schema.frequency)
        self.history_length = None
        self.gluonts_dataset = None

        if history_forecast_ratio:
            self.history_length = (
                self.data_schema.forecast_length * history_forecast_ratio
            )

        early_stopping = EarlyStopping(
            monitor="train_loss",
            patience=self.early_stopping_patience,
            min_delta=self.min_delta,
            verbose=True,
            mode="min",
        )

        if self.early_stopping:
            self.trainer_kwargs["callbacks"] = [early_stopping]

        if torch.cuda.is_available():
            print("GPU is available")
        else:
            print("GPU is not available")
            if self.trainer_kwargs.get("accelerator") == "gpu":
                self.trainer_kwargs.pop("accelerator")

        if not self.use_exogenous:
            num_covariates = 0
        else:
            num_covariates = len(data_schema.future_covariates)

        self.model = WaveNetEstimator(
            prediction_length=data_schema.forecast_length,
            num_bins=self.num_bins,
            num_residual_channels=self.num_residual_channels,
            num_skip_channels=self.num_skip_channels,
            dilation_depth=self.dilation_depth,
            num_stacks=self.num_stacks,
            lr=self.lr,
            weight_decay=self.weight_decay,
            use_log_scale_feature=self.use_log_scale_feature,
            temperature=self.temperature,
            batch_size=self.batch_size,
            freq=self.freq,
            num_feat_dynamic_real=num_covariates,
            trainer_kwargs=self.trainer_kwargs,
        )

    def prepare_time_column(
        self, data: pd.DataFrame, is_train: bool = True
    ) -> pd.DataFrame:
        """
        Adds time column of type DATETIME to datasets that have time column dtype as INT.

        Args:
            data (pd.DataFrame): The input dataset.
            is_train (bool): Set to true for training dataset and false for testing dataset.

            Returns (pd.DataFrame): The dataset after processing time column.
        """
        # sort data
        time_col_dtype = self.data_schema.time_col_dtype
        id_col = self.data_schema.id_col
        time_col = self.data_schema.time_col

        data = data.sort_values(by=[id_col, time_col])

        if time_col_dtype == "INT":
            # Find the number of rows for each location (assuming all locations have
            # the same number of rows)
            series_val_counts = data[id_col].value_counts()
            series_len = series_val_counts.iloc[0]
            num_series = series_val_counts.shape[0]

            if is_train:
                # since GluonTS requires a date column, we will make up a timeline
                start_date = pd.Timestamp(self.made_up_start_dt)
                datetimes = pd.date_range(
                    start=start_date, periods=series_len, freq=self.made_up_frequency
                )
                self.last_timestamp = datetimes[-1]
                self.timedelta = datetimes[-1] - datetimes[-2]

            else:
                start_date = self.last_timestamp + self.timedelta
                datetimes = pd.date_range(
                    start=start_date, periods=series_len, freq=self.made_up_frequency
                )
            int_vals = sorted(data[time_col].unique().tolist())
            self.time_to_int_map = dict(zip(datetimes, int_vals))
            # Repeat the datetime range for each location
            data[time_col] = list(datetimes) * num_series
        else:
            data[time_col] = pd.to_datetime(data[time_col])
            data[time_col] = data[time_col].dt.tz_localize(None)

        return data

    def prepare_training_data(
        self,
        history: pd.DataFrame,
    ) -> ListDataset:
        """
        Applys the history_forecast_ratio parameter and puts the training data into the shape expected by GluonTS.

        Args:
            history (pd.DataFrame): The input dataset.

        Returns (ListDataset): The processed dataset expected by GluonTS.
        """
        data_schema = self.data_schema
        # Make sure there is a date column
        history = self.prepare_time_column(data=history, is_train=True)

        # Manage each series in the training data separately
        all_covariates = []
        groups_by_ids = history.groupby(data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
            for id_ in all_ids
        ]

        # Enforces the history_forecast_ratio parameter
        if self.history_length:
            new_length = []
            for series in all_series:
                series = series.iloc[-self.history_length :]
                new_length.append(series.copy())
            all_series = new_length

        cov_names = []

        if self.use_exogenous:
            cov_names = data_schema.future_covariates

        # Put future covariates into separate list
        all_covariates = []

        for series in all_series:
            series_covariates = []

            for covariate in cov_names:
                series_covariates.append(series[covariate])

            all_covariates.append(series_covariates)

        # If future covariates are available for training, create a dataset with future covariate features,
        # otherwise a dataset with only target series will be created.
        if cov_names and self.use_exogenous:
            list_dataset = [
                {
                    "start": series[data_schema.time_col].iloc[0],
                    "target": series[data_schema.target],
                    "feat_dynamic_real": series_covariates,
                }
                for series, series_covariates in zip(all_series, all_covariates)
            ]
        else:
            list_dataset = [
                {
                    "start": series[data_schema.time_col].iloc[0],
                    "target": series[data_schema.target],
                }
                for series in all_series
            ]

        gluonts_dataset = ListDataset(list_dataset, freq=self.freq)

        self.training_all_series = all_series
        self.training_future_covariates = all_covariates
        self.all_ids = all_ids

        return gluonts_dataset

    def prepare_test_data(self, test_data: pd.DataFrame) -> ListDataset:
        """
        Puts the testing data into the shape expected by GluonTS.

         Args:
             test_data (pd.DataFrame): The input dataset.

         Returns (ListDataset): The processed dataset expected by GluonTS.
        """
        data_schema = self.data_schema
        test_data = self.prepare_time_column(data=test_data, is_train=False)
        groups_by_ids = test_data.groupby(self.data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
            for id_ in all_ids
        ]

        cov_names = []
        if self.use_exogenous:
            cov_names = data_schema.future_covariates

        all_covariates = []
        for series in all_series:
            series_covariates = []

            for covariate in cov_names:
                series_covariates.append(series[covariate])

            if series_covariates:
                all_covariates.append(series_covariates)

        all_concatenated_covariates = []
        for covariates, training_covariates in zip(
            all_covariates, self.training_future_covariates
        ):
            concatenated_covariates = [
                pd.concat([i, j]) for i, j in zip(training_covariates, covariates)
            ]
            all_concatenated_covariates.append(concatenated_covariates)

        if data_schema.future_covariates and self.use_exogenous:
            list_dataset = [
                {
                    "start": series[data_schema.time_col].iloc[0],
                    "target": series[data_schema.target],
                    "feat_dynamic_real": series_covariates,
                }
                for series, series_covariates in zip(
                    self.training_all_series, all_concatenated_covariates
                )
            ]
        else:
            list_dataset = [
                {
                    "start": series[data_schema.time_col].iloc[0],
                    "target": series[data_schema.target],
                }
                for series in self.training_all_series
            ]

        gluonts_dataset = ListDataset(list_dataset, freq=self.freq)

        return gluonts_dataset

    def map_frequency(self, frequency: str) -> str:
        """
        Maps the frequency in the data schema to the frequency expected by GluonTS.

        Args:
            frequency (str): The frequency from the schema.

        Returns (str): The mapped frequency.
        """
        frequency = frequency.lower()
        frequency = frequency.split("frequency.")[1]
        if frequency == "yearly":
            return "Y"
        if frequency == "quarterly":
            return "Q"
        if frequency == "monthly":
            return "M"
        if frequency == "weekly":
            return "W"
        if frequency == "daily":
            return "D"
        if frequency == "hourly":
            return "H"
        if frequency == "minutely":
            return "min"
        if frequency in ["secondly", "other"]:
            return "S"

    def fit(
        self,
        history: pd.DataFrame,
    ) -> None:
        """Fit the Forecaster to the training data.
        A separate WaveNet model is fit to each series that is contained
        in the data.

        Args:
            history (pandas.DataFrame): The features of the training data.
        """
        np.random.seed(self.random_state)
        seed_everything(self.random_state)

        history = self.prepare_training_data(history=history)
        self.predictor = self.model.train(history)
        self._is_trained = True

    def predict(
        self, test_data: pd.DataFrame, prediction_col_name: str
    ) -> pd.DataFrame:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        seed_everything(self.random_state)
        test_dataset = self.prepare_test_data(test_data=test_data)
        predictions = self.predictor.predict(test_dataset)
        predictions_df = test_data.copy()

        values = []
        for forecast in predictions:
            mean = list(forecast.mean)
            values += mean

        predictions_df[prediction_col_name] = values

        return predictions_df

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """

    model = Forecaster(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(history=history)
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
