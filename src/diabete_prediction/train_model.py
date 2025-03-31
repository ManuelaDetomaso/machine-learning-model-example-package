import mlflow
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from pyspark.sql import DataFrame

from diabete_prediction.config_loader import load_config
from diabete_prediction.utils import create_mlflow_schema_from_typed_list


class ModelTrainer:
    def __init__(self):
        self.config = load_config()
        self.risk_colname = self.config["InputData"]["risk_colname"]
        self.target_colname = self.config["InputData"]["target_colname"]
        self.feature_schema = self.config["InputData"]["feature_schema"]
        self.feature_names = [feat["name"] for feat in self.feature_schema]
        self.test_size = self.config["ModelTraining"]["test_size"]
        self.experiment_name = self.config["ModelGeneral"]["test_size"]

    def split_train_test_data(
        self, df_sp_prepared: DataFrame, target_type="numerical"
    ) -> tuple:
        """Split prepared training data into the feature set and the target variable

        Args:
            df_sp_prepared (DataFrame): _description_
            target_type (str, optional): Target variable type: either numerical or categorical. Defaults to "numerical".

        Raises:
            ValueError: Indicate a target type, either numerical or categorical

        Returns:
            tuple: X_train (training feature set), X_test(test feature set), y_train (training tartegt), y_test (test target)
        """
        from sklearn.model_selection import train_test_split

        # convert spark prepared data into a Pandas DataFrame
        df_prepared = df_sp_prepared.toPandas()
        # select feature set (X) and target (y) based on the decision to create
        # either a regression or a classification model
        if target_type == "numerical":
            X, y = (
                df_prepared[self.feature_names].values,
                df_prepared[self.target_colname].values,
            )
        elif target_type == "categorical":
            X, y = (
                df_prepared[self.feature_names].values,
                df_prepared[self.risk_colname].values,
            )
        else:
            raise ValueError("Indicate a target type, either numerical or categorical")

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=0
        )
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        return X_train, X_test, y_train, y_test

    def mlflow_training(
        self,
        experiment_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = "regression",
        with_signature: bool = True,
        logs: bool = False,
        save_model=True,
    ):
        """Train an ML model

        Args:
            experiment_name (str): name of the mlflow experiment
            X_train (pd.DataFrame): feature training set
            y_train (pd.Series): target variable
            model (object): scikit-leran model class
            model_type (str, optional): "regression" or "classification". Defaults to "regression".
            with_signature (bool, optional): aave Input Data names and data types in the mlflow experiment. Defaults to True.
            logs (bool, optional): save mlflow model logs. Defaults to False.

        Raises:
            ValueError: A model type can be either regression or classification_binary or classification_multi
        """
        from sklearn.linear_model import LinearRegression, LogisticRegression

        if model_type == "regression":
            output_schema = Schema([ColSpec("integer")])
            full_experiment_name = self.experiment_name + "-regression"
            model = LinearRegression()
            print(model)
        elif model_type == "categorical_binary":
            output_schema = Schema([ColSpec("binary")])
            full_experiment_name = self.experiment_name + "-classification"
            model = LogisticRegression(C=1 / 0.1, solver="liblinear")
            print(model)
        else:
            raise ValueError(
                "A model type can be either regression or classification_binary"
            )
        mlflow.set_experiment(full_experiment_name)

        with mlflow.start_run():
            # Use autologging for all other parameters and metrics
            mlflow.autolog(log_models=logs)

            # When you fit the model, all other information will be logged
            model.fit(X_train, y_train)
            print("Model training completed.")

            if with_signature:
                print("Applying Signature...")
                # Create the signature manually
                input_schema = create_mlflow_schema_from_typed_list(self.feature_schema)

                # Create the signature object
                signature = ModelSignature(inputs=input_schema, outputs=output_schema)

                # Manually log the model
                mlflow.sklearn.log_model(model, "model", signature=signature)

            if save_model:
                self.save_mlflow_model(full_experiment_name)

    def save_mlflow_model(self, experiment_name:str):
        """Save and register an mlflow model

        Args:
            full_experiment_name (str): full name of the mlflow experiment (full_experiment_name)
        """
        # Get the experiment by name
        model_name = experiment_name + "-model"
        exp = mlflow.get_experiment_by_name(experiment_name)

        # List the last experiment run
        last_run = mlflow.search_runs(
            exp.experiment_id, order_by=["start_time DESC"], max_results=1
        )

        # Retrieve the run ID of the last experiment run
        last_run_id = last_run.iloc[0]["run_id"]

        # Register the model that was trained in that run
        print("Registering the model from run :", last_run_id)
        # Create a path to the model output folder of the last experiment run
        model_uri = "runs:/{}/model".format(last_run_id)

        # Register or save the model by specifying the model folder and model name
        mv = mlflow.register_model(model_uri, model_name)
        print("Name: {}".format(mv.name))
        print("Version: {}".format(mv.version))
