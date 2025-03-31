
from synapse.ml.predict import MLFlowTransformer

class ModelScorer:

    model_names=["diabete_lr_regression_model", "diabete_classification_model"]

    # Use the model to generate diabetes predictions for each row
    for model_name in model_names:
        print("Using {} for prediction".format(model_name))
        model = MLFlowTransformer(
            inputCols=["AGE","SEX","BMI","BP","S1","S2","S3","S4","S5","S6"],
            outputCol="predictions_"+ model_name,
            modelName=model_name,
            modelVersion=1)