from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel

def create_spark_session():
    return SparkSession.builder.appName("StockPricePrediction").getOrCreate()

def prepare_data(spark, filepath):
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    # Ensure the Date column is excluded from the features and treated as metadata if needed
    return df

def build_and_evaluate_model(df):
    # Define the feature columns, ensure 'close' is not included in features
    feature_cols = [col for col in df.columns if col not in ['date', 'symbol', 'close']]
    
    # Assemble features into a single vector column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    
    # Split the data into training and test sets based on date
    # Assuming 'date' column is present and used to split the data
    train_data = df.filter(col("date") <= "2023-12-31")
    test_data = df.filter(col("date") > "2023-12-31")
    
    # Initialize the GBTRegressor
    gbt = GBTRegressor(featuresCol="features", labelCol="close")  # Use 'close' as the label column
    
    # Define the pipeline
    pipeline = Pipeline(stages=[assembler, gbt])
    
    # Fit the model
    model = pipeline.fit(train_data)
    
    # Make predictions
    predictions = model.transform(test_data)
    
    # Evaluate the model using RMSE and MAE
    evaluator_rmse = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="rmse")
    rmse = evaluator_rmse.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    evaluator_mae = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="mae")
    mae = evaluator_mae.evaluate(predictions)
    print(f"Mean Absolute Error (MAE): {mae}")
    
    # R2
    evaluator_r2 = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="r2")
    r2 = evaluator_r2.evaluate(predictions)
    print(f"R-squared (R2): {r2}")

    model_path = "/Users/iremkaymakcilar/BDAT1008-FinalProject/spark_processor/model"
    model.write().overwrite().save(model_path)
    print(f"Model saved successfully at {model_path}")

    return model, rmse, mae, r2


if __name__ == "__main__":
    spark_session = create_spark_session()
    filepath = "/Users/iremkaymakcilar/BDAT1008-FinalProject/spark_processor/data/part-00000-c04ffe34-90a5-47d1-b322-661854b3a230-c000.csv"
    df = prepare_data(spark_session, filepath)
    model, rmse, mae, r2 = build_and_evaluate_model(df)


