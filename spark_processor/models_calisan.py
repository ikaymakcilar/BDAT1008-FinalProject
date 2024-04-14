from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

def create_spark_session():
    spark = SparkSession.builder.appName("StockPricePrediction").getOrCreate()
    return spark

def prepare_data(spark, filepath):
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    # Converting the date column from string to date type
    df = df.withColumn("Date", col("Date").cast("date"))
    return df

def build_and_evaluate_model(df):
    # Define the feature columns
    feature_cols = df.columns[2:-1]  # Excluding 'Date' and 'symbol' for features, 'Close' as the label
    
    # Assemble features into a single vector column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    
    # Split the data into training and test sets based on date
    train_data = df.filter(col("Date") <= "2023-12-31")
    test_data = df.filter(col("Date") > "2023-12-31")
    
    # Initialize the RandomForestRegressor
    rf = RandomForestRegressor(featuresCol="features", labelCol="Close")  # 'Close' is the label column
    
    # Define the pipeline
    pipeline = Pipeline(stages=[assembler, rf])
    
    # Fit the model
    model = pipeline.fit(train_data)
    
    # Make predictions
    predictions = model.transform(test_data)
   
    # RMSE
    evaluator_rmse = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="rmse")
    rmse = evaluator_rmse.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # MAE
    evaluator_mae = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="mae")
    mae = evaluator_mae.evaluate(predictions)
    print(f"Mean Absolute Error (MAE): {mae}")

    # R2
    evaluator_r2 = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="r2")
    r2 = evaluator_r2.evaluate(predictions)
    print(f"R-squared (R2): {r2}")
    
    return model, rmse, mae, r2

if __name__ == "__main__":
    spark_session = create_spark_session()
    filepath = "/Users/iremkaymakcilar/BDAT1008-FinalProject/BDAT1008-FinalProject/spark_processor/data/part-00000-9d63c967-f2db-4b07-9505-15216ab9b0bf-c000.csv"
    df = prepare_data(spark_session, filepath)
    model, rmse, mae, r2 = build_and_evaluate_model(df)
