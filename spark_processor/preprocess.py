from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lag, avg, dayofweek, month, lit, last
from pyspark.sql.window import Window
import sys

def create_spark_session():
    spark = SparkSession.builder.appName("StockPricePreprocess")\
        .config("spark.mongodb.input.uri", "mongodb+srv://bdat1008:bdat1008@cluster0.wzablab.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")\
        .config("spark.mongodb.input.database", "stockData")\
        .config("spark.mongodb.input.collection", "historicalStockData")\
        .config("spark.mongodb.input.readPreference.name", "primaryPreferred")\
        .config("spark.mongodb.input.uri.ssl", "true")\
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1")\
        .getOrCreate()
    return spark

def preprocess_data(spark, database, collection):
    df = spark.read.format("mongo").option("database", database).option("collection", collection).load()
    
    # Columns are now directly accessible, no need to extract them from "data"
    df = df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))\
           .withColumn("open", col("Open").cast("float"))\
           .withColumn("high", col("High").cast("float"))\
           .withColumn("low", col("Low").cast("float"))\
           .withColumn("close", col("Close").cast("float"))\
           .withColumn("volume", col("Volume").cast("float"))

    df = df.withColumn("day_of_week", dayofweek(col("date")))\
           .withColumn("month", month(col("date")))

    # Filling null values for numerical columns
    numerical_cols = ["open", "high", "low", "close", "volume", "day_of_week", "month"]
    df = df.fillna(0, subset=numerical_cols)
    df = df.na.drop()

    # Calculate moving averages
    for window_length in [5, 10, 20]:
        df = df.withColumn(f'{window_length}_day_moving_avg', avg("close").over(Window.partitionBy().orderBy("date").rowsBetween(-window_length + 1, 0)))

    # Create lag features for close prices
    for lag_days in [1, 2, 3, 5, 7, 14]:
        df = df.withColumn(f'lag_{lag_days}', lag("close", lag_days).over(Window.partitionBy().orderBy("date")))
    
    return df

if __name__ == "__main__":
    spark_session = create_spark_session()
    preprocessed_df = preprocess_data(spark_session, "stockData", "historicalStockData")
    preprocessed_df = preprocessed_df.drop("_id") # When saving csv it gives an error so I dropped the column
    #preprocessed_df.show()  # For demonstration; consider saving or further processing

    output_path = "/Users/iremkaymakcilar/BDAT1008-FinalProject/BDAT1008-FinalProject/spark_processor/data"
    preprocessed_df.coalesce(1).write.option("header", "true").csv(output_path)
