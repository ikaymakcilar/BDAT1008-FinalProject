from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, date_format
import yfinance as yf
import schedule
import time
from datetime import datetime, timedelta

def create_spark_session():
    return SparkSession.builder.appName("StockPricePreprocess")\
        .config("spark.mongodb.input.uri", "mongodb+srv://bdat1008:bdat1008@cluster0.wzablab.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")\
        .config("spark.mongodb.output.uri", "mongodb+srv://bdat1008:bdat1008@cluster0.wzablab.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")\
        .config("spark.mongodb.input.database", "stockData")\
        .config("spark.mongodb.input.collection", "historicalStockData")\
        .config("spark.mongodb.output.database", "stockData")\
        .config("spark.mongodb.output.collection", "historicalStockData")\
        .config("spark.mongodb.input.readPreference.name", "primaryPreferred")\
        .config("spark.mongodb.input.uri.ssl", "true")\
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1")\
        .getOrCreate()

def fetch_stock_data(symbol, start_date=None):
    if start_date:
        start_date_str = start_date.strftime('%Y-%m-%d')
        data = yf.download(symbol, start=start_date_str)
    else:
        data = yf.download(symbol, period="max")
    data.reset_index(inplace=True)
    return data

def update_stock_data(spark, symbol):
    existing_data_df = spark.read.format("mongo").load()

    # Check if the collection is empty
    if existing_data_df.rdd.isEmpty():
        print("Collection is empty. Fetching all available data for symbol.")
        start_date = None
    else:
        # Fetch the latest date from the existing data in MongoDB
        latest_date_row = existing_data_df.select("date").orderBy("date", ascending=False).limit(1).collect()
        if latest_date_row:
            latest_date_str = latest_date_row[0]["date"]
            latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
            start_date = latest_date + timedelta(days=1)

    # Ensure not fetching data for future dates
    today = datetime.now()
    if start_date and start_date.date() > today.date():
        print("Latest data is up-to-date. No new data to fetch.")
        return

    stock_data_df = fetch_stock_data(symbol, start_date)

    if not stock_data_df.empty:
        spark_df = spark.createDataFrame(stock_data_df)
        spark_df = spark_df.withColumn("symbol", lit(symbol))\
                           .withColumn("date", date_format("Date", "yyyy-MM-dd"))\
                           .select("symbol", "date", "Open", "High", "Low", "Close", "Volume")

        spark_df.write.format("mongo").mode("append").option("database", "stockData").option("collection", "historicalStockData").save()

        print(f"Inserted {spark_df.count()} rows for {symbol} into MongoDB.")
    else:
        print("No new data available to insert.")

def scheduled_job():
    spark = create_spark_session()
    SYMBOL = "BX"
    update_stock_data(spark, SYMBOL)
    spark.stop()

if __name__ == "__main__":
    schedule.every().day.at("08:43").do(scheduled_job)
    while True:
        schedule.run_pending()
        time.sleep(10)
