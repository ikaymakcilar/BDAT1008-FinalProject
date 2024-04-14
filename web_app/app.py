from datetime import datetime, timedelta
import random
from flask import Flask, render_template
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import to_date, col, lag, dayofweek, month, avg, lit, date_format, format_number
from pyspark.sql.window import Window
import json
from pyspark.sql import Row
from api import api_bp

app = Flask(__name__)

app.register_blueprint(api_bp, url_prefix='/api')

def create_spark_session():
    spark = SparkSession.builder \
        .appName("FlaskWithSpark") \
        .config("spark.mongodb.input.uri", "mongodb+srv://bdat1008:bdat1008@cluster0.wzablab.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0") \
        .config("spark.mongodb.input.database", "stockData") \
        .config("spark.mongodb.input.collection", "historicalStockData") \
        .config("spark.mongodb.output.uri", "mongodb+srv://bdat1008:bdat1008@cluster0.wzablab.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0") \
        .config("spark.mongodb.output.database", "stockData") \
        .config("spark.mongodb.output.collection", "historicalStockDataPredictions") \
        .config("spark.mongodb.output.replaceDocument", "true") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .getOrCreate()
    return spark

spark = create_spark_session()

model_path = "/Users/iremkaymakcilar/BDAT1008-FinalProject/BDAT1008-FinalProject/spark_processor/model"
model = PipelineModel.load(model_path)

def preprocess_data(df):
    windowSpec = Window.orderBy("date")
    
    df = df.withColumn("date", to_date(col("date"), "yyyy-MM-dd")) \
           .withColumn("open", col("open").cast("float")) \
           .withColumn("high", col("high").cast("float")) \
           .withColumn("low", col("low").cast("float")) \
           .withColumn("close", col("close").cast("float")) \
           .withColumn("volume", col("volume").cast("float")) \
           .withColumn("day_of_week", dayofweek(col("date"))) \
           .withColumn("month", month(col("date")))

    for window_length in [5, 10, 20]:
        df = df.withColumn(f"{window_length}_day_moving_avg", avg("close").over(windowSpec.rowsBetween(-window_length + 1, 0)))
    
    for lag_days in [1, 2, 3, 5, 7, 14]:
        df = df.withColumn(f"lag_{lag_days}", col("close").cast("float") - lag("close", lag_days).over(windowSpec))
    
    return df

def prepare_data_for_json(df):
    # Ensure all date columns are formatted as strings
    df = df.withColumn("date", date_format(col("date"), "yyyy-MM-dd"))
    return df

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictions')
def predictions():
    df = spark.read.format("mongo").load()
    df = df.withColumnRenamed("Open", "open") \
           .withColumnRenamed("High", "high") \
           .withColumnRenamed("Low", "low") \
           .withColumnRenamed("Close", "close") \
           .withColumnRenamed("Volume", "volume") \
           .withColumn("date", to_date(col("date"), "yyyy-MM-dd"))

    df = df.filter(col("date") > lit("2023-12-31"))
    df_preprocessed = preprocess_data(df)
    predictions = model.transform(df_preprocessed)
    predictions = predictions.withColumn("predicted_close", col("prediction")) \
                            .withColumn("predicted_close", format_number(col("predicted_close"), 3)) \
                            .withColumn("close", format_number(col("close"), 3)) \
                            .withColumn("date", date_format(col("date"), "yyyy-MM-dd"))

    full_data_to_save = df_preprocessed.join(predictions.select("date", "predicted_close"), "date")
    full_data_to_save = prepare_data_for_json(full_data_to_save)  # Make sure all dates are strings

    # Collect data and prepare for JSON
    data_list = [row.asDict() for row in full_data_to_save.collect()]

    # Save to JSON
    with open('data.json', 'w') as f:
        json.dump(data_list, f)

    # Prepare display format for predictions
    predictions_list = [{"date": x["date"], "predicted_close": x["predicted_close"], "real_close": x["close"]} for x in predictions.collect()]

    with open('predictions.json', 'w') as f:
        json.dump(predictions_list, f)

    return render_template('predictions.html', predictions=predictions_list)

@app.route('/dashboard')
def dashboard():
     # Read the predictions from the JSON file
    try:
        with open('predictions.json', 'r') as f:
            predictions_list = json.load(f)
    except FileNotFoundError:
        predictions_list = []  # If the file does not exist, use an empty list

    return render_template('dashboard.html', predictions=predictions_list)

@app.route('/api-documentation')
def documentation():
        return render_template('documentation.html')

@app.route('/forecasting')
def forecasting():
    try:
        with open('data.json', 'r') as f:
            historical_data = json.load(f)

        latest_data = historical_data[-1]
        latest_date = datetime.strptime(latest_data['date'], '%Y-%m-%d')

        # Perform forecasting
        forecast_results = perform_forecasting(latest_date, latest_data, 10)

        # Save forecast results to forecasts.json
        with open('forecasts.json', 'w') as f_out:
            json.dump(forecast_results, f_out)

        # Extract historical dates and closes for plotting
        historical_dates = [item['date'] for item in historical_data]
        historical_closes = [item['close'] for item in historical_data]
        forecast_dates = [result['date'] for result in forecast_results]
        forecast_closes = [result['predicted_close'] for result in forecast_results]

    except Exception as e:
        return render_template('forecasting.html', error=f"Failed to load data: {str(e)}")

    return render_template('forecasting.html', historical_dates=historical_dates, historical_closes=historical_closes, forecast_dates=forecast_dates, forecast_closes=forecast_closes, forecasts=forecast_results)

def perform_forecasting(latest_date, last_data, days):
    results = []
    current_close = float(last_data['close'])
    volatility_factor = 0.02

    for i in range(1, days + 1):
        future_date = latest_date + timedelta(days=i)
        random_change = 1 + random.uniform(-volatility_factor, volatility_factor)
        current_close *= random_change

        future_data = {
            'date': future_date.strftime('%Y-%m-%d'),
            'open': current_close,
            'high': current_close * (1 + volatility_factor),
            'low': current_close * (1 - volatility_factor),
            'close': current_close,
            'volume': last_data['volume'],
            'day_of_week': future_date.weekday() + 1,
            'month': future_date.month,
            '5_day_moving_avg': current_close,
            '10_day_moving_avg': current_close,
            '20_day_moving_avg': current_close,
            'lag_1': current_close,
            'lag_2': current_close,
            'lag_3': current_close,
            'lag_5': current_close,
            'lag_7': current_close,
            'lag_14': current_close
        }

        future_df = spark.createDataFrame([Row(**future_data)])
        prediction = model.transform(future_df)
        predicted_close = prediction.select("prediction").first()[0]

        results.append({
            'date': future_date.strftime('%Y-%m-%d'),
            'predicted_close': round(predicted_close, 3)
        })

    return results

if __name__ == '__main__':
    app.run(port=8000, debug=True)
