# Databricks notebook source
from pyspark.sql.functions import current_timestamp, col
from pyspark.sql.types import *


LANDING_PATH    = "/Volumes/smartfactory/default/data/landing/sensor_events/"
CHECKPOINT_PATH = "/Volumes/smartfactory/default/data/checkpoint/sensor_event/"


sensor_schema = StructType([
    StructField("event_id",       StringType(),    False), 
    StructField("equipment_id",   StringType(),    False),
    StructField("line_id",        StringType(),    True), 
    StructField("equipment_type", StringType(),    False),
    StructField("event_time",     TimestampType(), False), 
    StructField("temperature_c",  DoubleType(),    True),
    StructField("vibration_ms2",  DoubleType(),    True),
    StructField("pressure_bar",   DoubleType(),    True),
    StructField("rpm",            DoubleType(),    True),
    StructField("quality_score",  DoubleType(),    True),
    StructField("ingested_at",    TimestampType(), True), 
])


sensor_stream = (
    spark.readStream
    .format("cloudFiles")                           # Auto Loader 포맷
    .option("cloudFiles.format", "parquet")            # 소스 파일 형식
    .option("cloudFiles.schemaLocation", CHECKPOINT_PATH + "schemalocation/")
    .option("cloudFiles.maxFilesPerTrigger", 1000)  # 트리거당 최대 파일 수
    .schema(sensor_schema)
    .load(LANDING_PATH)
    .withColumn("_ingestion_time", current_timestamp())
    .withColumn("_source_file",    col("_metadata.file_path"))
)


query = (
    sensor_stream.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH + "main/")
    .trigger(availableNow=True)   # 배치 모드: 현재 파일 모두 처리 후 종료


    .toTable("smartfactory.raw.sensor_events_job")
)
query.awaitTermination()