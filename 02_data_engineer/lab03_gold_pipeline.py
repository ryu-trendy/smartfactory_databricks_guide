# Databricks notebook source
from pyspark.sql import functions as F


# Silver에서 Gold로: 일별·라인·설비별 집계 테이블
silver_df = spark.table("smartfactory.processed.sensor_clean_job")


gold_df = silver_df.groupBy(
    F.to_date("event_time").alias("oee_date"),
    "line_id",
    "equipment_id",
).agg(
    F.avg("temperature_c").alias("avg_temp"),
    F.max("temperature_c").alias("max_temp"),
    F.avg("vibration_ms2").alias("avg_vib"),
    F.max("vibration_ms2").alias("max_vib"),
    F.avg("quality_score").alias("avg_quality"),
    F.sum(F.when(F.col("is_anomaly"), 1).otherwise(0)).alias("anomaly_count"),
    F.count("*").alias("record_count"),
)


gold_df.write.format("delta").mode("overwrite")     .saveAsTable("smartfactory.analytics.oee_daily_job")


print("Gold 집계 테이블 저장 완료!")