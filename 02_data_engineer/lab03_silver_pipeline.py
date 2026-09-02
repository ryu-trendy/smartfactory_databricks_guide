# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# Bronze에서 Silver로 변환 (정제 규칙 적용)
bronze_df = spark.table("smartfactory.raw.sensor_events_job")


silver_df = bronze_df.filter(
    # 1. 물리적으로 불가능한 값 제거
    (F.col("temperature_c").between(10, 150)) &
    (F.col("vibration_ms2") >= 0)             &
    (F.col("pressure_bar")   >= 0)             &
    (F.col("rpm")           >= 0)             &
    (F.col("quality_score").between(0, 1))
).withColumn(
    # 2. 설비별 Z-score 계산 (통계적 이상치 탐지)
    "temp_zscore",
    (F.col("temperature_c") - F.avg("temperature_c").over(
        Window.partitionBy("equipment_id")
    )) / F.stddev("temperature_c").over(
        Window.partitionBy("equipment_id")
    )
).withColumn(
    # 3. |Z-score| > 3 이면 통계적 이상치로 분류
    "is_anomaly",
    F.abs(F.col("temp_zscore")) > 3
).withColumn(
    "_processed_time", F.current_timestamp()
)


silver_df.write.format("delta").mode("overwrite").saveAsTable("smartfactory.processed.sensor_clean_job")


print(f"Silver 저장 완료: {silver_df.count():,}건")