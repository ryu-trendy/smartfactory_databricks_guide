# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# Silver 정제 데이터 읽기
df_silver = spark.read.table("smartfactory.processed.sensor_clean_job")


# 설비별 시간순 윈도우 (최근 60개 이벤트 = 단기, 360개 = 장기)
window_short = (Window.partitionBy("equipment_id")
                .orderBy("event_time")
                .rowsBetween(-60, 0))


window_long = (Window.partitionBy("equipment_id")
               .orderBy("event_time")
               .rowsBetween(-360, 0))


# 피처 계산
df_features = (
    df_silver
    # 온도 통계 피처
    .withColumn("temp_rolling_avg", F.avg("temperature_c").over(window_short))
    .withColumn("temp_rolling_std", F.stddev("temperature_c").over(window_short))
    .withColumn("temp_long_avg", F.avg("temperature_c").over(window_long))
    # 진동 통계 피처
    .withColumn("vibration_rolling_avg", F.avg("vibration_ms2").over(window_short))
    .withColumn("vibration_rolling_max", F.max("vibration_ms2").over(window_long))
    # 파생 피처
    .withColumn("temp_vibration_ratio",
                F.col("temperature_c") / F.greatest(F.col("vibration_ms2"), F.lit(0.01)))
    .withColumn("temp_deviation",
                F.col("temperature_c") - F.col("temp_long_avg"))
    # 시간 피처
    .withColumn("event_hour", F.hour("event_time"))
    .withColumn("is_night_shift",
                F.when((F.hour("event_time") >= 22) | (F.hour("event_time") <= 6), 1).otherwise(0))
)


# Feature table 저장
(df_features.write
 .mode("overwrite")
 .saveAsTable("smartfactory.analytics.sensor_features"))


print("✅ Feature table 저장 완료: smartfactory.analytics.sensor_features")
display(df_features.limit(5))