from pyspark import pipelines as dp
from pyspark.sql.functions import col, current_timestamp, avg, max, count


# Bronze: Auto Loader로 원시 데이터 수집
@dp.table(
    name="smartfactory.raw.sensor_events_raw",
    comment="Bronze: 설비 센서 원시 데이터",
    table_properties={"delta.enableChangeDataFeed": "true", "quality": "bronze"}
)
def sensor_events_raw():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .load("/Volumes/smartfactory/default/data/landing/sensor_events/")
        .withColumn("_ingestion_time", current_timestamp())
    )


# Silver: 데이터 품질 검증 + 정제
@dp.expect("유효한_장비ID", "equipment_id RLIKE '^EQ[0-9]{3}' AND LENGTH(equipment_id) = 5")
@dp.expect_or_drop("유효한_온도",  "temperature_c BETWEEN 0 AND 150")
@dp.expect_or_drop("유효한_진동",  "vibration_ms2 BETWEEN 0 AND 20")
@dp.expect_or_drop("널_아닌_시간", "event_time IS NOT NULL")
@dp.table(
    name="smartfactory.processed.sensor_events_clean",
    comment="Silver: 정제된 센서 데이터",
    table_properties={"quality": "silver"}
)
def sensor_events_clean():
    return (
        spark.readStream.table("sensor_events_raw")
        .filter(col("equipment_id").isNotNull())
        .withColumn("event_date", col("event_time").cast("date"))
    )


# Gold: 집계 테이블
@dp.materialized_view(name="smartfactory.analytics.oee_hourly", comment="Gold: 시간별 OEE 집계",
                      table_properties={"quality": "gold"})
def oee_hourly():
    return (
        spark.read.table("smartfactory.processed.sensor_events_clean")
        .groupBy("equipment_id", "line_id", "event_date")
        .agg(
            avg("temperature_c").alias("avg_temperature_c"),
            max("vibration_ms2").alias("max_vibration_ms2"),
            count("*").alias("event_count")
        )
    )
