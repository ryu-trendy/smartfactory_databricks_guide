# Databricks notebook source
# MAGIC %md
# MAGIC #4장 메달리온 아키텍처와 Delta Lake 기초 — 데이터 신뢰성의 토대

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Bronze 테이블 생성과 시뮬레이션 데이터

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Bronze 테이블 DDL
# MAGIC -- CREATE OR REPLACE TABLE 로 기존 테이블/스키마를 덮어씀
# MAGIC
# MAGIC CREATE OR REPLACE TABLE smartfactory.raw.sensor_events (
# MAGIC     event_id       STRING    NOT NULL COMMENT '이벤트 고유 ID ({equipment_id}_{YYYYMMDDHHmm})',
# MAGIC     equipment_id   STRING    NOT NULL COMMENT '설비 ID (EQ001~EQ050)',
# MAGIC     line_id        STRING    NOT NULL COMMENT '라인 ID (LINE01~LINE10)',
# MAGIC     equipment_type STRING    NOT NULL COMMENT '설비 유형 (CNC/AOI/ROBOT/PRESS/CONVEYOR)',
# MAGIC     event_time     TIMESTAMP NOT NULL COMMENT '센서 측정 시각',
# MAGIC     temperature_c DOUBLE             COMMENT '온도 (°C)',
# MAGIC     vibration_ms2 DOUBLE             COMMENT '진동 가속도 (m/s²)',
# MAGIC     pressure_bar  DOUBLE             COMMENT '압력 (bar)',
# MAGIC     rpm           DOUBLE             COMMENT '회전수 (RPM)',
# MAGIC     quality_score DOUBLE             COMMENT '품질 점수 (0~1)',
# MAGIC     ingested_at   TIMESTAMP          COMMENT '수집 시각',
# MAGIC     shift         STRING             COMMENT '교대 근무 (DAY/EVENING/NIGHT)',
# MAGIC     is_outlier    BOOLEAN            COMMENT '통계적 이상치 여부 (TRUE = 이상치, FALSE = 정상)'
# MAGIC )
# MAGIC CLUSTER BY (equipment_id, event_time)
# MAGIC TBLPROPERTIES (
# MAGIC     'quality' = 'bronze',
# MAGIC     delta.enableChangeDataFeed = true
# MAGIC )

# COMMAND ----------

from datetime import datetime, timedelta
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, BooleanType
import random


# 설비 유형별 정상/이상 범위 명시 분리 (정현우 책임연구원 ML 학습 스펙)
EQUIPMENT_CONFIG = {
    "CNC": {
        "ids":          [f"EQ{i:03d}" for i in range(1, 11)],
        "line":         "LINE01",
        "normal_temp":  (40, 80),    # 정상 온도 범위 (°C)
        "anomaly_temp": (80, 95),    # 고장 직전 온도 범위 (°C)
        "normal_vib":   (0.5, 3.0),  # 정상 진동 범위 (m/s²)
        "anomaly_vib":  (3.0, 7.0),  # 고장 직전 진동 범위 (m/s²)
        "pressure_range": (8, 22),
        "rpm_range":    (800, 3200),
    },
    "AOI": {
        "ids":          [f"EQ{i:03d}" for i in range(11, 21)],
        "line":         "LINE03",
        "normal_temp":  (25, 50),    # 정상 온도 범위 (°C)
        "anomaly_temp": (50, 55),    # 고장 직전 온도 범위 (°C)
        "normal_vib":   (0.2, 2.0),  # 정상 진동 범위 (m/s²)
        "anomaly_vib":  (2.0, 3.5),  # 고장 직전 진동 범위 (m/s²)
        "pressure_range": (3, 10),
        "rpm_range":    (0, 100),
    },
}


def generate_sensor_records(start_dt: datetime) -> list:
    records = []
    current_dt = start_dt
    end_dt     = start_dt + timedelta(days=30)
    anomaly_prob = 0.03  # 전체 3% 이상 패턴 → 불균형 클래스 반영


    while current_dt < end_dt:
        for eq_type, cfg in EQUIPMENT_CONFIG.items():
            for eq_id in cfg["ids"]:
                is_anomaly = random.random() < anomaly_prob


                if is_anomaly:
                    # 고장 직전: 온도·진동 동시 상승 (correlated anomaly)
                    temp    = round(random.uniform(*cfg["anomaly_temp"]), 2)
                    vib     = round(random.uniform(*cfg["anomaly_vib"]),  3)
                    quality = round(random.gauss(0.62, 0.06), 4)   # 품질 저하
                else:
                    # 정상: 정상 범위 중심 가우시안 (현실적인 센서 노이즈)
                    temp_c  = sum(cfg["normal_temp"]) / 2
                    vib_c   = sum(cfg["normal_vib"])  / 2
                    temp    = round(random.gauss(temp_c, 3.0), 2)
                    vib     = round(random.gauss(vib_c,  0.3), 3)
                    quality = round(random.gauss(0.96,   0.03), 4)


                pressure = round(random.uniform(*cfg["pressure_range"]), 2)
                rpm      = round(random.uniform(*cfg["rpm_range"]), 1)


                # 교대 근무: 00~07시 DAY, 08~15시 EVENING, 16~23시 NIGHT
                shift = "DAY" if current_dt.hour < 8 else ("EVENING" if current_dt.hour < 16 else "NIGHT")


                records.append((
                    f"{eq_id}_{current_dt.strftime('%Y%m%d%H%M')}",  # event_id
                    eq_id,
                    cfg["line"],
                    eq_type,                                           # equipment_type
                    current_dt,
                    # max(cfg["normal_temp"][0], min(temp, cfg["anomaly_temp"][1])),  # clamp
                    float(max(cfg["normal_temp"][0], min(temp, cfg["anomaly_temp"][1]))),  # clamp: int → float
                    max(0.0, vib),
                    max(0.0, pressure),
                    max(0.0, rpm),
                    min(1.0, max(0.0, quality)),
                    datetime.now(),
                    shift,            # 교대 근무 (DAY/EVENING/NIGHT)
                    is_anomaly,       # is_outlier: anomaly 패턴 감지 시 TRUE, 정상 시 FALSE
                ))
        current_dt += timedelta(minutes=10)  # ✓ 모든 설비 처리 후 시간 진행
    return records  # ✓ while 루프 종료 후 반환


# 데이터 생성 및 Bronze 테이블에 저장
# 예상: 20설비 × 4,320 time-step(10분 간격 × 30일) = 86,400건
start   = datetime.now() - timedelta(days=30)
records = generate_sensor_records(start)
print(f"생성된 레코드 수: {len(records):,}")


schema = StructType([
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
    StructField("shift",          StringType(),    True),
    StructField("is_outlier",     BooleanType(),   True),
])


df = spark.createDataFrame(records, schema)
df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("smartfactory.raw.sensor_events")


# 이후 활용할 경로에 저장
LANDING_PATH = "/Volumes/smartfactory/default/data/landing/sensor_events/"
df.coalesce(1).write.format("parquet").mode("overwrite").save(LANDING_PATH)
print("Bronze 테이블 저장 완료!")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 Silver 테이블 생성 — 데이터 정제

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Bronze에서 Silver로 변환 (정제 규칙 적용)
bronze_df = spark.table("smartfactory.raw.sensor_events")

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

silver_df.write.format("delta").mode("overwrite").saveAsTable("smartfactory.processed.sensor_clean")

print(f"Silver 저장 완료: {silver_df.count():,}건")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.4 Gold 테이블 생성 — 집계 분석

# COMMAND ----------

# DBTITLE 1,4.4 Gold 테이블 생성 — 집계 분석
# Silver에서 Gold로: 일별·라인·설비별 집계 테이블
silver_df = spark.table("smartfactory.processed.sensor_clean")

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

gold_df.write.format("delta").mode("overwrite").saveAsTable("smartfactory.analytics.equipment_daily")

print("Gold 집계 테이블 저장 완료!")