# Databricks notebook source
# MAGIC %md
# MAGIC # 15장: Feature Store 설계 — Training-Serving Skew를 없애다

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15.1 Feature Store 개념

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering xgboost --quiet
# MAGIC %restart_python

# COMMAND ----------

# smartfactory.ml.equipment_features_raw 초기 생성
from pyspark.sql.functions import avg, stddev, max, col, when, count, lit
from pyspark.sql.window import Window


def create_time_series_features(df):
    """센서 데이터에 시계열 롤링 피처 19개를 추가한다."""
    w_eq  = Window.partitionBy("equipment_id").orderBy("event_time")
    w_1h  = w_eq.rowsBetween(-359, 0)   # 1시간(360개 × 10초)
    w_4h  = w_eq.rowsBetween(-1439, 0)  # 4시간
    w_24h = w_eq.rowsBetween(-8639, 0)  # 24시간


    return df.withColumns({
        # 온도(Temperature) 관련 피처
        "temp_avg_1h":    avg("temperature_c").over(w_1h),
        "temp_max_1h":    max("temperature_c").over(w_1h),
        "temp_std_1h":    stddev("temperature_c").over(w_1h),
        "temp_avg_4h":    avg("temperature_c").over(w_4h),
        "temp_trend_4h":  (avg("temperature_c").over(w_1h) - avg("temperature_c").over(w_4h)),


        # 진동(Vibration) 관련 피처
        "vib_avg_1h":     avg("vibration_ms2").over(w_1h),
        "vib_max_1h":     max("vibration_ms2").over(w_1h),
        "vib_std_1h":     stddev("vibration_ms2").over(w_1h),
        "vib_avg_4h":     avg("vibration_ms2").over(w_4h),
        "vib_avg_24h":    avg("vibration_ms2").over(w_24h),
        "vib_trend_4h":   (avg("vibration_ms2").over(w_1h) - avg("vibration_ms2").over(w_4h)),


        # 압력(Pressure) 관련 피처
        "pressure_avg_1h": avg("pressure_bar").over(w_1h),
        "pressure_max_1h": max("pressure_bar").over(w_1h),


        # RPM 관련 피처
        "rpm_avg_1h":     avg("rpm").over(w_1h),
        "rpm_std_1h":     stddev("rpm").over(w_1h),


        # 품질(Quality) 피처
        "quality_avg_1h": avg("quality_score").over(w_1h),


        # 이상치 및 종합 스코어
        "anomaly_count_1h": count(
            when((col("temperature_c") > 90) | (col("vibration_ms2") > 6.0), 1)
        ).over(w_1h),
        "anomaly_count_4h": count(
            when((col("temperature_c") > 90) | (col("vibration_ms2") > 6.0), 1)
        ).over(w_4h),


        "composite_anomaly_score": (
            0.4 * (col("temperature_c") - 50) / 50 +
            0.4 * col("vibration_ms2") / 10 +
            0.2 * (col("pressure_bar") - 10) / 30
        ),
    })




# 전체 sensor_clean 데이터로 초기 피처 테이블 생성
# sensor_clean 원본에 완전 중복 행(3회 반복 적재)이 있으므로 PK 기준 중복 제거 후 피처 계산
sensor_df = (
    spark.table("smartfactory.processed.sensor_clean")
    .dropDuplicates(["equipment_id", "event_time"])
)
feature_df = create_time_series_features(sensor_df)


(
    feature_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("smartfactory.ml.equipment_features_raw")
)
print(f"equipment_features_raw(staging) 생성 완료: {feature_df.count():,} rows, {len(feature_df.columns)} columns")
display(feature_df.limit(3))

# COMMAND ----------

# pdm_labels: 설비별 48시간 내 고장 예측 레이블
# (equipment_features는 Cell 5에서 fe.create_table()로 생성)
import pyspark.sql.functions as F
from pyspark.sql.window import Window
features_raw = spark.table("smartfactory.ml.equipment_features_raw")


# ── pdm_labels ─────────────────────────────────────────────────────────
# 48시간 앞방 윈도 내 composite_anomaly_score 업계치(0.15) 초과 여부
# 레이블: failure_within_48h = 1(고장 예상) / 0(정상)
w_fwd = (
    Window.partitionBy("equipment_id")
          .orderBy(F.col("event_time").cast("long"))
          .rangeBetween(0, 48 * 3600)          # 현재 ~ 48시간 후
)


pdm_labels_df = (
    features_raw
    .withColumn(
        "failure_within_48h",
        (F.max("composite_anomaly_score").over(w_fwd) > 0.15).cast("int")
    )
    .select("equipment_id", "event_time", "failure_within_48h")
)
(
    pdm_labels_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("smartfactory.ml.pdm_labels")
)
print(f"pdm_labels 생성: {pdm_labels_df.count():,}줄")
print(f"고장 레이블(1) 비율: "
      f"{pdm_labels_df.filter('failure_within_48h=1').count() / pdm_labels_df.count():.1%}")
display(pdm_labels_df.groupBy("failure_within_48h").count().orderBy("failure_within_48h"))

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from pyspark.sql.functions import col


fe = FeatureEngineeringClient()



feature_df = spark.table("smartfactory.ml.equipment_features_raw")
fe.create_table(
    name="smartfactory.ml.equipment_features",
    primary_keys=["equipment_id", "event_time"],
    timeseries_column="event_time",        # Point-in-Time Join 활성화
    df=feature_df,
    description="스마트팩토리 예지 정비 피처 스토어 (19개 피처)",
)


# Feature Table 업데이트 (최근 24h 배치를 merge)
new_features = spark.table("smartfactory.ml.equipment_features")
fe.write_table(
    name="smartfactory.ml.equipment_features",
    df=new_features,
    mode="merge",   # upsert
)


# 학습 셋 생성 (Point-in-Time Join)
label_df = spark.table("smartfactory.ml.pdm_labels")


training_set = fe.create_training_set(
    df=label_df,
    feature_lookups=[
        FeatureLookup(
            table_name="smartfactory.ml.equipment_features",
            lookup_key=["equipment_id"],
            timestamp_lookup_key="event_time",   # 레이블 시점 이전 피처만
            feature_names=[
                "temp_avg_1h", "temp_trend_4h", "vib_avg_1h",
                "vib_trend_4h", "anomaly_count_1h", "composite_anomaly_score",
            ],
        )
    ],
    label="failure_within_48h",
)


training_df = training_set.load_df()
print(f"학습 셋 크기: ({training_df.count()}, {len(training_df.columns)})")
display(training_df.filter(col('composite_anomaly_score').isNotNull()).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # 16장: 피처 엔지니어링 — 센서 데이터에서 예측력을 끌어내다

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16.1 21개 피처 설계

# COMMAND ----------

from pyspark.sql.functions import avg, stddev, max, col, when, count, lit
from pyspark.sql.window import Window


def create_time_series_features(df):
    w_eq  = Window.partitionBy("equipment_id").orderBy("event_time")
    w_1h  = w_eq.rowsBetween(-359, 0)   # 1시간(360개 × 10초)
    w_4h  = w_eq.rowsBetween(-1439, 0)  # 4시간
    w_24h = w_eq.rowsBetween(-8639, 0)  # 24시간
    
    return df.withColumns({
        # 온도(Temperature) 관련 피처
        "temp_avg_1h":    avg("temperature_c").over(w_1h),
        "temp_max_1h":    max("temperature_c").over(w_1h),
        "temp_std_1h":    stddev("temperature_c").over(w_1h),
        "temp_avg_4h":    avg("temperature_c").over(w_4h),
        "temp_trend_4h":  (avg("temperature_c").over(w_1h) - avg("temperature_c").over(w_4h)),


        # 진동(Vibration) 관련 피처
        "vib_avg_1h":     avg("vibration_ms2").over(w_1h),
        "vib_max_1h":     max("vibration_ms2").over(w_1h),
        "vib_std_1h":     stddev("vibration_ms2").over(w_1h),
        "vib_avg_4h":     avg("vibration_ms2").over(w_4h),
        "vib_avg_24h":    avg("vibration_ms2").over(w_24h),
        "vib_trend_4h":   (avg("vibration_ms2").over(w_1h) - avg("vibration_ms2").over(w_4h)),


        # 압력(Pressure) 관련 피처
        "pressure_avg_1h": avg("pressure_bar").over(w_1h),
        "pressure_max_1h": max("pressure_bar").over(w_1h),


        # RPM 관련 피처
        "rpm_avg_1h":     avg("rpm").over(w_1h),
        "rpm_std_1h":     stddev("rpm").over(w_1h),


        # 품질(Quality) 피처
        "quality_avg_1h": avg("quality_score").over(w_1h),


        # 이상치 및 종합 스코어
        "anomaly_count_1h": count(
            when((col("temperature_c") > 90) | (col("vibration_ms2") > 6.0), 1)
        ).over(w_1h),
        "anomaly_count_4h": count(
            when((col("temperature_c") > 90) | (col("vibration_ms2") > 6.0), 1)
        ).over(w_4h),


        "composite_anomaly_score": (
            0.4 * (col("temperature_c") - 50) / 50 +
            0.4 * col("vibration_ms2") / 10 +
            0.2 * (col("pressure_bar") - 10) / 30
        ),
    })


# 1. 데이터 로드
sensor_df  = spark.table("smartfactory.processed.sensor_clean")


# 2. 시계열 피처 생성 함수 적용
feature_df = create_time_series_features(sensor_df)


# 3. Feature Store 테이블 업데이트 (Merge)
fe.write_table(
    name="smartfactory.ml.equipment_features",
    df=feature_df,
    mode="merge",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16.2 시간 기반 학습/검증/테스트 분할

# COMMAND ----------

from pyspark.sql.functions import col, count, lit, sum as _sum
from pyspark.sql import Window


training_df = training_set.load_df()


# 1. 시간 순 정렬 (미래 누수 방지를 위한 시간 기반 분할)
training_df = training_df.orderBy("event_time")


# 2. 전체 건수 기반 60/20/20 분할 (row_number 활용)
from pyspark.sql.functions import row_number, ceil as _ceil


w_all = Window.orderBy("event_time")
training_df = training_df.withColumn("_row_num", row_number().over(w_all))


n = training_df.count()
train_end = int(n * 0.60)
val_end   = int(n * 0.80)


train_df = training_df.filter(col("_row_num") <= train_end).drop("_row_num")
val_df   = training_df.filter((col("_row_num") > train_end) & (col("_row_num") <= val_end)).drop("_row_num")
test_df  = training_df.filter(col("_row_num") > val_end).drop("_row_num")


print(f"학습:   {train_df.count():,}건")
print(f"검증:   {val_df.count():,}건")
print(f"테스트: {test_df.count():,}건")


# 3. 클래스 불균형 보정 — scale_pos_weight = 음성/양성
train_counts = train_df.groupBy("failure_within_48h").count().collect()
counts_map = {row["failure_within_48h"]: row["count"] for row in train_counts}
neg_count = counts_map.get(0, 0)
pos_count = counts_map.get(1, 0)


scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
pos_rate = pos_count / (neg_count + pos_count)


print(f"\n양성 비율: {pos_rate:.2%}")
print(f"scale_pos_weight (음성/양성): {scale_pos_weight:.2f}")
print(f"\n→ XGBoost 학습 시 scale_pos_weight={scale_pos_weight:.2f} 설정으로 소수 클래스 보정")
