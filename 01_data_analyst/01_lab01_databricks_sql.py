# Databricks notebook source
# 5~6장: Databricks SQL + Delta Lake 고급 기능
# DBR 18 LTS 

# COMMAND ----------

# MAGIC %md
# MAGIC # 5~6장: Databricks SQL과 Delta Lake 고급 기능
# MAGIC
# MAGIC ## 실습 목표
# MAGIC - Databricks SQL로 Bronze → Silver 변환 (메달리온 아키텍처)
# MAGIC - Delta Lake 최적화: OPTIMIZE, ZORDER, Liquid Clustering
# MAGIC - Change Data Feed (CDF) 활성화 및 증분 처리
# MAGIC - ACID 트랜잭션: 동시 쓰기 충돌 방지

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.2 OPTIMIZE와 ZORDER — 쿼리 성능 최적화

# COMMAND ----------

# MAGIC %sql
# MAGIC -- OPTIMIZE: 소파일 병합으로 읽기 성능 향상
# MAGIC -- Lakeflow Jobs에서 매일 새벽 2시에 실행하는 것이 일반적입니다
# MAGIC OPTIMIZE smartfactory.processed.sensor_clean;
# MAGIC
# MAGIC -- OPTIMIZE + ZORDER: equipment_id, event_time 기준 데이터 물리적 정렬
# MAGIC -- WHERE equipment_id='EQ001' 같은 필터 쿼리 성능이 크게 향상됩니다
# MAGIC OPTIMIZE smartfactory.processed.sensor_clean
# MAGIC     ZORDER BY (equipment_id, event_time);
# MAGIC
# MAGIC -- 최적화 결과 확인 (DESCRIBE HISTORY)
# MAGIC DESCRIBE HISTORY smartfactory.processed.sensor_clean;
# MAGIC -- numFilesAdded, numFilesRemoved, numBytesAdded 등 통계 확인 가능

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.3 Time Travel — 과거 데이터 조회와 복원

# COMMAND ----------

# MAGIC %sql
# MAGIC -- CDF 기능 활성화
# MAGIC ALTER TABLE smartfactory.processed.sensor_clean 
# MAGIC SET TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 적재 전 레코드 수 확인
# MAGIC SELECT COUNT(*) FROM smartfactory.processed.sensor_clean;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 실수로 데이터 삭제
# MAGIC DELETE FROM smartfactory.processed.sensor_clean
# MAGIC WHERE quality_score > 0.95;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 실수 후 레코드 수 확인
# MAGIC SELECT COUNT(*) FROM smartfactory.processed.sensor_clean;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 버전 히스토리 확인
# MAGIC DESCRIBE HISTORY smartfactory.processed.sensor_clean;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 방법 1: 타임스탬프 기준으로 과거 데이터 조회 (DESCRIBE HISTORY 의 version 1)
# MAGIC SELECT COUNT(*) AS record_count
# MAGIC FROM smartfactory.processed.sensor_clean
# MAGIC TIMESTAMP AS OF '2026-08-31 04:48:01';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 방법 2: 버전 번호로 과거 데이터 조회 (DESCRIBE HISTORY 의 version 1)
# MAGIC SELECT * FROM smartfactory.processed.sensor_clean
# MAGIC VERSION AS OF 1
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 실수로 데이터가 삭제됐을 때 이전 버전으로 즉시 복원
# MAGIC RESTORE TABLE smartfactory.processed.sensor_clean
# MAGIC TO VERSION AS OF 2;
# MAGIC -- 또는 시각 기반: TO TIMESTAMP AS OF '2026-08-03 02:17:58'

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 버전 0의 데이터와 현재 데이터 비교
# MAGIC SELECT 'current' AS version, COUNT(*) AS cnt
# MAGIC FROM smartfactory.processed.sensor_clean
# MAGIC UNION ALL
# MAGIC SELECT 'version_0', COUNT(*)
# MAGIC FROM smartfactory.processed.sensor_clean VERSION AS OF 3; 

# COMMAND ----------

# PySpark에서 Time Travel
old_df = spark.read.format("delta")\
    .option("versionAsOf", 1)\
    .table("smartfactory.processed.sensor_clean")
old_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.4 MERGE INTO — Delta Lake의 핵심 기능

# COMMAND ----------

# MAGIC %sql
# MAGIC -- MERGE INTO: Upsert 패턴 (있으면 UPDATE, 없으면 INSERT)
# MAGIC -- 새로운 센서 데이터를 Silver 테이블에 멱등하게 병합
# MAGIC MERGE INTO smartfactory.processed.sensor_clean AS target
# MAGIC USING (
# MAGIC     -- 소스: 마지막으로 처리한 이후 수집된 원시 데이터
# MAGIC     SELECT equipment_id, line_id, event_time,
# MAGIC            temperature_c, vibration_ms2, pressure_bar, rpm, quality_score
# MAGIC     FROM   smartfactory.raw.sensor_events
# MAGIC     WHERE  event_time > (
# MAGIC         SELECT COALESCE(MAX(event_time), '2000-01-01')
# MAGIC         FROM smartfactory.processed.sensor_clean
# MAGIC     )
# MAGIC ) AS source
# MAGIC ON  target.equipment_id = source.equipment_id   -- 매칭 키 1: 설비 ID
# MAGIC AND target.event_time   = source.event_time     -- 매칭 키 2: 측정 시각
# MAGIC WHEN MATCHED THEN
# MAGIC     -- 같은 (설비, 시각) 조합이 이미 있으면 업데이트 (MES 정정 데이터 반영)
# MAGIC     UPDATE SET
# MAGIC         target.temperature_c = source.temperature_c,
# MAGIC         target.vibration_ms2 = source.vibration_ms2,
# MAGIC         target.pressure_bar   = source.pressure_bar
# MAGIC WHEN NOT MATCHED THEN
# MAGIC     -- 새로운 측정값이면 삽입
# MAGIC     INSERT (equipment_id, line_id, event_time, temperature_c,
# MAGIC             vibration_ms2, pressure_bar, rpm, quality_score)
# MAGIC     VALUES (source.equipment_id, source.line_id, source.event_time,
# MAGIC             source.temperature_c, source.vibration_ms2,
# MAGIC             source.pressure_bar, source.rpm, source.quality_score);
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.5 실습 Lab — 생산 KPI 대시보드 쿼리

# COMMAND ----------

import datetime
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType, DoubleType, BooleanType
import random


today = datetime.date.today()
random.seed(42)
EQUIPMENT_MASTER = []
eq_num = 1
for eq_type, line_range, count in [
    ("CNC",      range(1, 5),  15),
    ("AOI",      range(1, 5),  10),
    ("ROBOT",    range(5, 9),  10),
    ("PRESS",    range(5, 9),  10),
    ("CONVEYOR", range(1, 11),  5),
]:
    for _ in range(count):
        EQUIPMENT_MASTER.append({
            "equipment_id":   f"EQ{eq_num:03d}",
            "equipment_type": eq_type,
            "line_id":        f"LINE{random.choice(list(line_range)):02d}",
            "install_date":   "2021-01-01",
            "rated_capacity": random.randint(80, 120),
        })
        eq_num += 1


print(f"설비 수: {len(EQUIPMENT_MASTER)}")




def generate_production_logs(eq_list, days=30):
    records = []
    for eq in eq_list:
        for d in range(days):
            target_date = today - datetime.timedelta(days=30 - d)
            for shift, start_h in {"DAY": 8, "EVENING": 16, "NIGHT": 0}.items():
                planned_qty = random.randint(400, 600)
                downtime_min = random.randint(0, 60)
                actual_qty  = int(planned_qty * (1 - downtime_min / 480) * random.uniform(0.9, 1.0))
                defect_qty  = int(actual_qty * random.uniform(0, 0.05))


                records.append(Row(
                    log_id           = f"{eq['equipment_id']}_{target_date}_{shift}",
                    equipment_id     = eq["equipment_id"],
                    line_id          = eq["line_id"],
                    shift            = shift,
                    shift_start      = datetime.datetime.combine(target_date, datetime.time(start_h, 0)),
                    planned_qty      = planned_qty,
                    actual_qty       = actual_qty,
                    defect_qty       = defect_qty,
                    downtime_min     = downtime_min,
                    production_date  = target_date.isoformat(),
                ))
    return records


prod_records = generate_production_logs(EQUIPMENT_MASTER)
prod_schema = StructType([
    StructField("log_id",          StringType(),    False),
    StructField("equipment_id",    StringType(),    False),
    StructField("line_id",         StringType(),    True),
    StructField("shift",           StringType(),    True),
    StructField("shift_start",     TimestampType(), True),
    StructField("planned_qty",     IntegerType(),   True),
    StructField("actual_qty",      IntegerType(),   True),
    StructField("defect_qty",      IntegerType(),   True),
    StructField("downtime_min",    IntegerType(),   True),
    StructField("production_date", StringType(),    True),
])


# 1. 생산 로그 생성 : smartfactory.processed.production_logs 
df_prod = spark.createDataFrame(prod_records, prod_schema)
(df_prod.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("smartfactory.processed.production_logs"))
print("production_logs 저장 완료")


FAILURE_TYPES = ["BEARING_WEAR", "MOTOR_OVERHEAT", "SENSOR_DRIFT", "VIBRATION_EXCESS", "PRESSURE_DROP"]


def generate_maintenance_records(eq_list, days=30):
    records = []
    for eq in eq_list:
        for d in range(days):
            target_date = today - datetime.timedelta(days=30 - d)
            if random.random() < 0.05:
                failure = random.choice(FAILURE_TYPES)
                records.append(Row(
                    maintenance_id    = f"MNT_{eq['equipment_id']}_{target_date}_{d}",
                    equipment_id      = eq["equipment_id"],
                    line_id           = eq["line_id"],
                    maintenance_date  = target_date.isoformat(),
                    failure_type      = failure,
                    repair_duration_h = round(random.uniform(0.5, 8.0), 1),
                    technician_id     = f"TECH{random.randint(1, 10):02d}",
                    parts_replaced    = random.choice(["BEARING", "MOTOR", "SENSOR", "BELT", None]),
                    cost_usd          = round(random.uniform(100, 5000), 2),
                    failure_label     = 1,
                ))
    return records


maint_records = generate_maintenance_records(EQUIPMENT_MASTER)
maint_schema = StructType([
    StructField("maintenance_id",    StringType(), False),
    StructField("equipment_id",      StringType(), False),
    StructField("line_id",           StringType(), True),
    StructField("maintenance_date",  StringType(), True),
    StructField("failure_type",      StringType(), True),
    StructField("repair_duration_h", DoubleType(), True),
    StructField("technician_id",     StringType(), True),
    StructField("parts_replaced",    StringType(), True),
    StructField("cost_usd",          DoubleType(), True),
    StructField("failure_label",     IntegerType(),True),
])


# 2. 정비 이력 생성 : smartfactory.processed.maintenance_records
df_maint = spark.createDataFrame(maint_records, maint_schema)
(df_maint.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("smartfactory.processed.maintenance_records"))
print(f"maintenance_records 저장 완료 ({len(maint_records)}건)")


def generate_quality_inspections(eq_list, days=30):
    records = []
    for eq in eq_list:
        for d in range(days):
            target_date = today - datetime.timedelta(days=30 - d)
            for k in range(random.randint(3, 10)):
                passed = random.random() > 0.04
                records.append(Row(
                    inspection_id   = f"QI_{eq['equipment_id']}_{target_date}_{k}",
                    equipment_id    = eq["equipment_id"],
                    line_id         = eq["line_id"],
                    inspection_time = datetime.datetime.combine(
                        target_date,
                        datetime.time(random.randint(0, 23), random.randint(0, 59))
                    ),
                    defect_type     = None if passed else random.choice(["SCRATCH", "DIMENSION", "SURFACE", "COLOR"]),
                    severity        = None if passed else random.choice(["LOW", "MEDIUM", "HIGH"]),
                    passed          = passed,
                    inspector_id    = f"QI{random.randint(1, 5):02d}",
                ))
    return records


qi_records = generate_quality_inspections(EQUIPMENT_MASTER)
qi_schema = StructType([
    StructField("inspection_id",   StringType(),    False),
    StructField("equipment_id",    StringType(),    False),
    StructField("line_id",         StringType(),    True),
    StructField("inspection_time", TimestampType(), True),
    StructField("defect_type",     StringType(),    True),
    StructField("severity",        StringType(),    True),
    StructField("passed",          BooleanType(),   True),
    StructField("inspector_id",    StringType(),    True),
])


df_qi = spark.createDataFrame(qi_records, qi_schema)


# 3. 품질 검사 이력 생성 : smartfactory.processed.quality_inspections
(df_qi.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("smartfactory.processed.quality_inspections"))
print("quality_inspections 저장 완료")


eq_schema = StructType([
    StructField("equipment_id",   StringType(),  False),
    StructField("equipment_type", StringType(),  True),
    StructField("line_id",        StringType(),  True),
    StructField("install_date",   StringType(),  True),
    StructField("rated_capacity", IntegerType(), True),
])


df_eq = spark.createDataFrame([Row(**eq) for eq in EQUIPMENT_MASTER], eq_schema)


# 4. 설비 정보 생성 : smartfactory.processed.equipment_master
(df_eq.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("smartfactory.processed.equipment_master"))
print("equipment_master 저장 완료")


# 5. 스마트팩토리 코리아 silver 레이어 생성 완료 확인
tables = ["production_logs", "maintenance_records", "quality_inspections", "equipment_master"]
print("=" * 55)
print("스마트팩토리 코리아 Silver 레이어 생성 완료")
print("=" * 55)
for t in tables:
    cnt = spark.sql(f"SELECT COUNT(*) AS cnt FROM smartfactory.processed.{t}").collect()[0]["cnt"]
    print(f"  smartfactory.processed.{t}: {cnt:,}건")


# COMMAND ----------

# MAGIC %sql
# MAGIC -- 생산 KPI 대시보드 쿼리 (완전 버전)
# MAGIC USE CATALOG smartfactory;
# MAGIC USE SCHEMA analytics;
# MAGIC
# MAGIC
# MAGIC -- Step 1: 설비별 일간 OEE 계산
# MAGIC WITH availability AS (
# MAGIC   SELECT
# MAGIC     equipment_id,
# MAGIC     production_date,
# MAGIC     SUM(480 - downtime_min) AS op_min,
# MAGIC     SUM(480) AS plan_min,
# MAGIC     SUM(480 - downtime_min) * 1.0 / NULLIF(SUM(480), 0) AS availability
# MAGIC   FROM smartfactory.processed.production_logs
# MAGIC   WHERE production_date >= CAST(DATEADD(DAY, -30, CURRENT_DATE()) AS STRING)
# MAGIC   GROUP BY equipment_id, production_date
# MAGIC ),
# MAGIC performance AS (
# MAGIC   SELECT
# MAGIC     equipment_id,
# MAGIC     production_date,
# MAGIC     SUM(actual_qty) * 1.0 / NULLIF(SUM(planned_qty), 0) AS performance
# MAGIC   FROM smartfactory.processed.production_logs
# MAGIC   WHERE production_date >= CAST(DATEADD(DAY, -30, CURRENT_DATE()) AS STRING)
# MAGIC   GROUP BY equipment_id, production_date
# MAGIC ),
# MAGIC quality AS (
# MAGIC   SELECT
# MAGIC     equipment_id,
# MAGIC     CAST(DATE(inspection_time) AS STRING) AS production_date,
# MAGIC     SUM(CASE WHEN passed = true THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS quality
# MAGIC   FROM smartfactory.processed.quality_inspections
# MAGIC   WHERE DATE(inspection_time) >= DATEADD(DAY, -30, CURRENT_DATE())
# MAGIC   GROUP BY equipment_id, CAST(DATE(inspection_time) AS STRING)
# MAGIC )
# MAGIC SELECT
# MAGIC   a.equipment_id,
# MAGIC   a.production_date,
# MAGIC   ROUND(a.availability * 100, 1) AS availability_pct,
# MAGIC   ROUND(p.performance * 100, 1) AS performance_pct,
# MAGIC   ROUND(q.quality * 100, 1) AS quality_pct,
# MAGIC   ROUND(a.availability * p.performance * q.quality * 100, 1) AS oee_pct,
# MAGIC   CASE
# MAGIC     WHEN a.availability * p.performance * q.quality >= 0.85 THEN '목표 달성'
# MAGIC     WHEN a.availability * p.performance * q.quality >= 0.75 THEN '개선 필요'
# MAGIC     ELSE '긴급 대응'
# MAGIC   END AS status
# MAGIC FROM availability a
# MAGIC JOIN performance p ON a.equipment_id = p.equipment_id AND a.production_date = p.production_date
# MAGIC JOIN quality q ON a.equipment_id = q.equipment_id AND a.production_date = q.production_date
# MAGIC ORDER BY a.production_date DESC, oee_pct;
# MAGIC
# MAGIC -- Step 2: OPTIMIZE + Liquid Clustering (신규 테이블 권장)
# MAGIC -- 신규 테이블: CLUSTER BY (권장)
# MAGIC -- CREATE TABLE ... CLUSTER BY (equipment_id, production_date)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # 6장 | Delta Lake 심화 — Change Data Feed와 ACID 트랜잭션

# COMMAND ----------

# DBTITLE 1,트러블슈팅 가이드
# MAGIC %md
# MAGIC ## 트러블슈팅 가이드
# MAGIC
# MAGIC * `table_changes()`가 실패하면 먼저 `DESCRIBE HISTORY`로 실제 version 범위를 확인합니다.
# MAGIC * CDF 조회에서 `startingVersion = 0`은 항상 가능한 값이 아닙니다. 초기 스냅샷은 change event가 아닐 수 있습니다.
# MAGIC * `MERGE` 예제는 소스와 타깃 스키마를 맞춰야 하며, 없는 컬럼을 `INSERT` 목록에 넣으면 실패합니다.
# MAGIC * 스키마 진화 예제는 실제 신규 컬럼을 추가해야 효과가 보입니다.
# MAGIC * 교육용 재실행 시 과거 고정 timestamp보다 `DESCRIBE HISTORY` 결과를 우선 확인하세요.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.2.1 CDF 활성화와 조회

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 버전 히스토리 확인
# MAGIC DESCRIBE HISTORY smartfactory.raw.sensor_events;

# COMMAND ----------

# DBTITLE 1,CDF 활성화
# MAGIC %sql
# MAGIC -- sensor_events 테이블에 Legacy CDF 활성화
# MAGIC -- 이후 발생하는 INSERT/UPDATE/DELETE/MERGE 변경이 자동으로 기록됩니다
# MAGIC ALTER TABLE smartfactory.raw.sensor_events
# MAGIC SET TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# DBTITLE 1,실수로 데이터 삭제
# MAGIC %sql
# MAGIC -- 실수로 데이터 삭제 (CDF가 이 DELETE를 기록합니다)
# MAGIC DELETE FROM smartfactory.raw.sensor_events
# MAGIC WHERE quality_score > 0.95;

# COMMAND ----------

# DBTITLE 1,데이터 복구
# MAGIC %sql
# MAGIC -- 데이터 복구: DELETE 직전 버전으로 복원 (CDF가 RESTORE도 기록합니다)
# MAGIC -- ⚠️ 버전 번호는 Cell 22의 DESCRIBE HISTORY 결과를 확인하세요
# MAGIC RESTORE TABLE smartfactory.raw.sensor_events
# MAGIC TO VERSION AS OF 4;
# MAGIC -- v4 = ALTER TABLE (CDF 활성화) 시점 → DELETE 이전 상태

# COMMAND ----------

# DBTITLE 1,CDF 변경 이력 조회
# MAGIC %sql
# MAGIC -- CDF 변경 이력 조회: DELETE와 RESTORE 이벤트가 모두 기록됩니다
# MAGIC -- ⚠️ startingVersion은 CDF 활성화 직후 버전 (DELETE 발생 버전)
# MAGIC SELECT
# MAGIC     _change_type,   -- insert / update_preimage / update_postimage / delete
# MAGIC     _commit_version,
# MAGIC     _commit_timestamp,
# MAGIC     equipment_id,
# MAGIC     temperature_c,
# MAGIC     quality_score
# MAGIC FROM table_changes('smartfactory.raw.sensor_events', 5)  -- v5: DELETE 버전
# MAGIC ORDER BY _commit_version, _change_type
# MAGIC LIMIT 20;

# COMMAND ----------

# DBTITLE 1,CDF 변경 유형별 요약
# MAGIC %sql
# MAGIC -- 변경 유형별 요약: DELETE 건수와 RESTORE(insert) 건수를 비교합니다
# MAGIC SELECT
# MAGIC     _change_type,
# MAGIC     _commit_version,
# MAGIC     COUNT(*) AS row_count
# MAGIC FROM table_changes('smartfactory.raw.sensor_events', 5)
# MAGIC GROUP BY _change_type, _commit_version
# MAGIC ORDER BY _commit_version, _change_type;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.2.2 CDF로 증분 Silver 파이프라인 구현

# COMMAND ----------

# DBTITLE 1,CDF 증분 Silver 파이프라인
from pyspark.sql.functions import col, current_timestamp

spark.sql("""
    CREATE TABLE IF NOT EXISTS smartfactory.processed.pipeline_checkpoints (
        pipeline_name STRING,
        last_processed_version LONG,
        updated_at TIMESTAMP
    ) USING DELTA
    COMMENT '증분 파이프라인 체크포인트 관리 테이블'
""")


def incremental_silver_pipeline():
    # 1. CDF 활성화 확인 — 미설정 시 자동 활성화
    props = {r["key"]: r["value"] for r in spark.sql(
        "SHOW TBLPROPERTIES smartfactory.raw.sensor_events"
    ).collect()}
    if props.get("delta.enableChangeDataFeed") != "true":
        spark.sql("""
            ALTER TABLE smartfactory.raw.sensor_events
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        print("⚠️ CDF가 미설정이어서 활성화했습니다.")

    # 2. CDF 활성화 버전 찾기
    hist_df = spark.sql("DESCRIBE HISTORY smartfactory.raw.sensor_events")
    latest_version = hist_df.agg({"version": "max"}).first()[0]
    cdf_row = hist_df.filter("operation = 'SET TBLPROPERTIES'") \
                     .orderBy("version").first()
    min_cdf_version = cdf_row["version"] if cdf_row else latest_version

    # 3. 체크포인트에서 마지막 처리 버전 읽기
    ckpt_row = (spark.table("smartfactory.processed.pipeline_checkpoints")
                .filter("pipeline_name = 'bronze_to_silver'").first())
    saved = ckpt_row["last_processed_version"] if ckpt_row else 0
    last_version = max(saved, min_cdf_version)

    print(f"마지막 처리 버전: {last_version}  "
          f"(저장: {saved}, CDF 활성화: v{min_cdf_version}, 최신: v{latest_version})")

    if last_version >= latest_version:
        print("처리할 변경사항 없음 (이미 최신 버전)")
        return

    # 4. CDF로 INSERT 변경분만 읽기
    changes_df = (spark.read.format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", last_version + 1)
        .table("smartfactory.raw.sensor_events")
        .filter(col("_change_type") == "insert")
    )

    change_count = changes_df.count()
    if change_count == 0:
        print("처리할 INSERT 변경사항 없음")
        return
    print(f"처리 대상: {change_count:,}건")

    # 5. Silver 변환: 이상치 제거 + CDF 메타 컬럼 제거
    silver_df = (changes_df
        .filter(col("temperature_c").between(0, 150))
        .filter(col("vibration_ms2").between(0, 20))
        .drop("_change_type", "_commit_version", "_commit_timestamp")
    )
    print(f"정제 후: {silver_df.count():,}건")

    # 6. Silver 테이블에 append
    silver_df.write.format("delta") \
        .mode("append") \
        .saveAsTable("smartfactory.processed.sensor_clean")

    # 7. 체크포인트 업데이트
    new_version = changes_df.agg({"_commit_version": "max"}).first()[0]
    spark.sql(f"""
        MERGE INTO smartfactory.processed.pipeline_checkpoints AS t
        USING (SELECT 'bronze_to_silver' AS pipeline_name,
                      {new_version} AS last_processed_version,
                      current_timestamp() AS updated_at) AS s
        ON t.pipeline_name = s.pipeline_name
        WHEN MATCHED THEN UPDATE SET
            t.last_processed_version = s.last_processed_version,
            t.updated_at = s.updated_at
        WHEN NOT MATCHED THEN INSERT *
    """)
    print(f"✅ 처리 완료. 체크포인트: v{new_version}")


incremental_silver_pipeline()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.3 스키마 진화 (Schema Evolution)
# MAGIC

# COMMAND ----------

from datetime import datetime
# 스키마 진화: 새 컬럼 자동 추가
# 신규 컬럼(pressure_bar)이 포함된 데이터를 mergeSchema 옵션으로 삽입

new_df = spark.createDataFrame([
    ("EQ001", "LINE01", datetime.now(), "CNC", 65.0, 1.2, 12.0, 1500.0, 2.5)
], ["equipment_id", "line_id", "event_time", "equipment_type",
    "temperature_c", "vibration_ms2", "pressure_bar", "rpm"])  # 새 컬럼!

new_df.write.format("delta") \
    .option("mergeSchema", "true") \
    .mode("append") \
    .saveAsTable("smartfactory.raw.sensor_events")

# 결과: sensor_events 테이블에 pressure_bar 컬럼이 자동으로 추가됨
# 기존 행에는 NULL 값으로 채워짐

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 스키마 변경 이력 확인
# MAGIC DESCRIBE HISTORY smartfactory.raw.sensor_events;
# MAGIC -- operation: WRITE, operationParameters에 스키마 변경 정보 포함

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.4 실습 Lab — Delta Lake Time Travel 분석

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Lab 1-2: Time Travel 분석 (완전 버전)
# MAGIC -- 1. 현재 버전 확인
# MAGIC DESCRIBE HISTORY smartfactory.raw.sensor_events LIMIT 10;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 2. 어제 데이터와 오늘 데이터 비교
# MAGIC SELECT COUNT(*) AS today_count
# MAGIC FROM smartfactory.raw.sensor_events
# MAGIC WHERE DATE(event_time) = CURRENT_DATE();
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 실수로 데이터 삭제
# MAGIC DELETE FROM smartfactory.processed.sensor_clean
# MAGIC WHERE quality_score > 0.95;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 3. 실수로 삭제된 데이터 복구 (특정 버전으로 롤백)
# MAGIC -- 먼저 삭제 전 버전 찾기
# MAGIC DESCRIBE HISTORY smartfactory.processed.sensor_clean;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 버전 5 이전으로 복구
# MAGIC RESTORE TABLE smartfactory.processed.sensor_clean TO VERSION AS OF 6; -- MERGE 버전

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 4. Change Data Feed로 변경 이력 추적
# MAGIC SELECT *
# MAGIC FROM table_changes('smartfactory.raw.sensor_events', 7, 8)
# MAGIC WHERE _change_type IN ('insert', 'delete', 'update_preimage', 'update_postimage')
# MAGIC ORDER BY _commit_timestamp;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 5. 오래된 버전 정리 (저장 공간 확보)
# MAGIC VACUUM smartfactory.raw.sensor_events RETAIN 168 HOURS; -- 7일 보관

# COMMAND ----------

# DBTITLE 1,다음 단계
# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC
# MAGIC * **7장**: `01_data_analyst/02_lab02_production_kpi.py` (OEE 대시보드)
# MAGIC * RLS 실습은 이후 노트북에서 별도로 진행합니다.