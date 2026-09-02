# Databricks notebook source
# MAGIC %md
# MAGIC # 27장 | 오케스트레이션 엔진 — 에이전트 워크플로우 자동화
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 27.1 이벤트 기반 에이전트 트리거

# COMMAND ----------

# DBTITLE 1,Table Trigger Job
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    Task, NotebookTask, TableUpdateTriggerConfiguration,
    TriggerSettings, Condition,
)

# =====================================================================
# 27.1 Table Trigger Job — 센서 이상 자동 대응
# =====================================================================
# sensor_clean 테이블에 새 데이터가 쌓이면 자동으로 Job이 트리거되어
# lab02_sensor_alert_job 노트북을 실행합니다.
#
# 흐름: sensor_clean 업데이트 → Table Trigger 감지(30~60초)
#       → lab02_sensor_alert_job 실행 → load_model(@champion)
#       → 이상값 조회 + 멀티에이전트 자동 대응
# =====================================================================

w = WorkspaceClient()

# --- 노트북 경로: 현재 노트북 기준 상대경로로 구성 ---
# dbutils.notebook.entry_point → 현재 노트북 경로에서 디렉토리 추출
current_notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
current_dir = "/".join(current_notebook_path.rsplit("/", 1)[:-1])
NOTEBOOK_PATH = f"{current_dir}/lab02_sensor_alert_job"

JOB_NAME = "센서 이상 자동 대응 (Table Trigger)"
TRIGGER_TABLE = "smartfactory.processed.sensor_clean"

print(f"📂 Job 노트북: {NOTEBOOK_PATH}")
print(f"📡 트리거 테이블: {TRIGGER_TABLE}")

# --- Job 생성 (이미 존재하면 건너뛰기) ---
existing = [j for j in w.jobs.list(name=JOB_NAME)]
if existing:
    job_id = existing[0].job_id
    print(f"\n✅ Job 이미 존재: {JOB_NAME} (ID: {job_id})")
else:
    created_job = w.jobs.create(
        name=JOB_NAME,
        tasks=[
            Task(
                task_key="sensor_alert_respond",
                notebook_task=NotebookTask(notebook_path=NOTEBOOK_PATH),
            )
        ],
        trigger=TriggerSettings(
            table_update=TableUpdateTriggerConfiguration(
                table_names=[TRIGGER_TABLE],
                condition=Condition.ANY_UPDATED,
                min_time_between_triggers_seconds=60,
                wait_after_last_change_seconds=60,
            )
        ),
    )
    job_id = created_job.job_id
    print(f"\n✅ Table Trigger Job 생성 완료")

print(f"   Job ID: {job_id}")
print(f"   트리거: {TRIGGER_TABLE} 테이블 업데이트 시 자동 실행")
print(f"   최소 간격: 60초 (너무 잦은 트리거 방지)")
print(f"   대기 시간: 60초 (마지막 변경 후 안정화 대기, 최소값)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 27.2 실습 Lab — 이벤트 기반 오케스트레이션 엔진 구현
# MAGIC > 5~10분 뒤에 insert를 실행하여 UI로 확인합니다.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,27.1 Job 트리거 테스트 — 가데이터 INSERT
from pyspark.sql import functions as F
from datetime import datetime, timedelta
import random

# =====================================================================
# Table Trigger Job 테스트용 가데이터 INSERT
# =====================================================================
# sensor_clean 테이블에 이상값 데이터를 추가하면
# Table Trigger가 감지하여 28_sensor_alert_job을 자동 실행합니다.
# =====================================================================

# --- 이상값 센서 데이터 생성 (3건) ---
now = datetime.now()
fake_alerts = [
    # 진동 이상 (vibration > 5.0)
    ("EQ003", "LINE01", now - timedelta(seconds=30), 6.8, 78.5, 4.2, 1450, 0.92, False),
    # 온도 이상 (temperature > 90)
    ("EQ007", "LINE02", now - timedelta(seconds=15), 3.1, 94.2, 5.1, 1380, 0.88, False),
    # 복합 이상 (진동 + 온도 둘 다 초과)
    ("EQ001", "LINE01", now, 7.5, 96.3, 3.8, 1520, 0.85, True),
]

schema = "equipment_id STRING, line_id STRING, event_time TIMESTAMP, " \
         "vibration_ms2 DOUBLE, temperature_c DOUBLE, pressure_bar DOUBLE, " \
         "rpm INT, quality_score DOUBLE, is_anomaly BOOLEAN"

fake_df = spark.createDataFrame(fake_alerts, schema=schema) \
    .withColumn("temp_zscore", F.lit(3.5)) \
    .withColumn("_processed_time", F.current_timestamp())

# --- sensor_clean 테이블에 APPEND ---
fake_df.write.format("delta").mode("append") \
    .saveAsTable("smartfactory.processed.sensor_clean")

print(f"✅ 가데이터 {len(fake_alerts)}건 INSERT 완료")
print(f"   대상 테이블: smartfactory.processed.sensor_clean")
print(f"   이상 설비: EQ003(진동), EQ007(온도), EQ001(복합)")
print(f"\n⏳ Table Trigger Job이 2~5분 내에 자동 실행됩니다...")
print(f"   Job UI에서 확인: '{JOB_NAME}'")