# Databricks notebook source
# MAGIC %md
# MAGIC # 센서 이상 자동 대응 Job
# MAGIC
# MAGIC `smartfactory.processed.sensor_clean` 테이블 업데이트 시 자동 트리거되어 이상값을 감지하고 멀티에이전트를 호출합니다.

# COMMAND ----------

# DBTITLE 1,pip install
# MAGIC %pip install --upgrade "langgraph-supervisor>=0.0.10" "langchain>=1.0" "langgraph>=1.1.0" "databricks-langchain>=0.19.0" "databricks-sdk>=0.118.0" --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,모델 로드 + 자동 대응 + monitoring 저장
import os
import time
import uuid
import mlflow
from pyspark.sql.functions import col, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
from datetime import datetime

# === 1. 멀티에이전트 로드 ===
MODEL_NAME = "smartfactory.ai.multi_agent_supervisor"
multi_agent = mlflow.langchain.load_model(f"models:/{MODEL_NAME}@champion")
print(f"✅ 모델 로드: {MODEL_NAME}@champion")

# === 2. 이상값 조회 ===
alert_df = (
    spark.table("smartfactory.processed.sensor_clean")
    .filter((col("vibration_ms2") > 5.0) | (col("temperature_c") > 90))
    .orderBy(col("event_time").desc())
    .limit(10)
)
alerts = alert_df.collect()
print(f"🚨 이상 감지: {len(alerts)}건")

# === 3. 순차 처리 + 결과 수집 ===
REQUEST_DELAY = 3          # 요청 간 대기(초) — 429 방지
RETRY_DELAY   = 5          # 재시도 대기(초)
MAX_RETRIES   = 3

results = []
for i, row in enumerate(alerts):
    alert_type = "진동 이상" if row["vibration_ms2"] > 5.0 else "온도 이상"
    query = (
        f"{row['equipment_id']} 설비에서 {alert_type} 경보 발생. "
        f"센서값: vibration={row['vibration_ms2']}, temp={row['temperature_c']}. "
        f"고장 예측, 정비 절차, 라인 영향 분석을 수행해주세요."
    )

    response_text, status = "", "FAILED"
    for attempt in range(MAX_RETRIES):
        try:
            thread_id = f"alert-{row['equipment_id']}-{uuid.uuid4().hex[:8]}"
            t0 = time.time()
            result = multi_agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"recursion_limit": 30, "configurable": {"thread_id": thread_id}},
            )
            elapsed = round(time.time() - t0, 1)
            response_text = result["messages"][-1].content
            status = "SUCCESS"
            print(f"✅ [{i+1}/{len(alerts)}] {row['equipment_id']} 대응 완료 ({len(response_text)}자, {elapsed}s)")
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  ⚠️ {row['equipment_id']} 재시도 {attempt+1}/{MAX_RETRIES}: {str(e)[:80]}")
                time.sleep(RETRY_DELAY)
            else:
                elapsed = round(time.time() - t0, 1)
                response_text = f"ERROR: {str(e)[:500]}"
                print(f"  ❌ {row['equipment_id']} {MAX_RETRIES}회 실패: {str(e)[:100]}")

    results.append((
        f"alert-{uuid.uuid4().hex[:12]}",
        row["equipment_id"],
        row["line_id"],
        alert_type,
        float(row["vibration_ms2"]),
        float(row["temperature_c"]),
        query,
        response_text,
        status,
        thread_id,
        elapsed,
        datetime.utcnow(),
    ))

    # 순차 처리: 다음 요청 전 대기 (마지막 건 제외)
    if i < len(alerts) - 1:
        time.sleep(REQUEST_DELAY)

# === 4. monitoring 테이블 저장 ===
schema = StructType([
    StructField("alert_id",        StringType()),
    StructField("equipment_id",    StringType()),
    StructField("line_id",         StringType()),
    StructField("alert_type",      StringType()),
    StructField("vibration_ms2",   DoubleType()),
    StructField("temperature_c",   DoubleType()),
    StructField("query",           StringType()),
    StructField("agent_response",  StringType()),
    StructField("status",          StringType()),
    StructField("thread_id",       StringType()),
    StructField("response_time_s", DoubleType()),
    StructField("created_at",      TimestampType()),
])

result_df = spark.createDataFrame(results, schema=schema)
(
    result_df.write
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable("smartfactory.ai.ai_monitoring")
)

success = sum(1 for r in results if r[8] == "SUCCESS")
failed  = len(results) - success
print(f"\n🏁 완료: {len(alerts)}건 처리 (성공: {success}, 실패: {failed})")
print(f"📊 저장: smartfactory.ai.ai_monitoring ({len(results)}건 추가)")

# COMMAND ----------

# DBTITLE 1,모니터링 테이블 확인
# === ai_monitoring 테이블 최근 결과 확인 ===
display(
    spark.table("smartfactory.ai.ai_monitoring")
    .orderBy(col("created_at").desc())
    .limit(20)
)

# COMMAND ----------

# DBTITLE 1,트러블슈팅 가이드
# MAGIC %md
# MAGIC ### ⚠️ 트러블슈팅 가이드
# MAGIC
# MAGIC | 증상 | 원인 | 해결 |
# MAGIC | --- | --- | --- |
# MAGIC | `429 REQUEST_LIMIT_EXCEEDED` | LLM 서빙 엔드포인트 Rate Limit 초과 | `REQUEST_DELAY`를 3→5초로 늘리거나, 엔드포인트 Provisioned Throughput 조정 |
# MAGIC | `ReadTimeout (30s)` | `smartfactory-pdm` 엔드포인트 콜드 스타트 또는 응답 지연 | 엔드포인트 scale-to-zero 설정 확인, timeout 값 증가 |
# MAGIC | `NotFound: Model version 'X' does not exist` | champion alias가 삭제된 버전을 가리킴 | Cell 20(버전 등록 노트북)을 재실행하여 alias 갱신 |
# MAGIC | monitoring 테이블에 데이터 없음 | sensor_clean에 이상값이 없거나 필터 조건 불일치 | `vibration_ms2 > 5.0` 또는 `temperature_c > 90` 임계값 확인 |
# MAGIC | `AnalysisException: Table not found` | `smartfactory.ai.monitoring` 스키마/카탈로그 권한 부족 | `USE CATALOG smartfactory; USE SCHEMA ai;` 권한 확인 |
# MAGIC
# MAGIC > **팁**: Job 실행 시 `REQUEST_DELAY`와 `RETRY_DELAY`를 환경에 맞게 조정하세요. 프로덕션에서는 Provisioned Throughput 엔드포인트 사용을 권장합니다.