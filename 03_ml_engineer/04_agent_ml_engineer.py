# Databricks notebook source
# 20장: ML 모니터링 에이전트 — 모델이 스스로 자신을 관리하다
# DBR 17.3 LTS (Spark 4.0.0 / Python 3.12.3)

# COMMAND ----------

# MAGIC %md
# MAGIC # 20장: ML 모니터링 에이전트
# MAGIC
# MAGIC 모델 성능 드리프트를 자동 감지하고 재학습을 트리거하는 에이전트:
# MAGIC - PSI/CSI 기반 데이터 드리프트 감지
# MAGIC - 모델 성능 지표 모니터링
# MAGIC - 자동 재학습 Job 트리거

# COMMAND ----------

# MAGIC %pip install langgraph langchain-databricks --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from langchain_databricks import ChatDatabricks
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from databricks.sdk import WorkspaceClient
import numpy as np

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. ML 모니터링 도구 정의

# COMMAND ----------

@tool
def check_model_performance() -> str:
    """Model Serving Endpoint의 최근 성능 지표를 확인합니다."""
    try:
        df = spark.sql("""
            SELECT
                DATE(TIMESTAMP_MILLIS(timestamp_ms)) AS date,
                COUNT(*) AS request_count,
                AVG(execution_duration_ms) AS avg_latency_ms,
                SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) AS error_count
            FROM smartfactory.ml.failure_prediction_inference_payload
            WHERE timestamp_ms >= UNIX_TIMESTAMP(CURRENT_DATE() - INTERVAL 7 DAYS) * 1000
            GROUP BY DATE(TIMESTAMP_MILLIS(timestamp_ms))
            ORDER BY date
        """)
        return df.toPandas().to_string(index=False)
    except Exception as e:
        return f"성능 조회 오류: {e}"

@tool
def calculate_data_drift(feature_name: str) -> str:
    """특정 피처의 PSI를 계산하여 데이터 드리프트를 감지합니다."""
    try:
        # 기준: 모델 학습 시점 (2주 전), 현재: 최근 7일
        df_ref = spark.sql(f"""
            SELECT {feature_name} FROM smartfactory.ml.equipment_sensor_features
            WHERE event_time BETWEEN CURRENT_DATE() - INTERVAL 21 DAYS
                              AND CURRENT_DATE() - INTERVAL 14 DAYS
        """).toPandas()[feature_name].dropna()

        df_cur = spark.sql(f"""
            SELECT {feature_name} FROM smartfactory.ml.equipment_sensor_features
            WHERE event_time >= CURRENT_DATE() - INTERVAL 7 DAYS
        """).toPandas()[feature_name].dropna()

        if len(df_ref) < 100 or len(df_cur) < 100:
            return f"{feature_name}: 데이터 부족 (기준={len(df_ref)}, 현재={len(df_cur)})"

        bins = np.percentile(df_ref, np.linspace(0, 100, 11))
        bins[0] -= 1e-6
        bins[-1] += 1e-6

        ref_pct = np.histogram(df_ref, bins=bins)[0] / len(df_ref)
        cur_pct = np.histogram(df_cur, bins=bins)[0] / len(df_cur)
        ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
        cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        status = "안정" if psi < 0.1 else ("주의" if psi < 0.25 else "드리프트 감지")
        return f"{feature_name} PSI = {psi:.4f} ({status})"
    except Exception as e:
        return f"PSI 계산 오류: {e}"

@tool
def get_model_registry_status() -> str:
    """Unity Catalog Model Registry에서 모델 버전 현황을 조회합니다."""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(registry_uri="databricks-uc")
        versions = client.search_model_versions("name='smartfactory.ml.predictive_maintenance'")

        result = []
        for v in versions:
            aliases = v.aliases if hasattr(v, 'aliases') else []
            result.append(f"버전 {v.version}: 상태={v.current_stage}, 별칭={aliases}, 생성={v.creation_timestamp}")
        return "\n".join(result) if result else "모델 없음"
    except Exception as e:
        return f"모델 레지스트리 조회 오류: {e}"

@tool
def trigger_retraining_job() -> str:
    """예지 정비 모델 재학습 Job을 트리거합니다."""
    try:
        jobs = list(w.jobs.list(name="smartfactory-daily-pipeline"))
        if not jobs:
            return "재학습 Job을 찾을 수 없습니다. 먼저 13장 Lab을 실행하세요."
        job_id = jobs[0].job_id
        run = w.jobs.run_now(job_id=job_id)
        return f"재학습 Job 시작: run_id={run.run_id}, job_id={job_id}"
    except Exception as e:
        return f"Job 트리거 오류: {e}"

tools = [check_model_performance, calculate_data_drift, get_model_registry_status, trigger_retraining_job]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. ML 모니터링 에이전트 구성

# COMMAND ----------

ML_MONITOR_PROMPT = """당신은 스마트팩토리 코리아의 ML 모델 모니터링 에이전트입니다.
예지 정비 모델 (smartfactory.ml.predictive_maintenance)의 성능과 데이터 품질을 지속적으로 모니터링합니다.

모니터링 기준:
- PSI > 0.25: 재학습 즉시 권장
- PSI 0.1~0.25: 재학습 일정 수립 권장
- 오류율 > 5%: 즉시 알림
- 평균 지연 > 500ms: 스케일링 검토

문제 발견 시 구체적인 조치 방안을 제시하고, 필요시 재학습 Job을 트리거하세요."""

llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    max_tokens=2048,
    temperature=0.0,
)

ml_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=ML_MONITOR_PROMPT,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 모니터링 실행

# COMMAND ----------

def run_ml_agent(question: str) -> str:
    result = ml_agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 15},
    )
    answer = result["messages"][-1].content
    print(f"\n{'='*60}\n질문: {question}\n\n답변:\n{answer}\n{'='*60}")
    return answer

# COMMAND ----------

# 종합 모니터링 리포트
run_ml_agent("""
예지 정비 모델의 현재 상태를 종합 점검해주세요:
1. 모델 레지스트리 버전 현황
2. Serving Endpoint 성능 (최근 7일)
3. 핵심 피처(temp_avg_24h, vib_avg_24h, anomaly_rate_24h)의 데이터 드리프트
4. 재학습 필요 여부 판단 및 조치 권고
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC - **21~22장**: `04_ai_engineer/01_lab01_foundation_models.py`