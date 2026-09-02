# Databricks notebook source
# MAGIC %md
# MAGIC # 26장 | 멀티에이전트 아키텍처 — 전문 에이전트들의 협업
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC > 외부 PostgreSQL 인스턴스 사용을 위해 인프라팀의 데이터베이스 생성, 네트워크 연결, 보안 정책 및 방화벽 설정 등이 필요할 수 있습니다. <br>
# MAGIC > 반면 이 장에서는 별도의 인프라 구축 없이 독자가 직접 실습할 수 있도록 Neon Free Tier의 Serverless PostgreSQL을 사용합니다.<br>
# MAGIC > Cluster를 **Serverless** 로 교체하여 사용합니다. <br>
# MAGIC > 25장 실습 환경 안내 참조

# COMMAND ----------

# MAGIC %md
# MAGIC ##26.1 멀티에이전트 라우팅 아키텍처
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,27.2 패키지 설치
# MAGIC %pip install --upgrade "langgraph-supervisor>=0.0.10" "langchain>=1.0" "langchain-openai>=1.2" "langgraph>=1.1.0" "databricks-langchain>=0.19.0" "databricks-sdk>=0.118.0" "langgraph-checkpoint-postgres>=2.0.0" "psycopg[binary]>=3.1" "psycopg-pool>=3.1" --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

!pip list | grep langchain && pip list | grep langgraph && pip list | grep mlflow

# COMMAND ----------

# DBTITLE 1,도구 정의
import os
import json
from langchain.tools import tool
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
from databricks.sdk import WorkspaceClient
from langchain.agents import create_agent


# === LLM 설정 (전문 에이전트용 70B, 수퍼바이저 의도분류용 8B) ===
llm_70b = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    temperature=0.1, max_tokens=1024
)


# === SQL 실행 헬퍼 (Statement Execution API — 서빙 환경 호환) ===
# spark.sql()은 Model Serving에 SparkSession이 없어 작동하지 않음.
# statement_execution API는 REST 기반으로 노트북/서빙 모두 동작.
_ws = WorkspaceClient()


# Warehouse ID: 환경변수 우선 → 없으면 이름으로 자동 검색
_WAREHOUSE_NAME = "smartfactory-agent-wh"
_WAREHOUSE_ID = os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID")
if not _WAREHOUSE_ID:
    _warehouses = [wh for wh in _ws.warehouses.list() if wh.name == _WAREHOUSE_NAME]
    if _warehouses:
        _WAREHOUSE_ID = _warehouses[0].id
        os.environ["DATABRICKS_SQL_WAREHOUSE_ID"] = _WAREHOUSE_ID



def _execute_sql(query: str) -> list[dict]:
    """데이터브릭스 Statement Execution API로 SQL 실행 후 dict 리스트 반환"""
    resp = _ws.statement_execution.execute_statement(
        statement=query,
        warehouse_id=_WAREHOUSE_ID,
        wait_timeout="30s",
    )
    if not resp.result or not resp.result.data_array:
        return []
    columns = [col.name for col in resp.manifest.schema.columns]
    return [dict(zip(columns, row)) for row in resp.result.data_array]


# === 1. 정비 에이전트 도구 ===

# 1-1. 매뉴얼 검색 (VectorSearchRetrieverTool — 다국어 임베딩: qwen3-embedding-0-6b)
search_maintenance_manual = VectorSearchRetrieverTool(
    index_name="smartfactory.ai.manual_index",
    num_results=3,
    columns=["chunk_id", "doc_id", "title", "equipment_type", "doc_type", "page_id", "provider"],
    tool_name="search_maintenance_manual",
    tool_description=(
        "Search OSHA/NIOSH safety manuals for maintenance procedures, "
        "LOTO lockout/tagout steps, hazardous energy control, and machine guarding requirements. "
        "Supports Korean and English queries (multilingual embedding model)."
    ),
)


# 1-2. 실시간 센서 데이터 조회
@tool
def get_realtime_sensor_data(equipment_id: str) -> str:
    """설비의 최신 센서 데이터를 조회합니다. (temperature, vibration, pressure, rpm, anomaly 여부)"""
    rows = _execute_sql(f'''
        SELECT equipment_id, equipment_type, temperature_c, vibration_ms2,
               pressure_bar, rpm, is_anomaly, event_time
        FROM smartfactory.processed.sensor_clean
        WHERE equipment_id = '{equipment_id}'
        ORDER BY event_time DESC LIMIT 1
    ''')
    if not rows:
        return f"{equipment_id}: 센서 데이터 없음"
    row = rows[0]
    anomaly_status = "⚠️ ANOMALY 감지" if row['is_anomaly'] in ('true', True, '1') else "✅ 정상"
    return (
        f"현재 설비 상태 ({row['equipment_id']} / {row['equipment_type']}):\n"
        f"  시각: {row['event_time']}\n"
        f"  온도: {float(row['temperature_c']):.1f}°C | 진동: {float(row['vibration_ms2']):.2f}mm/s\n"
        f"  압력: {float(row['pressure_bar']):.1f}bar | RPM: {float(row['rpm']):.0f}\n"
        f"  상태: {anomaly_status}"
    )


# 1-3. 작업 지시서 생성
@tool
def create_work_order(equipment_id: str, issue_summary: str, priority: str = "HIGH") -> str:
    """정비 작업 지시서를 자동 생성합니다. priority: HIGH/MEDIUM/LOW"""
    import uuid
    wo_id = f"WO-{uuid.uuid4().hex[:8].upper()}"
    return (
        f"📋 작업 지시서 생성 완료\n"
        f"  지시번호: {wo_id}\n"
        f"  대상 설비: {equipment_id}\n"
        f"  우선순위: {priority}\n"
        f"  이슈 요약: {issue_summary}\n"
        f"  상태: 대기 (정비원 배정 필요)"
    )

# === 2. 분석 에이전트 도구 ===

@tool
def analyze_oee(line_id: str = "", days: int = 30) -> str:
    """생산 라인의 OEE(종합설비효율) 현황을 분석합니다. line_id는 LINE01~LINE10 중 하나를 지정하거나, 빈 문자열로 두면 전체 라인 비교."""
    where = f"WHERE line_id = '{line_id}'" if line_id and line_id.strip() else "WHERE 1=1"
    rows = _execute_sql(f"""
        SELECT line_id,
               ROUND(AVG(oee_pct), 1)         AS avg_oee_pct,
               ROUND(AVG(availability_pct), 1) AS avail_pct,
               ROUND(AVG(performance_pct), 1)  AS perf_pct,
               ROUND(AVG(quality_pct), 1)       AS qual_pct
        FROM smartfactory.analytics.oee_daily
        {where}
          AND production_date >= CAST(DATE_SUB(CURRENT_DATE(), 30) AS STRING)
        GROUP BY line_id ORDER BY avg_oee_pct
    """)
    if not rows:
        return "OEE 데이터 없음"
    import pandas as pd
    return pd.DataFrame(rows).to_string(index=False)


@tool
def query_sensor_anomalies(equipment_id: str = None, hours: int = 24) -> str:
    """최근 N시간 내 센서 이상 현황을 조회합니다. 설비별 이상 횟수와 평균 온도/진동을 반환."""
    where = f"AND equipment_id = '{equipment_id}'" if equipment_id else ""
    rows = _execute_sql(f"""
        SELECT equipment_id, equipment_type, line_id,
               COUNT(*) AS anomaly_count,
               ROUND(AVG(temperature_c), 1) AS avg_temp,
               ROUND(AVG(vibration_ms2), 3) AS avg_vib
        FROM smartfactory.processed.sensor_clean
        WHERE is_anomaly = TRUE
          AND event_time >= CURRENT_TIMESTAMP() - INTERVAL {hours} HOURS
          {where}
        GROUP BY equipment_id, equipment_type, line_id
        ORDER BY anomaly_count DESC LIMIT 10
    """)
    if not rows:
        return "최근 이상 감지 없음"
    import pandas as pd
    return pd.DataFrame(rows).to_string(index=False)




# === 3. ML 에이전트 도구 ===

@tool
def predict_failure_probability(equipment_id: str) -> str:
    """48시간 내 고장 확률을 ML 모델(smartfactory-pdm)로 예측합니다."""
    import requests as _requests


    rows = _execute_sql(f'''
        SELECT temperature_c, vibration_ms2, pressure_bar, rpm, quality_score, temp_zscore
        FROM smartfactory.processed.sensor_clean
        WHERE equipment_id = '{equipment_id}'
        ORDER BY event_time DESC LIMIT 1
    ''')


    if not rows:
        return f"{equipment_id}: 센서 데이터 없음 → 고장 예측 불가"


    row = rows[0]
    features = [[float(row['temperature_c']), float(row['vibration_ms2']),
                 float(row['pressure_bar']), float(row['rpm']),
                 float(row['quality_score']), float(row['temp_zscore'])]]
    _host = _ws.config.host.replace("https://", "")
    _token = _ws.config.token
    pdm_url = f"https://{_host}/serving-endpoints/smartfactory-pdm/invocations"


    try:
        resp = _requests.post(
            pdm_url,
            headers={"Authorization": f"Bearer {_token}", "Content-Type": "application/json"},
            json={"inputs": features},
            timeout=30,
        )
        if resp.status_code != 200:
            return f"{equipment_id}: PDM 엔드포인트 오류 (HTTP {resp.status_code})"
        prediction = resp.json().get("predictions", [0])[0]
    except Exception as e:
        return f"{equipment_id}: PDM 호출 실패 ({e})"


    risk = "HIGH ⚠️" if prediction == 1 else "LOW ✅"
    rec = "즉시 정비 필요" if prediction == 1 else "정상 운전 유지"
    return (
        f"{equipment_id} 고장예측: {risk}\n"
        f"  온도: {float(row['temperature_c']):.1f}°C | 진동: {float(row['vibration_ms2']):.2f}mm/s\n"
        f"  권장: {rec}"
    )


@tool
def get_model_champion_info() -> str:
    """현재 Champion 예측 모델의 성능 지표(AUC, F1)를 조회합니다."""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(registry_uri="databricks-uc")
        v = client.get_model_version_by_alias("smartfactory.ml.predictive_maintenance", "champion")
        metrics = client.get_run(v.run_id).data.metrics
        return (
            f"Champion 모델 정보:\n"
            f"  버전: {v.version}\n"
            f"  AUC: {metrics.get('test_auc', 'N/A'):.4f}\n"
            f"  F1: {metrics.get('test_f1', 'N/A'):.4f}\n"
            f"  등록일: {v.creation_timestamp}"
        )
    except Exception as e:
        return f"모델 정보 조회 오류: {e}"




# === 4. 보고서 에이전트 도구 ===


@tool
def generate_daily_report(line_id: str = None) -> str:
    """일일 종합 보고서를 생성합니다. OEE, 이상 감지, 정비 현황을 포함."""
    import pandas as pd
    # OEE 현황
    line_filter = f"AND line_id = '{line_id}'" if line_id else ""
    oee_rows = _execute_sql(f"""
        SELECT line_id,
               ROUND(AVG(oee_pct), 1) AS avg_oee,
               COUNT(*) AS days_tracked
        FROM smartfactory.analytics.oee_daily
        WHERE production_date >= CAST(DATE_SUB(CURRENT_DATE(), 30) AS STRING)
        {line_filter}
        GROUP BY line_id ORDER BY avg_oee
    """)
    # 이상 현황
    anomaly_rows = _execute_sql("""
        SELECT COUNT(*) AS total_anomalies,
               COUNT(DISTINCT equipment_id) AS affected_equipment
        FROM smartfactory.processed.sensor_clean
        WHERE is_anomaly = TRUE
          AND event_time >= CURRENT_TIMESTAMP() - INTERVAL 24 HOURS
    """)


    oee_summary = pd.DataFrame(oee_rows).to_string(index=False) if oee_rows else "데이터 없음"
    anomaly_row = anomaly_rows[0] if anomaly_rows else {"total_anomalies": 0, "affected_equipment": 0}


    return (
        f"📊 일일 종합 보고서 (최근 30일 기준)\n"
        f"{'='*50}\n"
        f"[OEE 현황]\n{oee_summary}\n\n"
        f"[24시간 이상 감지]\n"
        f"  총 이상 건수: {anomaly_row['total_anomalies']}건\n"
        f"  영향 설비 수: {anomaly_row['affected_equipment']}대\n"
        f"{'='*50}\n"
        f"보고서 생성 완료"
    )




# === 도구 그룹 정의 ===
MAINTENANCE_TOOLS = [search_maintenance_manual, get_realtime_sensor_data, create_work_order]
ANALYSIS_TOOLS = [analyze_oee, query_sensor_anomalies]
ML_TOOLS = [predict_failure_probability, get_model_champion_info]
REPORT_TOOLS = [generate_daily_report]




print("✅ 전문 에이전트 도구 정의 완료")
print(f"   정비 에이전트: {len(MAINTENANCE_TOOLS)}개 도구")
print(f"   분석 에이전트: {len(ANALYSIS_TOOLS)}개 도구")
print(f"   ML 에이전트: {len(ML_TOOLS)}개 도구")
print(f"   보고서 에이전트: {len(REPORT_TOOLS)}개 도구")

# COMMAND ----------

# DBTITLE 1,에이전트 생성
from langchain.agents import create_agent

# === 1. 정비 에이전트 ===
maintenance_agent = create_agent(
    model=llm_70b,
    tools=MAINTENANCE_TOOLS,
    system_prompt=(
        "당신은 스마트팩토리 코리아의 설비 정비 전문 에이전트입니다.\n"
        "담당 업무: 정비 매뉴얼 검색, 센서 데이터 확인, 작업 지시서 생성\n\n"
        "⚠️ 필수 규칙:\n"
        "- search_maintenance_manual 호출 시 한국어 또는 영어로 query 작성 가능 (다국어 임베딩)\n"
        "- ANOMALY 감지 시 LOTO 절차를 반드시 언급하세요\n"
        "- 고온·고진동 상태에서는 잔여 에너지 해소의 중요성을 강조하세요\n"
        "- 최종 답변은 한국어로 명확하게 요약하세요"
    ),
    name="maintenance_agent",
)

# === 2. 분석 에이전트 ===
analysis_agent = create_agent(
    model=llm_70b,
    tools=ANALYSIS_TOOLS,
    system_prompt=(
        "당신은 스마트팩토리 코리아의 데이터 분석 전문 에이전트입니다.\n"
        "담당 업무: OEE(종합설비효율) 분석, 센서 이상 현황 통계\n\n"
        "규칙:\n"
        "- OEE 80% 미만은 개선 필요 라인으로 표시\n"
        "- 이상 감지 설비는 원인 추정을 함께 제공\n"
        "- 데이터 기반으로 객관적 팬트를 제시\n"
        "- 한국어로 답변하세요"
    ),
    name="analysis_agent",
)

# === 3. ML 에이전트 ===
ml_agent = create_agent(
    model=llm_70b,
    tools=ML_TOOLS,
    system_prompt=(
        "당신은 스마트팩토리 코리아의 ML 엔지니어 전문 에이전트입니다.\n"
        "담당 업무: 고장 예측 모델 결과 해석, 모델 성능 보고\n\n"
        "규칙:\n"
        "- 고장 위험도 HIGH는 즉시 정비 권고 메시지 포함\n"
        "- 모델 성능 지표(AUC, F1)를 비전문가도 이해할 수 있게 설명\n"
        "- 한국어로 답변하세요"
    ),
    name="ml_agent",
)

# === 4. 보고서 에이전트 ===
report_agent = create_agent(
    model=llm_70b,
    tools=REPORT_TOOLS,
    system_prompt=(
        "당신은 스마트팩토리 코리아의 보고서 생성 전문 에이전트입니다.\n"
        "담당 업무: 일일/주간 종합 리포트 생성, 경영진 KPI 요약\n\n"
        "규칙:\n"
        "- 보고서 형식을 정렬하여 제공 (제목/요약/상세)\n"
        "- 핵심 지표는 숫자와 수치 권고를 함께 표시\n"
        "- 한국어로 답변하세요"
    ),
    name="report_agent",
)

print("✅ 4개 전문 에이전트 생성 완료")
print("   - maintenance_agent: 정비 매뉴얼 검색 + 센서 + 작업지시")
print("   - analysis_agent: OEE 분석 + 이상 현황")
print("   - ml_agent: 고장 예측 + 모델 성능")
print("   - report_agent: 일일 보고서 생성")

# COMMAND ----------

# DBTITLE 1,langgraph-supervisor 구성
from langgraph_supervisor import create_supervisor

# === Supervisor 생성 ===
# langgraph-supervisor는 LLM 기반으로 의도를 파악하고,
# handoff 메커니즘으로 적절한 전문 에이전트를 호출합니다.

supervisor = create_supervisor(
    agents=[maintenance_agent, analysis_agent, ml_agent, report_agent],
    model=llm_70b,
    prompt=(
        "당신은 스마트팩토리 코리아의 AI 오케스트레이터(수퍼바이저)입니다.\n"
        "사용자의 질문을 분석하여 적절한 전문 에이전트에게 작업을 위임하세요.\n\n"
        "전문 에이전트 역할:\n"
        "- maintenance_agent: 설비 정비 절차, 안전 매뉴얼, LOTO, 작업지시서, 센서 데이터 확인\n"
        "- analysis_agent: OEE 분석, 생산 KPI, 센서 이상 통계, 효율 비교\n"
        "- ml_agent: 고장 예측, 모델 성능 조회, 위험도 평가\n"
        "- report_agent: 일일/주간 보고서, KPI 요약, 경영진 리포트\n\n"
        "복합 질의 처리 규칙:\n"
        "- 질문이 여러 도메인에 걸치면 해당 에이전트들을 순차적으로 호출하세요\n"
        "- 예: 'OEE 낮은 이유가 고장 때문인가?' → analysis_agent + ml_agent\n"
        "- 예: '정비 전 절차와 예측 결과' → maintenance_agent + ml_agent\n"
        "- 각 에이전트 결과를 종합하여 최종 답변을 한국어로 제공하세요"
    ),
    output_mode="full_history",  # 에이전트 간 전체 대화 이력 공유
)

# 그래프 컴파일
multi_agent_app = supervisor.compile()

print("✅ langgraph-supervisor 멀티에이전트 시스템 구성 완료")
print("   Supervisor → 4개 전문 에이전트 라우팅 준비")
print(f"   그래프 노드: {list(multi_agent_app.get_graph().nodes.keys())}")

# === 그래프 시각화 (Mermaid PNG) ===
from IPython.display import Image, display
try:
    img_data = multi_agent_app.get_graph().draw_mermaid_png()
    display(Image(img_data))
except Exception as e:
    # Mermaid 렌더링 실패 시 ASCII 폴백
    print(f"\n📊 그래프 구조 (ASCII):")
    multi_agent_app.get_graph().print_ascii()




# COMMAND ----------

# DBTITLE 1,테스트
# === 테스트 헬퍼 ===
def ask_multi_agent(question: str) -> str:
    """멀티에이전트 시스템에 질문하고 결과를 출력"""
    print(f"\n{'='*60}")
    print(f"🗣️  질문: {question}")
    print(f"{'='*60}")


    result = multi_agent_app.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 25},
    )


    # 에이전트 호출 추적
    agents_called = []
    for msg in result["messages"]:
        if hasattr(msg, 'name') and msg.name and msg.name != 'supervisor':
            if msg.name not in agents_called:
                agents_called.append(msg.name)


    if agents_called:
        print(f"\n🔄 호출된 에이전트: {' → '.join(agents_called)}")


    # 최종 답변
    final_answer = result["messages"][-1].content
    print(f"\n💬 최종 답변:\n{final_answer[:800]}")
    return final_answer


# --- 테스트 1: 단일 도메인 (분석) ---
ask_multi_agent("현재 OEE가 80% 미만인 생산 라인을 분석해주세요.")


# --- 테스트 2: 복합 질의 (분석 + ML) ---
ask_multi_agent("LINE05 OEE가 낮은 이유가 설비 고장 때문인가요? 이상 감지된 설비의 고장 예측 결과도 함께 알려주세요.")


# --- 테스트 3: 복합 질의 (정비 + ML) ---
ask_multi_agent(
    "EQ005 CNC 설비에서 anomaly가 감지되었습니다. "
    "현재 상태를 확인하고, 48시간 내 고장 확률을 예측해주세요. "
    "위험도가 높으면 정비 전 안전 절차(LOTO)와 작업 지시서까지 생성해주세요."
)


# COMMAND ----------

# DBTITLE 1,27.5 모델 등록 — 재사용성과 운영 확장성
# MAGIC %md
# MAGIC ##26.2 멀티에이전트 수퍼바이저 시스템 구현

# COMMAND ----------

import os
from databricks.sdk import WorkspaceClient
import psycopg
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

# --- SQL Warehouse ID 설정 (multi_agent.py와 동일한 로직) ---
# multi_agent.py 내부에도 동일한 자동 검색 로직이 포함되어 있어
# load_model() 시에도 자동으로 Warehouse ID가 설정됩니다.
# 서빙 환경에서는 env_vars에 DATABRICKS_SQL_WAREHOUSE_ID를 직접 설정합니다.

WAREHOUSE_NAME = "smartfactory-agent-wh"

if not os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID"):
    w = WorkspaceClient()
    warehouses = [wh for wh in w.warehouses.list() if wh.name == WAREHOUSE_NAME]
    if warehouses:
        os.environ["DATABRICKS_SQL_WAREHOUSE_ID"] = warehouses[0].id

warehouse_id = os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID")
if warehouse_id:
    print(f"✅ SQL Warehouse: {WAREHOUSE_NAME} (ID: {warehouse_id[:3]}...)")
else:
    print(f"⚠️ '{WAREHOUSE_NAME}' 웨어하우스를 찾을 수 없습니다.")
    w = WorkspaceClient()
    for wh in w.warehouses.list():
        print(f"   - {wh.name} (ID: {wh.id})")


# ── 1. Neon Postgres 연결 정보 ──────────────────────────────────────────
# Neon Console (https://console.neon.tech) 에서:
#   1. 프로젝트 생성 (Free tier: 0.5GB 저장, 24/7 사용 가능)
#   2. Connection Details 에서 Connection String 복사
#   형식: postgresql://<user>:<password>@<host>/<dbname>?sslmode=require
# ───────────────────────────────────────────────────────────

# ⬇️ 여기에 Neon Connection String 입력
NEON_CONN_STRING = "postgresql://neondb_owner:....ap-southeast-1.aws.neon.tech/neondb?sslmode=require"

# 프로덕션은 dbutils.secrets.get(scope="neon", key="conn_string") 사용 권장
# NEON_CONN_STRING = dbutils.secrets.get(scope="neon", key="conn_string")


# 운영용 Connection Pool
pool = ConnectionPool(conninfo=NEON_CONN_STRING, min_size=1, max_size=5)
checkpointer = PostgresSaver(pool)

# ── 2. 연결 검증 ──────────────────────────────────────────────
with pool.connection() as conn:
    result = conn.execute("SELECT 1").fetchone()
    assert result == (1,), "DB 연결 실패"

# ── 3. 환경변수 저장 (서빙에서 사용) ─────────
os.environ["NEON_CONN_STRING"] = NEON_CONN_STRING

print(f"\n📝 서빙 시 필요한 환경변수:")
print(f"   DATABRICKS_SQL_WAREHOUSE_ID = {os.environ.get('DATABRICKS_SQL_WAREHOUSE_ID')[:3]}...")
print(f"   NEON_CONN_STRING = {os.environ.get('NEON_CONN_STRING')[:10]}...")

# COMMAND ----------

# DBTITLE 1,26.5 MLflow 모델 등록 (Unity Catalog)
import mlflow
import os
from mlflow.models import infer_signature
from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksSQLWarehouse,
    DatabricksServingEndpoint,
    DatabricksTable,
)

# === 멀티에이전트 수퍼바이저를 MLflow 모델로 UC에 등록 (models-from-code) ===
mlflow.set_registry_uri("databricks-uc")

username = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{username}/multi_agent_supervisor")

MODEL_NAME = "smartfactory.ai.multi_agent_supervisor"

# 모델 정의 파일 경로 (models-from-code 패턴)
MODEL_CODE_PATH = "lab01_multi_agent.py"

print(f"📄 모델 코드 경로: {MODEL_CODE_PATH}")

# 입력 예시 (시그니처 추론용 — thread_id 포함: Checkpointer 필수)
input_example = {
    "messages": [{"role": "user", "content": "현재 OEE가 80% 미만인 라인을 분석해주세요."}],
    "custom_inputs": {"thread_id": "session-01"},
}

# 서빙 환경에 필요한 패키지
pip_requirements=[
    "langgraph-supervisor>=0.0.10",
    "databricks-langchain>=0.19.0",
    "langchain>=1.0",
    "langgraph>=1.1.0",
    "langchain-openai>=1.2",
    "langgraph-checkpoint-postgres>=2.0.0",
    "psycopg[binary]>=3.1",
    "psycopg-pool>=3.1",
    "databricks-sdk>=0.118.0",
    "pandas",
    "mlflow"
]
# 리소스 선언 — agents.deploy() 시 자동 인증 토큰에 권한 자동 부여
# DatabricksSQLWarehouse: CAN_USE 권한
# DatabricksTable: SELECT 권한 (USE CATALOG/SCHEMA 포함)
# DatabricksServingEndpoint: CAN_QUERY 권한
resources = [
    DatabricksVectorSearchIndex(index_name="smartfactory.ai.manual_index"),
    DatabricksSQLWarehouse(warehouse_id=os.environ.get('DATABRICKS_SQL_WAREHOUSE_ID')),
    DatabricksTable(table_name="smartfactory.processed.sensor_clean"),
    DatabricksTable(table_name="smartfactory.analytics.oee_daily"),
    DatabricksServingEndpoint(endpoint_name="smartfactory-pdm"),
    DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct"),
]

# 모델 시그니처 (UC 등록 필수 — Checkpointer로 인해 자동 추론 불가, 명시적 지정)
signature = infer_signature(
    model_input={
        "messages": [{"role": "user", "content": "hello"}],
        "custom_inputs": {"thread_id": "session-01"},
    },
    model_output="assistant response",
)

# MLflow models-from-code 로깅
with mlflow.start_run(run_name="multi-agent-supervisor-v1") as run:
    model_info = mlflow.langchain.log_model(
        lc_model=MODEL_CODE_PATH,  # 파일 경로 전달 (models-from-code)
        name="multi_agent_supervisor",
        input_example=input_example,
        signature=signature,
        pip_requirements=pip_requirements,
        resources=resources,  # SP 자동 권한 부여 대상
    )
    print(f"✅ MLflow 로깅 완료: {run.info.run_id}")

# Unity Catalog 모델 레지스트리에 등록
registered = mlflow.register_model(
    model_uri=f"runs:/{run.info.run_id}/multi_agent_supervisor",
    name=MODEL_NAME,
)
print(f"✅ UC 모델 등록: {MODEL_NAME} v{registered.version}")

# @champion 에일리어스 설정
cli = mlflow.MlflowClient()
cli.set_registered_model_alias(MODEL_NAME, "champion", registered.version)
print(f"✅ @champion 설정 완료: {MODEL_NAME}@champion (v{registered.version})")

# COMMAND ----------

# DBTITLE 1,등록 모델 로드 테스트
import mlflow

# === 등록된 멀티에이전트 모델 로드 및 테스트 ===
MODEL_NAME = "smartfactory.ai.multi_agent_supervisor"

loaded_model = mlflow.langchain.load_model(f"models:/{MODEL_NAME}@champion")
print(f"✅ 모델 로드 완료: {MODEL_NAME}@champion")

# --- 테스트 1: 단일 도메인 ---
result = loaded_model.invoke(
    {"messages": [{"role": "user", "content": "현재 OEE가 80% 미만인 라인을 분석해주세요."}]},
    config={"recursion_limit": 25, "configurable": {"thread_id": "test-session-01"}},
)
print(f"\n💬 답변:\n{result['messages'][-1].content[:500]}")

# --- 테스트 2: 복합 질의 (ML + 정비) ---
result2 = loaded_model.invoke(
    {"messages": [{"role": "user", "content": "EQ005 설비의 고장 예측 결과와 LOTO 안전 절차를 알려주세요."}]},
    config={"recursion_limit": 25, "configurable": {"thread_id": "test-session-02"}},
)
print(f"\n💬 복합 질의 답변:\n{result2['messages'][-1].content[:500]}")