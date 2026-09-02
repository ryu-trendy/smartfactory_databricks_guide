import mlflow

import os
import json
from langchain.tools import tool
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
from databricks.sdk import WorkspaceClient
from langgraph_supervisor import create_supervisor
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
    _WAREHOUSE_ID = _warehouses[0].id if _warehouses else None
    if _WAREHOUSE_ID:
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
    output_mode="full_history",
)

# === PostgresSaver checkpointer (Neon) ===
NEON_CONN_STRING = os.environ.get("NEON_CONN_STRING", "")

if NEON_CONN_STRING:
    from psycopg_pool import ConnectionPool
    from langgraph.checkpoint.postgres import PostgresSaver

    _pool = ConnectionPool(conninfo=NEON_CONN_STRING, min_size=1, max_size=5)
    checkpointer = PostgresSaver(_pool)
    # setup()은 최초 1회만 필요
else:
    # 환경변수 미설정 시 InMemorySaver 폴백 (노트북 개발용)
    from langgraph.checkpoint.memory import InMemorySaver
    checkpointer = InMemorySaver()

# === 그래프 컴파일 (checkpointer 포함) ===
multi_agent_app = supervisor.compile(checkpointer=checkpointer)

# === RunnableLambda 래퍼 — custom_inputs.thread_id → configurable 매핑 ===
# mlflow.langchain.log_model()은 LangChain Runnable 객체를 기대.
# RunnableLambda로 감싸서 custom_inputs.thread_id를 config.configurable에 주입.
from langchain_core.runnables import RunnableLambda


def _invoke_with_thread_id(input_data: dict, config: dict = None) -> dict:
    """MLflow 서빙에서 custom_inputs.thread_id → configurable.thread_id 매핑"""
    # custom_inputs에서 thread_id 추출 후 input에서 제거
    custom_inputs = input_data.pop("custom_inputs", None) or {}
    if isinstance(custom_inputs, str):
        import json as _json
        custom_inputs = _json.loads(custom_inputs) if custom_inputs else {}
    thread_id = custom_inputs.get("thread_id", "default") if isinstance(custom_inputs, dict) else "default"

    # config에 thread_id 주입
    config = dict(config) if config else {}
    config.setdefault("configurable", {})
    config["configurable"]["thread_id"] = thread_id
    config.setdefault("recursion_limit", 20)

    return multi_agent_app.invoke(input_data, config=config)


# models-from-code: MLflow가 이 파일을 로드할 때 모델로 인식
mlflow.models.set_model(RunnableLambda(_invoke_with_thread_id))

