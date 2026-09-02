# Databricks notebook source
# MAGIC %md
# MAGIC # 24장 | Databricks AI Agent Framework — 정비 AI 어시스턴트 구축

# COMMAND ----------

# MAGIC %md
# MAGIC ## 24.1 에이전트 도구 구성

# COMMAND ----------

# MAGIC %pip install --upgrade "langchain>=1.0" "langgraph>=1.1.0" "databricks-langchain>=0.19.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain.tools import tool
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool


# === LLM 설정 ===
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct",
                     temperature=0.1, max_tokens=800)


# === 1. 매뉴얼 검색 도구 (VectorSearchRetrieverTool — 다국어 임베딩) ===
# AI Search 인덱스가 qwen3-embedding-0-6b(다국어)로 구성되어
# 한국어/영어 모두 직접 검색 가능합니다. 번역 불필요.
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


def get_sensor_context(equipment_id: str, anomaly_only: bool = False) -> str:
    """센서 데이터 조회 (sensor_clean 테이블)"""
    anomaly_filter = "AND is_anomaly = true" if anomaly_only else ""
    sensor_df = spark.sql(f'''
        SELECT equipment_id, equipment_type, temperature_c, vibration_ms2,
               pressure_bar, rpm, is_anomaly, event_time
        FROM smartfactory.processed.sensor_clean
        WHERE equipment_id = '{equipment_id}' {anomaly_filter}
        ORDER BY event_time DESC LIMIT 1
    ''')
    if sensor_df.isEmpty():
        return "현재 센서 데이터 없음"
    row = sensor_df.first()
    anomaly_status = "⚠️ ANOMALY 감지" if row['is_anomaly'] else "✅ 정상"
    return (
        f"현재 설비 상태 ({row['equipment_id']} / {row['equipment_type']}):\n"
        f"  시각: {row['event_time']}\n"
        f"  온도: {row['temperature_c']:.1f}°C | 진동: {row['vibration_ms2']:.2f}mm/s\n"
        f"  압력: {row['pressure_bar']:.1f}bar | RPM: {row['rpm']:.0f}\n"
        f"  상태: {anomaly_status}"
    )


# === 에이전트 도구 정의 ===


# --- 2. 센서 데이터 조회 도구 (24장 get_sensor_context 재사용) ---
@tool
def get_realtime_sensor_data(equipment_id: str) -> str:
    "설비의 최신 센서 데이터를 조회합니다. (temperature, vibration, pressure, rpm, anomaly 여부)"
    return get_sensor_context(equipment_id, anomaly_only=False)


# --- 3. 고장 확률 예측 도구 (smartfactory-pdm 엔드포인트 연동) ---
@tool
def predict_failure_probability(equipment_id: str) -> str:
    """48시간 내 고장 확률을 ML 모델(smartfactory-pdm)로 예측합니다."""
    import requests as _requests


    # 1. 최신 센서 데이터 조회 (6개 피쳐)
    sensor_df = spark.sql(f'''
        SELECT temperature_c, vibration_ms2, pressure_bar, rpm, quality_score, temp_zscore
        FROM smartfactory.processed.sensor_clean
        WHERE equipment_id = '{equipment_id}'
        ORDER BY event_time DESC LIMIT 1
    ''').toPandas()


    if sensor_df.empty:
        return f"{equipment_id}: 센서 데이터 없음 → 고장 예측 불가"


    features = sensor_df.values.tolist()  # [[temp, vib, press, rpm, quality, zscore]]


    # 2. smartfactory-pdm 엔드포인트 호출
    _host = spark.conf.get("spark.databricks.workspaceUrl")
    _token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    pdm_url = f"https://{_host}/serving-endpoints/smartfactory-pdm/invocations"


    resp = _requests.post(
        pdm_url,
        headers={"Authorization": f"Bearer {_token}", "Content-Type": "application/json"},
        json={"inputs": features},
        timeout=30,
    )


    if resp.status_code != 200:
        return f"{equipment_id}: PDM 엔드포인트 오류 (HTTP {resp.status_code})"


    prediction = resp.json().get("predictions", [0])[0]


    # 3. 결과 해석
    row = sensor_df.iloc[0]
    if prediction == 1:
        risk, rec = "HIGH ⚠️", "즉시 정비 필요 (72시간 내 점검 필수)"
    else:
        risk, rec = "LOW ✅", "정상 운전 유지, 정기 모니터링 지속"


    return (
        f"{equipment_id} 고장예측: {risk} (ML 모델 판정)\n"
        f"  입력 피쳐: temp={row['temperature_c']:.1f}°C, vib={row['vibration_ms2']:.2f}mm/s, "
        f"press={row['pressure_bar']:.1f}bar, rpm={row['rpm']:.0f}\n"
        f"  권고: {rec}"
    )


# --- 4. 작업 지시서 생성 도구 (Mock) ---
@tool
def create_work_order(equipment_id: str, priority: str, description: str) -> str:
    "정비 작업 지시서를 생성합니다."
    import uuid
    work_order_id = f"WO-{uuid.uuid4().hex[:8].upper()}"
    # TODO: 실제 테이블 연동 예정 (smartfactory.processed.work_orders)
    return f"작업 지시 생성 완료: {work_order_id} (설비: {equipment_id}, 우선순위: {priority})"


MAINTENANCE_TOOLS = [
    search_maintenance_manual,
    get_realtime_sensor_data,
    predict_failure_probability,
    create_work_order,
]


# COMMAND ----------

# MAGIC %md
# MAGIC ## 24.2 에이전트 실행

# COMMAND ----------

from langchain.agents import create_agent


# --- 에이전트 시스템 프롬프트 ---
system_message = (
    "당신은 스마트팩토리 코리아의 정비 AI 어시스턴트입니다.\n"
    "설비 이상 신호를 감지하면 다음 순서로 처리하세요:\n"
    "1. 실시간 센서 데이터 확인 (get_realtime_sensor_data)\n"
    "2. 고장 확률 예측 (predict_failure_probability)\n"
    "3. OSHA/NIOSH 안전 매뉴얼 검색 (search_maintenance_manual)\n"
    "4. 위험도가 MEDIUM 이상이면 작업 지시서 자동 생성 (create_work_order)\n"
    "5. 정비원에게 단계별 조치 안내 (한국어로 종합 요약)\n\n"
    "⚠️ 필수 규칙:\n"
    "- ANOMALY 감지 시 LOTO 절차를 반드시 언급하세요\n"
    "- 고온·고진동 상태에서는 잔여 에너지 해소의 중요성을 강조하세요\n"
    "- 최종 답변은 한국어로 명확하게 요약하세요"
)


# --- 에이전트 생성 (create_agent 패턴) ---
agent_llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", temperature=0.0)
agent = create_agent(model=agent_llm, tools=MAINTENANCE_TOOLS, system_prompt=system_message)


# --- 에이전트 실행 ---
config = {"recursion_limit": 20}  # 무한 루프 방지
result = agent.invoke(
    {"messages": [("human", "EQ005 CNC 설비에서 anomaly가 감지되었습니다. 현재 상태를 확인하고 정비 전 안전 절차와 필요한 조치를 안내해주세요.")]},
    config=config,
)
print("\n=== 에이전트 최종 답변 ===")
print(result["messages"][-1].content)


# COMMAND ----------

# MAGIC %md
# MAGIC ##24.3 유지보수 어시스턴트 에이전트

# COMMAND ----------

from langchain.agents import create_agent


# --- 에이전트 생성 ---
lab_system_prompt = (
    "당신은 스마트팩토리 코리아의 정비 AI 어시스턴트입니다.\n"
    "설비 이상 신호를 감지하면 다음 순서로 처리하세요:\n"
    "1. 실시간 센서 데이터 확인\n"
    "2. 고장 확률 예측\n"
    "3. OSHA/NIOSH 안전 매뉴얼 검색\n"
    "4. 위험도가 MEDIUM 이상이면 작업 지시서 생성\n"
    "5. 정비원에게 단계별 조치 안내 (한국어로)\n\n"
    "⚠️ ANOMALY 감지 시 LOTO 절차 반드시 언급. 고온 상태에서는 잔여 에너지 해소 강조."
)


lab_agent = create_agent(
    model=ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", temperature=0.0),
    tools=MAINTENANCE_TOOLS,
    system_prompt=lab_system_prompt,
)


# --- 테스트 시나리오 ---
lab_scenarios = [
    "EQ005 CNC 설비에서 anomaly가 감지되었습니다. 현재 상태를 확인하고 정비 권고사항을 알려주세요.",
    "EQ007 CNC의 안전장치(guard) 점검이 필요합니다. OSHA 기준 요구사항과 절차를 안내해주세요.",
    "EQ005 CNC 설비의 48시간 내 고장 확률을 예측해주세요. 위험도가 높으면 작업 지시서도 생성해주세요.",
]


config = {"recursion_limit": 20}
for scenario in lab_scenarios:
    print(f"\n{'='*60}")
    print(f"🔍 질문: {scenario}")
    print(f"{'-'*60}")
    result = lab_agent.invoke({"messages": [("human", scenario)]}, config=config)
    print(result["messages"][-1].content)


print(f"\n{'='*60}")
print("✅ Lab 25.3 유지보수 에이전트 실습 완료")
