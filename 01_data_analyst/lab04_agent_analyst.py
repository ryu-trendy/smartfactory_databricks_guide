# Databricks notebook source
# MAGIC %md
# MAGIC # 9장: SQL 분석 에이전트
# MAGIC
# MAGIC LangChain ReAct 패턴으로 자율 SQL 분석 에이전트를 구축합니다.
# MAGIC - **도구**: SQL 실행, 스키마 조회, OEE 계산
# MAGIC - **모델**: Foundation Models API (Llama 3.3 70B)
# MAGIC - **패턴**: ReAct (추론 → 행동 → 관찰 반복, LangGraph 기반)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.1.1 schema_registry 설계
# MAGIC

# COMMAND ----------

# 스키마 레지스트리 정의 (에이전트의 컨텍스트)
schema_registry = {
    "smartfactory.analytics.oee_daily": {
        "description": "일별 설비별 OEE 집계 데이터 (CLUSTER BY equipment_id, production_date)",
        "columns": {
            "equipment_id":     "설비 ID (EQ001~EQ050, STRING)",
            "line_id":          "라인 ID (LINE01~LINE10, STRING)",
            "equipment_type":   "설비 유형 (CNC/AOI/ROBOT/PRESS/CONVEYOR, STRING)",
            "production_date":  "생산 날짜 (STRING, 'YYYY-MM-DD' 형식)",
            "oee_pct":          "OEE 백분률 (0~100, DOUBLE)",
            "availability_pct": "가용률 백분률 (0~100, DOUBLE)",
            "performance_pct":  "성능률 백분률 (0~100, DOUBLE)",
            "quality_pct":      "품질률 백분률 (0~100, DOUBLE)",
            "availability":     "가용률 비율 (0~1, DOUBLE)",
            "performance":      "성능률 비율 (0~1, DOUBLE)",
            "quality":          "품질률 비율 (0~1, DOUBLE)",
            "total_actual_qty": "실제 생산량 합계 (BIGINT)",
            "updated_at":       "레코드 갱신 시각 (TIMESTAMP)",
        },
        "example": "SELECT line_id, ROUND(AVG(oee_pct), 1) AS avg_oee FROM smartfactory.analytics.oee_daily WHERE production_date >= CAST(DATEADD(DAY, -7, CURRENT_DATE()) AS STRING) GROUP BY line_id ORDER BY avg_oee",
    },
    "smartfactory.processed.sensor_clean": {
        "description": "정제된 설비 센서 데이터 (10초 간격, Z-score 이상치 탐지 포함)",
        "columns": {
            "event_id":        "이벤트 고유 ID (STRING)",
            "equipment_id":    "설비 ID (STRING)",
            "line_id":         "라인 ID (STRING)",
            "equipment_type":  "설비 유형 (STRING)",
            "event_time":      "이벤트 시각 (TIMESTAMP)",
            "temperature_c":   "온도 섭씨 (DOUBLE)",
            "vibration_ms2":   "진동 RMS mm/s (DOUBLE)",
            "pressure_bar":    "압력 bar (DOUBLE)",
            "rpm":             "회전수 (DOUBLE)",
            "quality_score":   "품질 점수 (DOUBLE)",
            "shift":           "근무 교대 (STRING)",
            "temp_zscore":     "온도 Z-score (DOUBLE)",
            "is_anomaly":      "이상치 여부 (BOOLEAN)",
        },
        "example": "SELECT equipment_id, event_time, temperature_c, vibration_ms2 FROM smartfactory.processed.sensor_clean WHERE is_anomaly = true AND event_time >= DATEADD(DAY, -7, CURRENT_DATE()) ORDER BY event_time DESC LIMIT 20",
    },
}
def get_schema_context() -> str:
    lines = ["사용 가능한 테이블 목록:"]
    for tbl, info in schema_registry.items():
        lines.append(f"\n[{tbl}]")
        lines.append(f"  설명: {info['description']}")
        lines.append("  컬럼:")
        for col, desc in info["columns"].items():
            lines.append(f"    - {col}: {desc}")
        if "example" in info:
            lines.append(f"  예시: {info['example']}")
    return "\n".join(lines)
get_schema_context()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.1.2 SQL 에이전트 도구 구현
# MAGIC

# COMMAND ----------

# MAGIC %pip install --upgrade "langchain>=1.0" "langgraph>=1.1.0" "databricks-langchain>=0.19.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# – restartPython() 의 사용으로 인해 변수명이 초기화 되므로 schema_registry 재정의가 필요합니다.
schema_registry = {
    "smartfactory.analytics.oee_daily": {
        "description": "일별 설비별 OEE 집계 데이터 (CLUSTER BY equipment_id, production_date)",
        "columns": {
            "equipment_id":     "설비 ID (EQ001~EQ050, STRING)",
            "line_id":          "라인 ID (LINE01~LINE10, STRING)",
            "equipment_type":   "설비 유형 (CNC/AOI/ROBOT/PRESS/CONVEYOR, STRING)",
            "production_date":  "생산 날짜 (STRING, 'YYYY-MM-DD' 형식)",
            "oee_pct":          "OEE 백분률 (0~100, DOUBLE)",
            "availability_pct": "가용률 백분률 (0~100, DOUBLE)",
            "performance_pct":  "성능률 백분률 (0~100, DOUBLE)",
            "quality_pct":      "품질률 백분률 (0~100, DOUBLE)",
            "availability":     "가용률 비율 (0~1, DOUBLE)",
            "performance":      "성능률 비율 (0~1, DOUBLE)",
            "quality":          "품질률 비율 (0~1, DOUBLE)",
            "total_actual_qty": "실제 생산량 합계 (BIGINT)",
            "updated_at":       "레코드 갱신 시각 (TIMESTAMP)",
        },
        "example": "SELECT line_id, ROUND(AVG(oee_pct), 1) AS avg_oee FROM smartfactory.analytics.oee_daily WHERE production_date >= CAST(DATEADD(DAY, -7, CURRENT_DATE()) AS STRING) GROUP BY line_id ORDER BY avg_oee",
    },
    "smartfactory.processed.sensor_clean": {
        "description": "정제된 설비 센서 데이터 (10초 간격, Z-score 이상치 탐지 포함)",
        "columns": {
            "event_id":        "이벤트 고유 ID (STRING)",
            "equipment_id":    "설비 ID (STRING)",
            "line_id":         "라인 ID (STRING)",
            "equipment_type":  "설비 유형 (STRING)",
            "event_time":      "이벤트 시각 (TIMESTAMP)",
            "temperature_c":   "온도 섭씨 (DOUBLE)",
            "vibration_ms2":   "진동 RMS mm/s (DOUBLE)",
            "pressure_bar":    "압력 bar (DOUBLE)",
            "rpm":             "회전수 (DOUBLE)",
            "quality_score":   "품질 점수 (DOUBLE)",
            "shift":           "근무 교대 (STRING)",
            "temp_zscore":     "온도 Z-score (DOUBLE)",
            "is_anomaly":      "이상치 여부 (BOOLEAN)",
        },
        "example": "SELECT equipment_id, event_time, temperature_c, vibration_ms2 FROM smartfactory.processed.sensor_clean WHERE is_anomaly = true AND event_time >= DATEADD(DAY, -7, CURRENT_DATE()) ORDER BY event_time DESC LIMIT 20",
    },
}
def get_schema_context() -> str:
    lines = ["사용 가능한 테이블 목록:"]
    for tbl, info in schema_registry.items():
        lines.append(f"\n[{tbl}]")
        lines.append(f"  설명: {info['description']}")
        lines.append("  컬럼:")
        for col, desc in info["columns"].items():
            lines.append(f"    - {col}: {desc}")
        if "example" in info:
            lines.append(f"  예시: {info['example']}")
    return "\n".join(lines)

# COMMAND ----------

from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain.tools import tool

# 도구 1: SQL 실행
@tool
def execute_sql(query: str) -> str:
    "주어진 SQL 쿼리를 실행하고 결과를 문자열로 반환합니다."
    try:
        df = spark.sql(query)
        rows = df.limit(50).collect()
        if not rows:
            return "결과 없음"
        cols = df.columns
        result_lines = [" | ".join(cols)]
        result_lines.append("-" * 60)
        for row in rows[:10]:
            result_lines.append(" | ".join(str(v) for v in row))
        if len(rows) > 10:
            result_lines.append(f"... (총 {len(rows)}행 중 10행 표시)")
        return "\n".join(result_lines)
    except Exception as e:
        return f"SQL 오류: {e}"


# 도구 2: 테이블 스키마 조회
@tool
def get_table_schema(table_name: str) -> str:
    "특정 테이블의 컬럼 정보를 반환합니다."
    if table_name in schema_registry:
        info = schema_registry[table_name]
        return f"테이블: {table_name}\n설명: {info['description']}\n컬럼: {info['columns']}"
    try:
        df = spark.sql(f"DESCRIBE {table_name}")
        return df.toPandas().to_string()
    except Exception as e:
        return f"오류: {e}"


# 에이전트 생성 및 실행 (Langchain)
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    temperature=0.0,
)
tools = [execute_sql, get_table_schema]


system_message = (
    "당신은 스마트팩토리 데이터 분석 전문가입니다.\n\n"
    + get_schema_context()
    + "\n\n분석 결과는 한국어로 명확하게 요약해 주세요."
)
agent = create_agent(model=llm, tools=tools, system_prompt=system_message)


config = {"recursion_limit": 20}  # 최대 반복 횟수 제어 (무한 루프 방지)
result = agent.invoke(
    {"messages": [("human", "지난 7일간 LINE03의 일별 OEE 추이를 분석해 주세요.")]},
    config=config,
)
print(result["messages"][-1].content)

# COMMAND ----------

# MAGIC %md
# MAGIC #9.2 실습 Lab — SQL 분석 에이전트 배포
# MAGIC

# COMMAND ----------

# Lab: analyst 에이전트 전체 구현
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain.tools import tool
import json

# LLM 설정
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    temperature=0,
    max_tokens=2048
)

# 스키마 레지스트리
SCHEMA_REGISTRY = {
    "smartfactory.analytics.oee_daily": {
        "description": "일별 설비별 OEE 집계 데이터 (CLUSTER BY equipment_id, production_date)",
        "columns": {
            "equipment_id":     "설비 ID (EQ001~EQ050, STRING)",
            "line_id":          "라인 ID (LINE01~LINE10, STRING)",
            "equipment_type":   "설비 유형 (CNC/AOI/ROBOT/PRESS/CONVEYOR, STRING)",
            "production_date":  "생산 날짜 (STRING, 'YYYY-MM-DD' 형식)",
            "oee_pct":          "OEE 백분률 (0~100, DOUBLE)",
            "availability_pct": "가용률 백분률 (0~100, DOUBLE)",
            "performance_pct":  "성능률 백분률 (0~100, DOUBLE)",
            "quality_pct":      "품질률 백분률 (0~100, DOUBLE)",
            "total_actual_qty": "실제 생산량 합계 (BIGINT)",
        },
        "example": "SELECT equipment_id, ROUND(AVG(oee_pct), 1) AS avg_oee FROM smartfactory.analytics.oee_daily WHERE production_date >= CAST(DATE_SUB(CURRENT_DATE(), 7) AS STRING) GROUP BY equipment_id ORDER BY avg_oee LIMIT 5",
    },
    "smartfactory.processed.production_logs": {
        "description": "생산 실적 로그 (production_date는 STRING 'YYYY-MM-DD' 형식)",
        "columns": {
            "log_id":           "로그 고유 ID (STRING)",
            "equipment_id":     "설비 ID (STRING)",
            "line_id":          "라인 ID (STRING)",
            "shift":            "근무 교대 (Day/Night, STRING)",
            "production_date":  "생산 날짜 (STRING, 'YYYY-MM-DD' 형식)",
            "planned_qty":      "계획 생산량 (INT)",
            "actual_qty":       "실제 생산량 (INT)",
            "defect_qty":       "불량 수량 (INT)",
            "downtime_min":     "비가동 시간 분 (INT)",
        },
        "example": "SELECT equipment_id, SUM(actual_qty) AS total_qty, SUM(defect_qty) AS total_defects FROM smartfactory.processed.production_logs WHERE production_date >= CAST(DATE_SUB(CURRENT_DATE(), 7) AS STRING) GROUP BY equipment_id ORDER BY total_defects DESC LIMIT 5",
    }
}


@tool
def execute_sql(query: str) -> str:
    """SQL 쿼리를 실행하고 결과를 반환합니다."""
    try:
        result = spark.sql(query)
        pdf = result.limit(20).toPandas()
        return pdf.to_string(index=False)
    except Exception as e:
        return f"오류: {str(e)}"


@tool
def get_schema_info(table_name: str) -> str:
    """테이블 스키마 정보를 반환합니다. 전체 경로(catalog.schema.table) 또는 테이블명으로 검색합니다."""
    # 전체 경로 직접 매치
    info = SCHEMA_REGISTRY.get(table_name)
    # 단축명 폴백 검색
    if not info:
        for key, val in SCHEMA_REGISTRY.items():
            if key.endswith(f".{table_name}") or table_name in key:
                info = val
                table_name = key
                break
    if info:
        return json.dumps({"table": table_name, **info}, ensure_ascii=False, indent=2)
    return f"{table_name} 테이블 없음. 사용 가능: {list(SCHEMA_REGISTRY.keys())}"


# 에이전트 생성 (LangChain v1 create_agent)
# 테이블 전체 경로 리스트 생성 (에이전트가 정확한 경로 사용하도록)
def get_schema_context_lab() -> str:
    lines = []
    for tbl, info in SCHEMA_REGISTRY.items():
        lines.append(f"\n[{tbl}]")
        lines.append(f"  설명: {info['description']}")
        lines.append("  컬럼:")
        for col, desc in info["columns"].items():
            lines.append(f"    - {col}: {desc}")
        if "example" in info:
            lines.append(f"  예시 SQL: {info['example']}")
    return "\n".join(lines)


table_context = get_schema_context_lab()
system_message = (
    "당신은 스마트팩토리 코리아의 데이터 분석 전문 AI 에이전트입니다.\n\n"
    f"사용 가능한 테이블:\n{table_context}\n\n"
    "⚠️ 필수 규칙:\n"
    "1. SQL에서 테이블명은 반드시 전체 경로(catalog.schema.table)를 사용하세요.\n"
    "2. production_date 컬럼은 STRING 타입이므로 날짜 비교 시 CAST(DATE_SUB(CURRENT_DATE(), 7) AS STRING)을 사용하세요. (DATEADD 사용 금지)\n"
    "3. 항상 SQL 실행 전에 get_schema_info로 스키마를 확인하세요.\n"
    "4. 분석 결과는 한국어로 명확하게 요약해 주세요."
)
tools = [execute_sql, get_schema_info]
agent = create_agent(model=llm, tools=tools, system_prompt=system_message)


# 테스트
config = {"recursion_limit": 20}
result = agent.invoke(
    {"messages": [("human", "지난주 OEE가 가장 낮은 설비 3개는?")]},
    config=config,
)
print(result["messages"][-1].content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC - **10장**: `02_data_engineer/01_lab01_unity_catalog.py` (Unity Catalog 거버넌스)