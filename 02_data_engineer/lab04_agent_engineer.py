# Databricks notebook source
# MAGIC %md
# MAGIC #14장 | 데이터 엔지니어링 에이전트 — 파이프라인을 스스로 관리하다
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##14.1 에이전트 도구 구성
# MAGIC

# COMMAND ----------

# MAGIC %pip install --upgrade "langchain>=1.0" "langgraph>=1.1.0" "databricks-langchain>=0.19.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain.tools import tool
from databricks.sdk import WorkspaceClient


w = WorkspaceClient()

@tool
def get_pipeline_status(pipeline_id: str) -> str:
    "파이프라인의 현재 상태를 반환합니다."
    try:
        info = w.pipelines.get(pipeline_id=pipeline_id)
        return (f"파이프라인 '{info.name}': 상태={info.state}, "
                f"마지막 업데이트={info.last_modified}")
    except Exception as e:
        return f"오류: {e}"

@tool
def get_pipeline_errors(pipeline_id: str) -> str:
    "파이프라인의 최근 오류 이벤트를 반환합니다."
    try:
        events = list(w.pipelines.list_pipeline_events(
            pipeline_id=pipeline_id, max_results=10
        ))
        errors = [e for e in events if e.level == "ERROR"]
        if not errors:
            return "최근 오류 없음"
        return "\n".join(f"[{e.timestamp}] {e.message}" for e in errors[:5])
    except Exception as e:
        return f"오류: {e}"

@tool
def restart_pipeline(pipeline_id: str) -> str:
    "파이프라인을 증분 재시작합니다."
    try:
        update = w.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=False)
        return f"재시작 성공. 업데이트 ID: {update.update_id}"
    except Exception as e:
        return f"재시작 실패: {e}"

@tool
def send_alert(message: str) -> str:
    "Slack 또는 이메일로 알림을 발송합니다."
    print(f"[알림 발송] {message}")
    return f"알림 발송 완료: {message[:50]}..."

tools = [get_pipeline_status, get_pipeline_errors, restart_pipeline, send_alert]

# 에이전트 실행 (LangGraph)
system_message = (
    "당신은 Databricks DE 전문가입니다. 파이프라인 상태를 진단하고 해결합니다.\n"
    "문제 발견 시: 원인 파악 → 자동 수복 시도 → 불가 시 알림 발송\n"
    "분석 결과는 한국어로 간결하게 요약해 주세요."
)

llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct", 
    temperature=0.0
)

agent = create_agent(model=llm, tools=tools, system_prompt=system_message)
config = {"recursion_limit": 20}

result = agent.invoke(
    {"messages": [("human", "파이프라인 'pipeline-abc123' 상태를 확인하고, 문제가 있으면 재시작하세요.")]},
    config=config,
)

print(result["messages"][-1].content)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 14.2 실습 Lab — 엔지니어링 에이전트 구현

# COMMAND ----------

from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain.tools import tool
from databricks.sdk import WorkspaceClient
import json

w = WorkspaceClient()

llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", temperature=0)

@tool
def get_pipeline_status(pipeline_id: str) -> str:
    """SDP 파이프라인 상태를 조회합니다."""
    try:
        pipeline = w.pipelines.get(pipeline_id)
        return json.dumps({
            "state": pipeline.state.value,
            "last_modified": str(pipeline.last_modified),
            "health": pipeline.health.value if pipeline.health else "UNKNOWN"
        }, ensure_ascii=False)
    except Exception as e:
        return f"오류: {e}"

@tool
def get_job_run_history(job_id: str, limit: int = 5) -> str:
    """최근 Job 실행 이력을 조회합니다."""
    runs = w.jobs.list_runs(job_id=int(job_id), limit=limit)
    results = []
    for run in runs:
        results.append({
            "run_id": run.run_id,
            "state": run.state.result_state.value if run.state.result_state else "RUNNING",
            "start_time": str(run.start_time),
            "duration_ms": run.run_duration
        })
    return json.dumps(results, ensure_ascii=False, indent=2)

@tool
def restart_pipeline(pipeline_id: str) -> str:
    """실패한 파이프라인을 재시작합니다."""
    w.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=False)
    return f"파이프라인 {pipeline_id} 재시작 완료"

@tool
def send_alert(message: str) -> str:
    "Slack 또는 이메일로 알림을 발송합니다."
    print(f"[알림 발송] {message}")
    return f"알림 발송 완료: {message[:50]}..."

tools = [get_pipeline_status, get_job_run_history, restart_pipeline, send_alert]

print("엔지니어링 에이전트 준비 완료")

system_message = (
    "당신은 Databricks DE 전문가입니다. 파이프라인 상태를 진단하고 해결합니다.\n"
    "문제 발견 시: 원인 파악 → 자동 수복 시도 → 불가 시 알림 발송\n"
    "분석 결과는 한국어로 간결하게 요약해 주세요."
)

llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct", 
    temperature=0.0
)

agent = create_agent(model=llm, tools=tools, system_prompt=system_message)
config = {"recursion_limit": 20}

result = agent.invoke(
    {"messages": [("human", "파이프라인 '187f7de1-0a72-481a-bb64-3fe708c8b161' 상태를 확인하고 알려주세요.")]},
    config=config,
)

print(result["messages"][-1].content)


# COMMAND ----------

def run_de_agent(question: str) -> str:
    result = de_agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 15},
    )
    answer = result["messages"][-1].content
    print(f"\n질문: {question}\n\n답변:\n{answer}\n{'='*60}")
    return answer

# COMMAND ----------

# 파이프라인 전체 상태 점검
run_de_agent("현재 smartfactory 데이터 파이프라인 상태를 점검하고, 문제가 있으면 원인을 분석해주세요.")

# COMMAND ----------

# 데이터 신선도 + 품질 종합 점검
run_de_agent("""
smartfactory.processed.sensor_clean 테이블의
1) 데이터 신선도 (마지막 업데이트 시점)
2) 데이터 품질 (NULL, 이상값)
3) 최근 7일 레코드 수 트렌드
를 분석하고 조치 방안을 제시해주세요.
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC - **15~16장**: `03_ml_engineer/01_lab01_feature_store.py`