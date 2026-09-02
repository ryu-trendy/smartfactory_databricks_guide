# Databricks notebook source
# MAGIC %md
# MAGIC # 25장 | AI 에이전트 배포 — 프로덕션에서 에이전트 운영하기

# COMMAND ----------

# MAGIC %md
# MAGIC ## 25.1 MLflow로 에이전트 패키징

# COMMAND ----------

# MAGIC %pip install --upgrade "databricks-sdk" "langchain>=1.0" "langgraph>=1.1.0" "databricks-langchain>=0.19.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
from databricks.sdk import WorkspaceClient


# --- SQL Warehouse ID 설정 (3장에서 생성한 Warehouse ID)---
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

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
import pandas as pd


class MaintenanceAgentWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # 에이전트 초기화 (로드 시 1회)
        from databricks.sdk import WorkspaceClient
        from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool 
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.tools import tool
        from langchain.agents import create_agent


        # SDK 클라이언트 (서빙 환경에서 자동 인증)
        ws = WorkspaceClient()
        warehouse_id = os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID")


        llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", temperature=0.0, max_tokens=800)


       # VectorSearchRetrieverTool (다국어 임베딩: qwen3-embedding-0-6b — 한국어/영어 직접 검색 가능)
        search_maintenance_manual = VectorSearchRetrieverTool(
            index_name="smartfactory.ai.manual_index",
            num_results=3,
            columns=["chunk_id", "doc_id", "title", "equipment_type", "doc_type"],
            tool_name="search_maintenance_manual",
            tool_description=(
                "Search OSHA/NIOSH safety manuals for maintenance procedures, "
                "LOTO lockout/tagout steps, hazardous energy control, and machine guarding requirements. "
                "Supports Korean and English queries (multilingual embedding model)."
            ),
        )


        @tool
        def get_realtime_sensor_data(equipment_id: str) -> str:
            "설비 센서 데이터 조회 (SDK Statement Execution API)"
            stmt = ws.statement_execution.execute_statement(
                warehouse_id=warehouse_id,
                statement=(
                    f"SELECT equipment_id, equipment_type, temperature_c, vibration_ms2, "
                    f"pressure_bar, rpm, is_anomaly, CAST(event_time AS STRING) "
                    f"FROM smartfactory.processed.sensor_clean "
                    f"WHERE equipment_id = '{equipment_id}' ORDER BY event_time DESC LIMIT 1"
                ),
                wait_timeout="30s",
            )
            if not stmt.result or not stmt.result.data_array:
                return f"{equipment_id}: 센서 데이터 없음"
            r = stmt.result.data_array[0]
            status = "⚠️ ANOMALY" if str(r[6]).lower() == "true" else "✅ 정상"
            return f"{r[0]}({r[1]}): 온도{r[2]}°C 진동{r[3]}mm/s 압력{r[4]}bar RPM{r[5]} [{status}]"


        @tool
        def predict_failure_probability(equipment_id: str) -> str:
            """48시간 내 고장 확률을 ML 모델(smartfactory-pdm)로 예측합니다."""
            import requests as _requests


            # 1. 최신 센서 데이터 조회 (SDK Statement Execution API)
            stmt = ws.statement_execution.execute_statement(
                warehouse_id=warehouse_id,
                statement=(
                    f"SELECT temperature_c, vibration_ms2, pressure_bar, rpm, "
                    f"quality_score, temp_zscore "
                    f"FROM smartfactory.processed.sensor_clean "
                    f"WHERE equipment_id = '{equipment_id}' "
                    f"ORDER BY event_time DESC LIMIT 1"
                ),
                wait_timeout="30s",
            )
            if not stmt.result or not stmt.result.data_array:
                return f"{equipment_id}: 센서 데이터 없음 → 고장 예측 불가"
            row = stmt.result.data_array[0]
            features = [[float(v) for v in row]]


            # 2. smartfactory-pdm 엔드포인트 호출 (SDK 인증 사용)
            resp = _requests.post(
                f"{serving_host}/serving-endpoints/smartfactory-pdm/invocations",
                headers={"Authorization": f"Bearer {serving_token}",
                         "Content-Type": "application/json"},
                json={"inputs": features},
                timeout=30,
            )


            if resp.status_code != 200:
                return f"{equipment_id}: PDM 엔드포인트 오류 (HTTP {resp.status_code})"


            prediction = resp.json().get("predictions", [0])[0]
            risk = "HIGH ⚠️" if prediction == 1 else "LOW ✅"
            action = "72시간 내 점검 필수" if prediction == 1 else "정상 운전 유지"
            return (
                f"{equipment_id} 고장예측: {risk} (ML 모델 판정)\n"
                f"  입력: temp={row[0]}°C, vib={row[1]}mm/s, press={row[2]}bar, rpm={row[3]}\n"
                f"  권장: {action}"
            )


        @tool
        def create_work_order(equipment_id: str, priority: str, description: str) -> str:
            "작업 지시서 생성"
            import uuid
            return f"WO-{uuid.uuid4().hex[:8].upper()} 생성 ({equipment_id}, {priority})"


        self.system_prompt = (
            "정비 AI 어시스턴트. 순서: 1.센서확인 2.고장예측 3.매뉴얼검색 4.작업지시 5.한국어요약. "
            "search_maintenance_manual 호출 시 query는 반드시 영어로 작성. "
            "(예: '잠금 절차' → query='lockout tagout LOTO procedure') "
            "ANOMALY 시 LOTO 필수. 고온 시 잔여에너지해소 강조."
        )
        self.tools = [search_maintenance_manual, get_realtime_sensor_data,
                      predict_failure_probability, create_work_order]
        self.llm = llm
        self._create_agent()  # 자식 클래스에서 override 가능


    def _create_agent(self):
        """Agent 생성 — 자식 클래스에서 checkpointer 추가 시 override"""
        from langchain.agents import create_agent
        self.agent = create_agent(model=self.llm, tools=self.tools, system_prompt=self.system_prompt)


    def predict(self, context, model_input: pd.DataFrame) -> list:
        """각 요청마다 호출 — model_input['query'] 컬럼 처리"""
        config = {"recursion_limit": 20}
        results = []
        for _, row in model_input.iterrows():
            state = self.agent.invoke(
                {"messages": [("human", row["query"])]}, config=config
            )
            results.append(state["messages"][-1].content)
        return results


# MLflow에 에이전트 등록
# UC 등록 시 signature 필수 — 입력/출력 타입 명시
sample_input = pd.DataFrame({"query": ["EQ005 anomaly 감지. 정비 절차를 알려주세요."]})
sample_output = ["EQ005 온도 92.5°C, ANOMALY. LOTO 절차 수행 후 정비 필요..."]
signature = infer_signature(sample_input, sample_output)


username = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{username}/maintenance_agent")


with mlflow.start_run(run_name="maintenance_agent_v1"):
    mlflow.pyfunc.log_model(
        name="agent",
        python_model=MaintenanceAgentWrapper(),
        signature=signature,
        pip_requirements=["databricks-langchain>=0.19.0", "langchain>=1.0", "langgraph>=1.1.0", "databricks-sdk"],
        registered_model_name="smartfactory.ai.maintenance_agent",
    )
    print("✅ 에이전트 UC 등록 완료!")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 25.2 세션 관리 — PostgresSaver로 대화 맥락 유지
# MAGIC > 외부 PostgreSQL 인스턴스 사용을 위해 인프라팀의 데이터베이스 생성, 네트워크 연결, 보안 정책 및 방화벽 설정 등이 필요할 수 있습니다. <br>
# MAGIC > 반면 이 장에서는 별도의 인프라 구축 없이 독자가 직접 실습할 수 있도록 Neon Free Tier의 Serverless PostgreSQL을 사용합니다.<br>
# MAGIC > Cluster를 **Serverless** 로 교체하여 사용합니다. <br>
# MAGIC > 26장 실습 환경 안내 참조

# COMMAND ----------

# MAGIC %pip install "langchain>=1.0" "databricks-langchain>=0.19.0" "langgraph>=1.1.0,<1.2" "langgraph-checkpoint-postgres>=2.0.0,<3.0" "psycopg[binary]>=3.1" "psycopg-pool>=3.1"

# COMMAND ----------

# MAGIC %md
# MAGIC ### ⚠️ langgraph 버전 호환성 트러블슈팅
# MAGIC
# MAGIC | 증상 | 원인 | 해결 |
# MAGIC | --- | --- | --- |
# MAGIC | `ImportError: cannot import name 'ExecutionInfo' from 'langgraph.runtime'` | `langgraph` 1.2.x에서 내부 API가 제거/이동됨 | `langgraph>=1.1.0,<1.2` 로 상한 고정 |
# MAGIC | `ImportError: cannot import name 'StreamPart' from 'langgraph.types'` | `--force-reinstall` 후 1.2.x의 잔여 `.py` 파일이 남아 1.1.x `types.py`와 충돌 | 클린 환경에서 `%pip install`로 재설치 (force-reinstall 지양) |
# MAGIC | `databricks-sdk` 버전 경고 | `pip_requirements`에 `databricks-sdk>=0.118.0` 지정했으나 런타임 내장 버전이 낮음 | 런타임 코어 패키지는 `%pip install`로 업그레이드 불가 — 서빙 환경에서만 적용됨 (노트북 경고 무시 가능) |
# MAGIC
# MAGIC > **핵심**: `langgraph` 1.2.x는 `langgraph-checkpoint-postgres` 2.x와 내부 API 비호환. 반드시 `<1.2` 상한을 명시할 것.

# COMMAND ----------

import os
from databricks.sdk import WorkspaceClient
import psycopg
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

# ── 1. Neon Postgres 연결 정보 ──────────────────────────────────────────
# Neon Console (https://console.neon.tech) 에서:
#   1. 프로젝트 생성 (Free tier: 0.5GB 저장, 24/7 사용 가능)
#   2. Connection Details 에서 Connection String 복사
#   형식: postgresql://<user>:<password>@<host>/<dbname>?sslmode=require
# ───────────────────────────────────────────────────────────


# ⬇️ 여기에 Neon Connection String 입력
NEON_CONN_STRING = "postgresql://neondb_owner:....ap-southeast-1.aws.neon.tech/neondb?sslmode=require" # Connection String 복사 수정


# 프로덕션은 dbutils.secrets.get(scope="neon", key="conn_string") 사용 권장
# NEON_CONN_STRING = dbutils.secrets.get(scope="neon", key="conn_string")


# ── 2. PostgresSaver 테이블 생성 (autocommit 필수 — CREATE INDEX CONCURRENTLY) ──

# setup()은 CREATE INDEX CONCURRENTLY 사용 → autocommit 필수
with psycopg.connect(NEON_CONN_STRING, autocommit=True) as setup_conn:
    checkpointer = PostgresSaver(setup_conn)
    checkpointer.setup()


# 운영용 Connection Pool
pool = ConnectionPool(conninfo=NEON_CONN_STRING, min_size=1, max_size=5)
checkpointer = PostgresSaver(pool)


# ── 3. 연결 검증 ──────────────────────────────────────────────
with pool.connection() as conn:
    result = conn.execute("SELECT 1").fetchone()
    assert result == (1,), "DB 연결 실패"


print(f"   PostgresSaver checkpoint 테이블 생성 완료")


# ── 4. 환경변수 저장 (서빙에서 사용) ─────────
os.environ["NEON_CONN_STRING"] = NEON_CONN_STRING


# --- SQL Warehouse ID 설정 (3장에서 생성한 Warehouse ID)---

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

print(f"\n📝 서빙 시 필요한 환경변수:")
print(f"   NEON_CONN_STRING = {os.environ.get('NEON_CONN_STRING').strip()[:10]}...")
print(f"   DATABRICKS_SQL_WAREHOUSE_ID = {os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID")[:3]}...")

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from mlflow.models.resources import (
    DatabricksVectorSearchIndex, DatabricksSQLWarehouse,
    DatabricksTable, DatabricksServingEndpoint,
)
import pandas as pd
import os




class MaintenanceAgentPG(mlflow.pyfunc.PythonModel):
    """PostgresSaver(Lakebase) 기반 멀티턴 정비 AI 에이전트
    
    아키텍처:
    - load_context(): 1회 초기화 (LLM, 도구, PostgresSaver)
    - predict(): 매 요청마다 thread_id로 세션 관리
    - PostgresSaver: 모든 워커가 같은 DB 공유 → 세션 항상 유지
    """


    def load_context(self, context):
        from databricks.sdk import WorkspaceClient
        from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.tools import tool
        from langchain.agents import create_agent
        from psycopg_pool import ConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver
        import requests as _requests


        # --- SDK 클라이언트 ---
        ws = WorkspaceClient()
        warehouse_id = os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID")
        serving_host = ws.config.host.rstrip("/")
        serving_token = ws.config.token


        # --- PostgresSaver 초기화 (Neon Postgres) ---
        import psycopg
        conn_str = os.environ.get("NEON_CONN_STRING")
        # setup()은 CREATE INDEX CONCURRENTLY 사용 → autocommit 필수
        with psycopg.connect(conn_str, autocommit=True) as setup_conn:
            PostgresSaver(setup_conn).setup()
        # 운영용 pool: min_size=0 → 요청 시 신규 연결 생성 (Neon pooler 유휴 타임아웃 방지)
        self._pool = ConnectionPool(
            conninfo=conn_str,
            min_size=0,   # ← 유휴 연결 유지 안 함 (Neon이 끊어도 문제없음)
            max_size=5,
            timeout=30,
        )
        self._checkpointer = PostgresSaver(self._pool)


        # --- LLM ---
        llm = ChatDatabricks(
            endpoint="databricks-meta-llama-3-3-70b-instruct",
            temperature=0.0, max_tokens=800,
        )
       # --- VectorSearchRetrieverTool (다국어 임베딩: qwen3-embedding-0-6b) ---
        search_maintenance_manual = VectorSearchRetrieverTool(
            index_name="smartfactory.ai.manual_index",
            num_results=3,
            columns=["chunk_id", "doc_id", "title", "equipment_type", "doc_type"],
            tool_name="search_maintenance_manual",
            tool_description=(
                "Search OSHA/NIOSH safety manuals for maintenance procedures, "
                "LOTO lockout/tagout steps, hazardous energy control, and machine guarding requirements. "
                "Supports Korean and English queries (multilingual embedding model)."
            ),
        )


        # --- 도구 정의 ---
        @tool
        def get_realtime_sensor_data(equipment_id: str) -> str:
            "설비 센서 데이터 조회 (SDK Statement Execution API)"
            stmt = ws.statement_execution.execute_statement(
                warehouse_id=warehouse_id,
                statement=(
                    f"SELECT equipment_id, equipment_type, temperature_c, vibration_ms2, "
                    f"pressure_bar, rpm, is_anomaly, CAST(event_time AS STRING) "
                    f"FROM smartfactory.processed.sensor_clean "
                    f"WHERE equipment_id = '{equipment_id}' ORDER BY event_time DESC LIMIT 1"
                ),
                wait_timeout="30s",
            )
            if not stmt.result or not stmt.result.data_array:
                return f"{equipment_id}: 센서 데이터 없음"
            r = stmt.result.data_array[0]
            status = "⚠️ ANOMALY" if str(r[6]).lower() == "true" else "✅ 정상"
            return f"{r[0]}({r[1]}): 온도{r[2]}°C 진동{r[3]}mm/s 압력{r[4]}bar RPM{r[5]} [{status}]"


        @tool
        def predict_failure_probability(equipment_id: str) -> str:
            """48시간 내 고장 확률을 ML 모델(smartfactory-pdm)로 예측합니다."""
            stmt = ws.statement_execution.execute_statement(
                warehouse_id=warehouse_id,
                statement=(
                    f"SELECT temperature_c, vibration_ms2, pressure_bar, rpm, "
                    f"quality_score, temp_zscore "
                    f"FROM smartfactory.processed.sensor_clean "
                    f"WHERE equipment_id = '{equipment_id}' "
                    f"ORDER BY event_time DESC LIMIT 1"
                ),
                wait_timeout="30s",
            )
            if not stmt.result or not stmt.result.data_array:
                return f"{equipment_id}: 센서 데이터 없음 → 고장 예측 불가"
            row = stmt.result.data_array[0]
            features = [[float(v) for v in row]]


            _host = serving_host
            _token = serving_token
            resp = _requests.post(
                f"{_host}/serving-endpoints/smartfactory-pdm/invocations",
                headers={"Authorization": f"Bearer {_token}",
                         "Content-Type": "application/json"},
                json={"inputs": features},
                timeout=30,
            )
            if resp.status_code != 200:
                return f"{equipment_id}: PDM 엔드포인트 오류 (HTTP {resp.status_code})"


            prediction = resp.json().get("predictions", [0])[0]
            risk = "HIGH ⚠️" if prediction == 1 else "LOW ✅"
            action = "72시간 내 점검 필수" if prediction == 1 else "정상 운전 유지"
            return (
                f"{equipment_id} 고장예측: {risk} (ML 모델 판정)\n"
                f"  입력: temp={row[0]}°C, vib={row[1]}mm/s, press={row[2]}bar, rpm={row[3]}\n"
                f"  권장: {action}"
            )


        @tool
        def create_work_order(equipment_id: str, priority: str, description: str) -> str:
            "작업 지시서 생성"
            import uuid
            return f"WO-{uuid.uuid4().hex[:8].upper()} 생성 ({equipment_id}, {priority}): {description[:50]}"


       # --- 에이전트 생성 (PostgresSaver checkpointer) ---
        system_prompt = (
            "정비 AI 어시스턴트. 순서: 1.센서확인 2.고장예측 3.매뉴얼검색 4.작업지시 5.한국어요약. "
            "search_maintenance_manual 호출 시 한국어 또는 영어로 query 작성 가능 (다국어 임베딩). "
            "ANOMALY 시 LOTO 필수. 고온 시 잔여에너지해소 강조."
        )
        self.agent = create_agent(
            model=llm,
            tools=[search_maintenance_manual, get_realtime_sensor_data,
                   predict_failure_probability, create_work_order],
            system_prompt=system_prompt,
            checkpointer=self._checkpointer,  # ← PostgresSaver!
        )


    def predict(self, context, model_input, params=None):
        """매 요청 처리: ChatCompletion 형식 + thread_id 세션 관리"""
        import json as _json


        # --- 입력 파싱 (DataFrame / dict 모두 지원) ---
        if isinstance(model_input, pd.DataFrame):
            row = model_input.iloc[0]
            messages = row.get("messages", [{"role": "user", "content": str(row.get("query", ""))}])
            custom_inputs = row.get("custom_inputs", {})
        elif isinstance(model_input, dict):
            messages = model_input.get("messages", [])
            custom_inputs = model_input.get("custom_inputs", {})
        else:
            messages = [{"role": "user", "content": str(model_input)}]
            custom_inputs = {}


        # custom_inputs가 JSON 문자열인 경우 처리
        if isinstance(custom_inputs, str):
            custom_inputs = _json.loads(custom_inputs) if custom_inputs else {}
        thread_id = custom_inputs.get("thread_id", "default") if isinstance(custom_inputs, dict) else "default"


        # --- LangGraph 호출 ---
        config = {
            "recursion_limit": 20,
            "configurable": {"thread_id": thread_id},
        }
        lc_messages = []
        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict):
                    lc_messages.append((m.get("role", "human"), m.get("content", "")))
                else:
                    lc_messages.append(("human", str(m)))


        state = self.agent.invoke({"messages": lc_messages}, config=config)
        return state["messages"][-1].content


# COMMAND ----------

# ── UC 등록 ────────────────────────────────────────────────────────────────
model_name = "smartfactory.ai.maintenance_agent"


input_example = {
    "messages": [{"role": "user", "content": "EQ005 anomaly 감지. 조치 알려주세요."}],
    "custom_inputs": {"thread_id": "session-01"},
}
output_example = "EQ005 설비 점검 필요. LOTO 절차 수행 후 정비 진행."
signature = infer_signature(input_example, output_example)


resources = [
    DatabricksVectorSearchIndex(index_name="smartfactory.ai.manual_index"),
    DatabricksSQLWarehouse(warehouse_id=os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID")),
    DatabricksTable(table_name="smartfactory.processed.sensor_clean"),
    DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct"),
    DatabricksServingEndpoint(endpoint_name="smartfactory-pdm"),
]


username = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{username}/maintenance_agent")

with mlflow.start_run(run_name="maintenance_agent_postgres_saver"):
    mlflow.pyfunc.log_model(
        name="agent",
        python_model=MaintenanceAgentPG(),
        signature=signature,
        pip_requirements=[
            "databricks-langchain>=0.19.0",
            "langchain>=1.0",
            "langgraph>=1.1.0,<1.2",
            "langgraph-checkpoint-postgres>=2.0.0,<3.0",
            "psycopg[binary]>=3.1",
            "psycopg-pool>=3.1",
            "databricks-sdk>=0.118.0",
        ],
        registered_model_name=model_name,
        resources=resources,
    )


# COMMAND ----------

cli = mlflow.MlflowClient()
ver = max(int(v.version) for v in cli.search_model_versions(f"name='{model_name}'"))
cli.set_registered_model_alias(model_name, "champion", str(ver))
print(f"✅ UC 등록 완료: {model_name} v{ver} (@champion)")
print(f"   방식: PythonModel + PostgresSaver")
print(f"   멀티턴: 모든 워커가 같은 PostgreSQL DB 공유 → 세션 항상 유지")

# COMMAND ----------

# --- @champion 모델 로드 ---
model_name = "smartfactory.ai.maintenance_agent"
loaded_agent = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
print(f"✅ 로드 완료: {model_name}@champion\n")


# --- 멀티턴 테스트 (ChatCompletion + custom_inputs 형식) ---
thread = "verify-session-03"


def call_agent(msg, tid):
    """ChatCompletion 형식으로 호출 (StringResponse 반환)"""
    return loaded_agent.predict({
        "messages": [{"role": "user", "content": msg}],
        "custom_inputs": {"thread_id": tid},
    })


# 턴 1: 이름 + 설비 정보 전달
r1 = call_agent("내 이름은 박지원이고, EQ005 CNC에서 온도 92.5°C anomaly가 감지되었습니다. 조치 사항 알려주세요.", thread)
print(f"[턴 1] {r1[:250]}\n")


# 턴 2: 이전 대화 기억 확인 (같은 thread_id)
r2 = call_agent("내 이름이 뭐라고 ?", thread)
print(f"[턴 2] {r2[:250]}\n")


# 턴 3: 다른 thread_id → 독립 세션 (기억 없음)
r3 = call_agent("내 이름이 뭐라고 했지?", "different-session")
print(f"[턴 3 - 다른 세션] {r3[:150]}")
print("\n→ 다른 thread_id는 이전 대화를 모릅니다 (독립 세션 확인 ✅)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 25.3 에이전트 MLflow 패키징 및 REST API 배포

# COMMAND ----------

import mlflow
import os
from databricks import agents


# ── 1. @champion 버전 확인 ─────────────────────────────────────────
model_name = "smartfactory.ai.maintenance_agent"


cli = mlflow.MlflowClient()
champion = cli.get_model_version_by_alias(model_name, "champion")
print(f"📦 배포 대상: {model_name} v{champion.version} (@champion)")


# ── 3. agents.deploy() — 에이전트 전용 배포 ──────────────────────────
# agents.deploy()는:
#   - 서빙 컨테이너에 WorkspaceClient() 인증 자동 주입
#   - Inference Table (request/response 로깅) 자동 구성
#   - Review App 자동 생성
#   - resources 선언된 리소스에 SP 권한 자동 부여
deployment = agents.deploy(
    model_name=model_name,
    model_version=champion.version,
    scale_to_zero_enabled=False,
    environment_vars={
        "DATABRICKS_SQL_WAREHOUSE_ID": os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID"),
        "NEON_CONN_STRING": os.environ.get("NEON_CONN_STRING"),  # ← PostgresSaver용
    },
)


print(f"\n✅ 에이전트 배포 요청 완료")
print(f"   엔드포인트: {deployment.endpoint_name}")
print(f"   버전: v{champion.version} (PythonModel + PostgresSaver)")
print(f"   배포까지 5~10분 소요")
print(f"\n🔗 REST API URL:")
print(f"   POST https://<host>/serving-endpoints/{deployment.endpoint_name}/invocations")
print(f"\n📝 환경변수:")
print(f"   DATABRICKS_SQL_WAREHOUSE_ID = {os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID")[:3]}...")
print(f"   NEON_CONN_STRING = postgresql://...@...neon.tech/neondb")


# COMMAND ----------

import requests
import time


# ── 1. 엔드포인트 배포 대기 ──────────────────────────────────────
endpoint_name = "agents_smartfactory-ai-maintenance_agent"


from databricks.sdk import WorkspaceClient
wc = WorkspaceClient()


print("⏳ 엔드포인트 배포 대기 중...")
for i in range(30):  # 최대 15분 대기
    ep = wc.serving_endpoints.get(endpoint_name)
    if "READY" in str(ep.state.ready) and "NOT_UPDATING" in str(ep.state.config_update):
        print(f"✅ 엔드포인트 준비 완료! ({i*30}초 대기)")
        break
    print(f"   [{i*30}s] 상태: {ep.state.ready} / {ep.state.config_update}")
    time.sleep(30)
else:
    raise TimeoutError("엔드포인트가 15분 내에 준비되지 않았습니다.")


# ── 2. REST API 호출 설정 ───────────────────────────────────────
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = f"https://{host}/serving-endpoints/{endpoint_name}/invocations"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def parse_response(resp):
    """agents.deploy() 응답 파싱 (ChatCompletion / predictions 두 형식 지원)"""
    data = resp.json()
    if "predictions" in data:
        return data["predictions"][0] if isinstance(data["predictions"], list) else str(data["predictions"])
    elif "choices" in data:
        return data["choices"][0]["message"]["content"]
    elif "output" in data:
        return str(data["output"])
    else:
        return str(data)[:300]


# ── 3. 멀티턴 메모리 검증 (ChatCompletion + custom_inputs) ────
thread = "rest-verify-02"


# 턴 1: 이름 + 설비 정보 전달
payload_1 = {
    "messages": [{"role": "user", "content": "내 이름은 박지원이고, EQ005 CNC 온도 92.5°C anomaly 조치 알려주세요"}],
    "custom_inputs": {"thread_id": thread},
}
r1 = requests.post(url, headers=headers, json=payload_1, timeout=120)
print(f"[턴 1] HTTP {r1.status_code}")
print(f"  {parse_response(r1)}\n")


# 턴 2: 이전 대화 기억 확인 (같은 thread_id)
payload_2 = {
    "messages": [{"role": "user", "content": "내 이름이 보라고 했지?"}],
    "custom_inputs": {"thread_id": thread},
}
r2 = requests.post(url, headers=headers, json=payload_2, timeout=120)
print(f"[턴 2 - 메모리 확인] HTTP {r2.status_code}")
print(f"  {parse_response(r2)}\n")


# 턴 3: 다른 thread_id → 독립 세션 (기억 없음)
payload_3 = {
    "messages": [{"role": "user", "content": "내 이름이 보라고 했지?"}],
    "custom_inputs": {"thread_id": "different-session"},
}
r3 = requests.post(url, headers=headers, json=payload_3, timeout=120)
print(f"[턴 3 - 다른 세션] HTTP {r3.status_code}")
print(f"  {parse_response(r3)}")
print("\n→ 다른 thread_id는 이전 대화를 모릅니다 (독립 세션 확인 ✅)")
