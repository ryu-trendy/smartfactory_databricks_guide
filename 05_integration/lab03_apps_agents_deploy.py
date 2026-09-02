# Databricks notebook source
# DBTITLE 1,29장 개요
# MAGIC %md
# MAGIC # 28장 | Databricks Apps — 웹 인터페이스로 AI 서비스 제공
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC > 외부 PostgreSQL 인스턴스 사용을 위해 인프라팀의 데이터베이스 생성, 네트워크 연결, 보안 정책 및 방화벽 설정 등이 필요할 수 있습니다. <br>
# MAGIC > 반면 이 장에서는 별도의 인프라 구축 없이 독자가 직접 실습할 수 있도록 Neon Free Tier의 Serverless PostgreSQL을 사용합니다.<br>
# MAGIC > Cluster를 **Serverless** 로 교체하여 사용합니다. <br>
# MAGIC > 25장 실습 환경 안내 참조

# COMMAND ----------

# DBTITLE 1,29.1 멀티에이전트 모델 배포
# MAGIC %md
# MAGIC ## 28.1 Gradio 기반 채팅 인터페이스

# COMMAND ----------

# MAGIC %pip install --upgrade "langgraph-checkpoint-postgres>=2.0.0" "psycopg-pool>=3.1" --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from databricks.sdk import WorkspaceClient
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
NEON_CONN_STRING = "postgresql://neondb_owner:....aws.neon.tech/neondb?sslmode=require"

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

import mlflow
import os
from databricks import agents

# ── 1. @champion 버전 확인 ─────────────────────────────────────────
model_name = "smartfactory.ai.multi_agent_supervisor"

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
print(f"   NEON_CONN_STRING = postgresql://...@ep-hidden-block-...neon.tech/neondb")

# COMMAND ----------

# DBTITLE 1,29.2 엔드포인트 REST API 테스트
import requests
import json

# === 엔드포인트 설정 ===
endpoint_name = "agents_smartfactory-ai-multi_agent_supervisor"
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

url = f"https://{host}/serving-endpoints/{endpoint_name}/invocations"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

def call_agent(message: str, thread_id: str) -> str:
    """에이전트 엔드포인트 호출 후 마지막 AI 응답 추출"""
    payload = {
        "messages": [{"role": "user", "content": message}],
        "custom_inputs": {"thread_id": thread_id},
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    if resp.status_code != 200:
        return f"❌ HTTP {resp.status_code}: {resp.text[:200]}"

    result = resp.json()
    if result is None:
        return "❌ 응답이 null입니다. 서빙 로그를 확인하세요."

    # 응답 파싱: messages 배열에서 마지막 AI 메시지 content 추출
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if msg.get("type") == "ai" and msg.get("content"):
            return msg["content"]

    # fallback: 전체 응답 요약
    return json.dumps(result, ensure_ascii=False)[:500]


# === 테스트 1: 단일 질의 ===
thread_id = "multi-turn-test-001"
print("=" * 60)
print(f"🧪 테스트 1: 단일 질의 (thread_id={thread_id})")
print("=" * 60)

q1 = "LINE01 OEE 확인해줘"
print(f"\n👤 사용자: {q1}")
a1 = call_agent(q1, thread_id)
print(f"\n🤖 에이전트:\n{a1[:800]}")

# === 테스트 2: 멀티턴 (같은 thread_id → 대화 이력 유지) ===
print("\n" + "=" * 60)
print(f"🧪 테스트 2: 멀티턴 후속 질문 (동일 thread_id={thread_id})")
print("=" * 60)

q2 = "그 라인에서 가장 이상이 많은 설비는?"
print(f"\n👤 사용자: {q2}")
a2 = call_agent(q2, thread_id)
print(f"\n🤖 에이전트:\n{a2[:800]}")

# === 테스트 3: 다른 thread_id (독립 세션 확인) ===
print("\n" + "=" * 60)
print(f"🧪 테스트 3: 독립 세션 (새 thread_id)")
print("=" * 60)

q3 = "EQ005 고장 예측 결과 알려줘"
print(f"\n👤 사용자: {q3}")
a3 = call_agent(q3, "independent-session-002")
print(f"\n🤖 에이전트:\n{a3[:800]}")

# COMMAND ----------

# DBTITLE 1,29.2 Databricks Apps 배포
# MAGIC %md
# MAGIC ## 28.2 Databricks Apps 배포 설정
# MAGIC
# MAGIC 아키텍처:
# MAGIC ```
# MAGIC gradio_app/
# MAGIC   ├── app.py            ← Gradio ChatInterface (REST API 호출)
# MAGIC   ├── app.yaml          ← 앱 설정 (command, env, resources)
# MAGIC   └── requirements.txt  ← 의존성 (gradio~=4.44, starlette==0.37.2)
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,28.2 Databricks Apps 배포
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import App, AppDeployment

w = WorkspaceClient()

APP_NAME = "smartfactory-maintenance-ai"

# 현재 노트북 경로 기준으로 gradio_app/ 소스 경로 자동 구성
_notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
SOURCE_PATH = "/Workspace" + os.path.dirname(_notebook_path) + "/gradio_app"

print(f"🚀 Databricks Apps 배포 시작")
print(f"   앱 이름: {APP_NAME}")
print(f"   소스 경로: {SOURCE_PATH}")
print(f"   파일 구성: app.py / app.yaml / requirements.txt")
print()

# ── 1. 앱 생성 (이미 있으면 스킵) ──────────────────────────────────
try:
    app = w.apps.get(name=APP_NAME)
    print(f"✅ 앱 '{APP_NAME}' 이미 존재 — 업데이트 배포 진행")
    print(f"   URL: {app.url or '(배포 후 할당)'}")
except Exception:
    print(f"🆕 앱 '{APP_NAME}' 생성 중...")
    app = w.apps.create_and_wait(
        App(
            name=APP_NAME,
            description="스마트팩토리 코리아 정비 AI 어시스턴트 (Gradio 채팅 앱)",
        )
    )
    print(f"   ✅ 앱 생성 완료")
    print(f"   URL: {app.url or '(배포 후 할당)'}")

# ── 2. 앱 배포 ────────────────────────────────────────────────────
print(f"\n📦 배포 실행 중...")
try:
    deployment = w.apps.deploy_and_wait(
        app_name=APP_NAME,
        app_deployment=AppDeployment(source_code_path=SOURCE_PATH),
    )
    print(f"✅ 배포 완료!")
    print(f"   배포 ID: {deployment.deployment_id}")
    print(f"   상태: {deployment.status.state.value if deployment.status else 'N/A'}")
except Exception as e:
    print(f"❌ 배포 실패: {e}")

# ── 3. 최종 앱 상태 확인 ──────────────────────────────────────────
print(f"\n🔗 앱 최종 상태:")
try:
    app = w.apps.get(name=APP_NAME)
    print(f"   상태: {app.app_status.state.value if app.app_status else 'UNKNOWN'}")
    print(f"   URL: {app.url or '할당 대기 중'}")
    if app.url:
        print(f"\n   👉 브라우저에서 접속: {app.url}")
except Exception as e:
    print(f"   상태 확인 실패: {e}")

# COMMAND ----------

# DBTITLE 1,29.3 완전 실습 Lab
# MAGIC %md
# MAGIC ## 28.3 실습 Lab — Gradio AI 어시스턴트 앱 구현
# MAGIC
# MAGIC
# MAGIC 권한 구조:
# MAGIC ```
# MAGIC [사용자 브라우저]
# MAGIC     │
# MAGIC     ▼
# MAGIC [앱 SP]  ── CAN_QUERY (수동) ──▶  [서빙 엔드포인트]
# MAGIC                                             │
# MAGIC                                      자동 인증 토큰
# MAGIC                                             │
# MAGIC                               ┌─────────┴─────────┐
# MAGIC                               │                       │
# MAGIC                     CAN_USE (자동)        SELECT (자동)
# MAGIC                               │                       │
# MAGIC                      [SQL Warehouse]         [UC 테이블]
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,28.3 앱 SP 권한 부여
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel

w = WorkspaceClient()

APP_NAME = "smartfactory-maintenance-ai"
ENDPOINT_NAME = "agents_smartfactory-ai-multi_agent_supervisor"

# ══════════════════════════════════════════════════════════════════════
# PART 1: 앱 SP → 서빙 엔드포인트 CAN_QUERY (수동)
# ══════════════════════════════════════════════════════════════════════
# Databricks Apps는 앱마다 SP(Service Principal)를 자동 생성합니다.
# 이 SP가 서빙 엔드포인트를 호출할 수 있도록 CAN_QUERY 권한을 부여합니다.
# 주의: app.service_principal_name은 display_name이므로,
#       w.service_principals.get(id=...)로 application_id를 조회해야 합니다.

print("=" * 60)
print("PART 1: 앱 SP → 서빙 엔드포인트 CAN_QUERY")
print("=" * 60)

# 앱 SP 정보 조회
app = w.apps.get(name=APP_NAME)
app_sp = w.service_principals.get(id=app.service_principal_id)
print(f"🔑 앱 SP: {app_sp.display_name}")
print(f"   application_id: {app_sp.application_id}")

# 엔드포인트에 CAN_QUERY 권한 부여
endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
w.serving_endpoints.set_permissions(
    serving_endpoint_id=endpoint.id,
    access_control_list=[
        AccessControlRequest(
            service_principal_name=app_sp.application_id,
            permission_level=PermissionLevel.CAN_QUERY,
        )
    ],
)
print(f"✅ {app_sp.display_name} → CAN_QUERY on {ENDPOINT_NAME}")

# ══════════════════════════════════════════════════════════════════════
# PART 2: 서빙 엔드포인트 자동 인증 → UC 테이블/Warehouse 권한 (자동)
# ══════════════════════════════════════════════════════════════════════
# agents.deploy()는 모델의 resources 선언을 읽어 자동 인증 토큰에 권한 자동 부여.
print(f"\n{'=' * 60}")
print("PART 2: 서빙 엔드포인트 자동 인증 → UC 테이블 권한 (자동)")
print("=" * 60)
print("""
📌 agents.deploy()는 모델 등록 시 선언된 resources를 기반으로
   자동 인증 토큰에 권한을 자동으로 부여합니다.

   27장 내용:

   resources = [
       DatabricksVectorSearchIndex(index_name="smartfactory.ai.manual_index"),
       DatabricksSQLWarehouse(warehouse_id=os.environ.get('DATABRICKS_SQL_WAREHOUSE_ID')),
       DatabricksTable(table_name="smartfactory.processed.sensor_clean"),  # ← SELECT 자동 부여
       DatabricksTable(table_name="smartfactory.analytics.oee_daily"),     # ← SELECT 자동 부여
       DatabricksServingEndpoint(endpoint_name="smartfactory-pdm"),
       DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct"),
   ]

   → agents.deploy() 실행 시 자동 인증 토큰에 다음 권한이 자동 부여됩니다:
     • SQL Warehouse  → CAN_USE
     • smartfactory.processed.sensor_clean  → SELECT
     • smartfactory.analytics.oee_daily     → SELECT
     • smartfactory-pdm                     → CAN_QUERY
     • databricks-meta-llama-3-3-70b-instruct → CAN_QUERY
     • smartfactory.ai.manual_index         → Vector Search 접근
""")

# ══════════════════════════════════════════════════════════════════════
# 요약
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("✅ 권한 설정 요약")
print("=" * 60)
print(f"""
  [수동] 앱 SP ({app_sp.display_name})
       → CAN_QUERY → [{ENDPOINT_NAME}]

  [자동] 서빙 자동인증 (resources 선언 기반)
       → CAN_USE  → [SQL Warehouse]
       → SELECT   → [smartfactory.processed.sensor_clean]
       → SELECT   → [smartfactory.analytics.oee_daily]
       → CAN_QUERY → [smartfactory-pdm]

  📝 테이블 권한이 없다면:
       1. 27장 모델 재등록(모델 재등록, DatabricksTable resources 포함)
       2. 29장 모델 재 배포 (agents.deploy() 재배포 → 권한 자동 부여)

  앱 URL: {app.url}
""")