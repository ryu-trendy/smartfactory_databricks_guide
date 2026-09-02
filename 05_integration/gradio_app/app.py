"""스마트팩토리 코리아 — 정비 AI 어시스턴트 (Gradio 채팅 앱)

Databricks Apps 배포용.
- 서빙 패턴: gradio~=4.44 + starlette==0.37.2 + gr.ChatInterface (Databricks Apps 표준)
- 인증: WorkspaceClient() 자동 인증 (OAuth M2M)
- 멀티턴: custom_inputs.thread_id → PostgresSaver 세션 유지
- 주의: gr.State를 inputs/outputs에 사용하면 gradio_client 스키마 버그 발생. ChatInterface 사용으로 회피.
"""

import os
import uuid

import gradio as gr
import requests
from databricks.sdk import WorkspaceClient

# ─── 설정 ──────────────────────────────────────────────────────────────────
ENDPOINT_NAME = os.environ.get(
    "SERVING_ENDPOINT", "agents_smartfactory-ai-multi_agent_supervisor"
)

_client = None


def _get_client() -> WorkspaceClient:
    global _client
    if _client is None:
        _client = WorkspaceClient()
    return _client


def _get_url() -> str:
    try:
        host = _get_client().config.host.rstrip("/")
    except Exception:
        host = os.environ.get("DATABRICKS_HOST", "")
    return f"{host}/serving-endpoints/{ENDPOINT_NAME}/invocations"


def _get_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    try:
        client = _get_client()
        auth = client.config.authenticate()
        if callable(auth):
            auth(headers)
        elif isinstance(auth, dict):
            headers.update(auth)
    except Exception as e:
        print(f"[WARN] 인증 실패: {e}")
    return headers


# ─── 에이전트 호출 ─────────────────────────────────────────────────────────
def call_agent_api(message: str, thread_id: str) -> str:
    payload = {
        "messages": [{"role": "user", "content": message}],
        "custom_inputs": {"thread_id": thread_id},
    }
    try:
        resp = requests.post(_get_url(), json=payload, headers=_get_headers(), timeout=120)
        if resp.status_code != 200:
            return f"\u274c 서버 오류 (HTTP {resp.status_code}): {resp.text[:200]}"
        result = resp.json()
        if result is None:
            return "\u274c 응답이 없습니다. 잠시 후 다시 시도해주세요."
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if msg.get("type") == "ai" and msg.get("content"):
                return msg["content"]
        return "응답을 처리 중입니다. 잠시 후 다시 시도해주세요."
    except requests.exceptions.Timeout:
        return "\u23f3 응답 시간 초과. 더 간단한 질문으로 다시 시도해주세요."
    except Exception as e:
        return f"\u274c 오류: {str(e)[:200]}"


# ─── 세션 관리 ──────────────────────────────────────────────────────
# 첫 번째 사용자 메시지를 키로 thread_id를 고정.
# history[0][0] (첫 질문)이 동일하면 동일 세션.
_session_map = {}  # {first_user_message: thread_id}


def respond(message: str, history: list) -> str:
    """채팅 응답 함수 (gr.ChatInterface 전용)"""
    if not history:
        # 첫 번째 메시지: thread_id 생성 후 저장
        thread_id = f"app-{uuid.uuid4().hex[:8]}"
        _session_map[message] = thread_id
    else:
        # 후속 메시지: 첫 질문으로 thread_id 조회
        first_msg = history[0][0]
        thread_id = _session_map.get(first_msg, f"app-{uuid.uuid4().hex[:8]}")
    return call_agent_api(message, thread_id)


# ─── Gradio ChatInterface (Databricks Apps 표준 패턴) ─────────────────
demo = gr.ChatInterface(
    fn=respond,
    title="\U0001f3ed 스마트팩토리 코리아 \u2014 정비 AI 어시스턴트",
    description="설비 이상 감지, OEE 분석, 고장 예측, 정비 매뉴얼 검색 등 다양한 질문을 한국어로 입력하세요.",
    examples=[
        "EQ005 현재 상태 확인",
        "LINE01 OEE 확인해줘",
        "최근 한달간 이상 감지 현황",
        "EQ005 고장 예측 결과",
        "오늘 일일 보고서 생성",
    ],
    theme=gr.themes.Soft(),
    css="footer {visibility: hidden}",
)

# ─── 앱 실행 (Databricks Apps 표준 패턴) ──────────────────────────────
if __name__ == "__main__":
    demo.launch()
