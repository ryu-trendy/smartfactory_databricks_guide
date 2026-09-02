# Databricks notebook source
# MAGIC %md
# MAGIC # 20장: Foundation Models API — LLM을 워크스페이스 안으로
# MAGIC
# MAGIC ## 실습 목표
# MAGIC - Databricks Foundation Models API 직접 호출
# MAGIC - 다양한 프롬프트 패턴 실습 (Zero-Shot, Few-Shot, Chain-of-Thought)
# MAGIC - 정비 진단 프롬프트 최적화
# MAGIC - 구조화된 출력 (JSON 모드)

# COMMAND ----------

# MAGIC %md
# MAGIC ##20.1 Foundation Models API 개요
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

def chat(messages: list, endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
         temperature: float = 0.1, max_tokens: int = 512) -> str:
    response = client.predict(
        endpoint=endpoint,
        inputs={
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
        },
    )
    return response["choices"][0]["message"]["content"]

answer = chat([
    {"role": "system", "content": "당신은 스마트팩토리 설비 전문가입니다. 한국어로 답변하세요."},
    {"role": "user",   "content": "CNC 스핀들 온도 85도 초과 시 조치는?"},
])
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 20.2 구조화 출력 — JSON 모드

# COMMAND ----------

import json


def analyze_defect_structured(defect_description: str) -> dict:
    prompt = (
        "다음 불량 설명을 분석하고 JSON 형식으로 반환하세요.\n"
        "반드시 다음 키만 포함: category, severity, root_cause, action\n\n"
        f"불량 설명: {defect_description}\n\n"
        "JSON만 반환하세요 (마크다운, 설명 텍스트 없이):"
    )
    response = chat( # 21.1 의 chat() 함수를 사용합니다. 앞의 코드가 선행되어야 합니다.
        [{"role": "system", "content": "당신은 JSON만 반환하는 불량 분석 AI입니다."},
         {"role": "user",   "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    try:
        return json.loads(response.strip().strip('```json').strip('```'))
    except json.JSONDecodeError:
        return {"error": "파싱 실패", "raw": response}


result = analyze_defect_structured("PCB 기판 솔더링 불량 — 납 볼 크기 불균일")
print(json.dumps(result, ensure_ascii=False, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ##20.3 임베딩 API
# MAGIC

# COMMAND ----------

import numpy as np

def get_embedding(text: str) -> list:
    response = client.predict(
        endpoint="databricks-qwen3-embedding-0-6b",
        inputs={"input": [text]},
    )
    return response["data"][0]["embedding"]  # 1024차원 벡터

def compute_similarity(text1: str, text2: str) -> float:
    emb1 = np.array(get_embedding(text1))
    emb2 = np.array(get_embedding(text2))
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

# 다국어 모델이므로 한국어 정비 용어를 직접 테스트 (번역 불필요)
pairs = [
    ("스핀들 과열", "CNC 스핀들 온도 이상"),       # 유사 의미
    ("스핀들 과열", "컨베이어 벨트 마모"),       # 다른 의미
    ("유압 압력 저하", "오일펌프 압력 손실"),     # 유사 의미
    ("스핀들 과열", "spindle overheat"),       # 한영 교차 유사도
]
for t1, t2 in pairs:
    print(f"'{t1}' vs '{t2}': {compute_similarity(t1, t2):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##20.4 완전 실습 Lab — Foundation Models 활용 설비 이상 분석 파이프라인
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT equipment_id,
# MAGIC         ROUND(AVG(temperature_c), 1)  AS avg_temp,
# MAGIC         ROUND(AVG(vibration_ms2), 2)  AS avg_vib,
# MAGIC         ROUND(AVG(pressure_bar), 2)   AS avg_curr
# MAGIC FROM   smartfactory.processed.sensor_clean
# MAGIC WHERE  (temperature_c > 85 OR vibration_ms2 > 5.0)
# MAGIC GROUP  BY equipment_id
# MAGIC ORDER  BY MAX(event_time) DESC
# MAGIC LIMIT  10 -- 토큰 소모를 예방하기 위한 테스트 데이터 10건 확인
# MAGIC

# COMMAND ----------

# DBTITLE 1,Foundation Models 설비 이상 분석 파이프라인
import mlflow.deployments
import json
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructType, StructField

client = mlflow.deployments.get_deploy_client("databricks")

# --- 1. Foundation Models 클라이언트 함수 ---
def chat(messages: list, endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
         temperature: float = 0.1, max_tokens: int = 512) -> str:
    resp = client.predict(endpoint=endpoint,
                          inputs={"messages": messages, "temperature": temperature,
                                  "max_tokens": max_tokens})
    return resp["choices"][0]["message"]["content"]


def get_embedding(text: str) -> list:
    """다국어 임베딩 모델(Qwen3 Embedding 0.6B)로 텍스트를 벡터로 변환. 한국어 직접 지원."""
    resp = client.predict(endpoint="databricks-qwen3-embedding-0-6b", inputs={"input": [text]})
    return resp["data"][0]["embedding"]

# --- 2. 구조화 분석 함수 ---
def analyze_equipment_alert(equipment_id: str, temperature_c: float,
                              vibration_ms2: float, pressure_bar: float) -> dict:
    prompt = (
        f"설비 ID: {equipment_id}\n"
        f"온도: {temperature_c}°C, 진동: {vibration_ms2}mm/s, 전류: {pressure_bar}A\n\n"
        "이상 원인을 분석하고 JSON으로 반환하세요 (키: cause, severity, action, estimated_downtime_hours):\n"
        "JSON만 반환하세요 (마크다운 없이):"
    )
    raw = chat(
        [{"role": "system", "content": "당신은 CNC 설비 전문가입니다. JSON만 반환합니다."},
         {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=256,
    )
    try:
        return json.loads(raw.strip().strip('```json').strip('```'))
    except json.JSONDecodeError:
        return {"cause": "파싱 실패", "severity": "UNKNOWN", "action": raw[:200],
                "estimated_downtime_hours": -1}

# --- 3. 이상 설비 배치 분석 ---
alert_rows = spark.sql('''
   SELECT equipment_id,
           ROUND(AVG(temperature_c), 1)  AS avg_temp,
           ROUND(AVG(vibration_ms2), 2)  AS avg_vib,
           ROUND(AVG(pressure_bar), 2)   AS avg_curr
    FROM   smartfactory.processed.sensor_clean
    WHERE  (temperature_c > 85 OR vibration_ms2 > 5.0)
    GROUP  BY equipment_id
    ORDER  BY MAX(event_time) DESC
    LIMIT  10
''').collect()
print(f"이상 설비 {len(alert_rows)}건 감지 → LLM 분석 시작")


# 기준 고장 패턴 (임베딩 유사도 비교용) — 다국어 모델이므로 한국어 직접 사용
REFERENCE_PATTERNS = [
    "베어링 마모로 인한 스핀들 과열",
    "과도 진동으로 인한 모터 과부하",
    "씨링 노화로 인한 유압 압력 손실",
    "제어반 전기 단락",
]
ref_embeddings = {p: np.array(get_embedding(p)) for p in REFERENCE_PATTERNS}


def find_best_match(cause_text: str) -> tuple:
    """원인 텍스트와 가장 유사한 기준 패턴을 찾아 (pattern, score) 반환"""
    cause_emb = np.array(get_embedding(cause_text))
    best_pattern, best_score = "", 0.0
    for pattern, ref_emb in ref_embeddings.items():
        score = float(np.dot(cause_emb, ref_emb) / (np.linalg.norm(cause_emb) * np.linalg.norm(ref_emb)))
        if score > best_score:
            best_pattern, best_score = pattern, score
    return best_pattern, best_score

results = []
for row in alert_rows:
    analysis = analyze_equipment_alert(
        row.equipment_id, row.avg_temp, row.avg_vib, row.avg_curr
    )
    cause = analysis.get("cause", "N/A")
    matched_pattern, sim_score = find_best_match(cause) if cause != "N/A" else ("", 0.0)
    results.append({
        "equipment_id":          row.equipment_id,
        "cause":                 cause,
        "severity":              analysis.get("severity", "N/A"),
        "action":                analysis.get("action", "N/A"),
        "estimated_downtime_hrs": analysis.get("estimated_downtime_hours", -1),
        "matched_pattern":       matched_pattern,
        "similarity_score":      round(sim_score, 4),
    })
    print(f"[{row.equipment_id}] {analysis.get('severity','?')} — {cause} (best match: {matched_pattern[:30]}.. {sim_score:.3f})")

# --- 4. 결과 저장 ---
if results:
    df_alerts = spark.createDataFrame(results)
    (df_alerts.write
     .format("delta")
     .mode("append")
     .option("mergeSchema", "true")
     .saveAsTable("smartfactory.ai.equipment_alerts"))
    print(f"✅ {len(results)}건 이상 분석 결과 저장 완료")

# --- 5. 의미 유사도 검증 ---
pairs = [
    ("스핀들 과열", "CNC 스핀들 온도 이상"),       # 유사 의미
    ("스핀들 과열", "컨베이어 벨트 마모"),       # 다른 의미
    ("스핀들 과열", "spindle overheat"),       # 한영 교차 유사도
]
for t1, t2 in pairs:
    e1, e2 = np.array(get_embedding(t1)), np.array(get_embedding(t2))
    sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    print(f"유사도 '{t1}' vs '{t2}': {sim:.4f}")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM smartfactory.ai.equipment_alerts

# COMMAND ----------

# MAGIC %md
# MAGIC # 21장: 프롬프트 엔지니어링 — LLM에서 최고의 성능 끌어내기

# COMMAND ----------

# MAGIC %md
# MAGIC ## 21.1 효과적인 프롬프트의 5원칙

# COMMAND ----------

import json
import mlflow.deployments


client = mlflow.deployments.get_deploy_client("databricks")


def chat(messages: list, endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
         temperature: float = 0.1, max_tokens: int = 512) -> str:
    response = client.predict(
        endpoint=endpoint,
        inputs={
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
        },
    )
    return response["choices"][0]["message"]["content"]


DEFECT_ANALYSIS_PROMPT = '''
[역할] 당신은 20년 경력의 스마트팩토리 설비 정비 전문가입니다.

[목표] 다음 센서 이상 데이터를 분석하여 근본 원인과 즉각 조치를 제시하세요.

[제약]
- 기술적 근거 없는 추측 금지
- 안전 관련 사항은 반드시 LOTO 절차 언급
- 한국어로 답변

[형식] 다음 JSON 형식으로만 반환:
{{
  "primary_cause": "주요 원인 (50자 이내)",
  "immediate_action": "즉각 조치 (1문장)",
  "requires_loto": true/false,
  "urgency": "HIGH/MEDIUM/LOW"
}}

[예시]
입력: 온도 92도, 진동 7.2mm/s, CNC 스핀들
출력: {{"primary_cause": "스핀들 베어링 과부하 또는 윤활 부족", ...}}

[실제 데이터]
설비: {equipment_id} ({equipment_type})
온도: {temperature_c}°C, 진동: {vibration_ms2}mm/s
'''

def analyze_equipment_anomaly(equipment_id, equipment_type,
                               temperature_c, vibration_ms2):
    prompt = DEFECT_ANALYSIS_PROMPT.format(
        equipment_id=equipment_id, equipment_type=equipment_type,
        temperature_c=temperature_c, vibration_ms2=vibration_ms2,
    )
    response = chat([{"role": "user", "content": prompt}], temperature=0.0)
    return json.loads(response.strip().strip('```json').strip('```'))

result = analyze_equipment_anomaly("EQ015", "CNC", 88.5, 6.8)
print(json.dumps(result, ensure_ascii=False, indent=2))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 21.2 Chain-of-Thought (CoT) 프롬프팅

# COMMAND ----------

COT_PROMPT = '''
[설비 고장 근본 원인 분석]

다음 정보를 바탕으로 단계적으로 분석하세요:

1단계: 각 센서 값이 정상 범위인지 확인
   - CNC 온도 정상: 40-80도C, 진동 정상: 0.5-3.0mm/s
2단계: 이상 패턴의 조합을 분석
   - 온도+진동 동시 상승 = 기계적 마찰 또는 냉각 문제
   - 전류 급증 + 진동 = 과부하 또는 이물질
3단계: 가장 가능성 높은 원인 결론

설비: {equipment_id}, 마지막 정비: {last_maintenance}
온도: {temp}도C, 진동: {vib}mm/s, 전류: {curr}A

분석 결과 (단계별 사고 과정 포함):
'''

def analyze_with_cot(equipment_id, temp, vib, curr, last_maintenance):
    return chat(
        [{"role": "user", "content": COT_PROMPT.format(
            equipment_id=equipment_id, temp=temp, vib=vib,
            curr=curr, last_maintenance=last_maintenance,
        )}],
        temperature=0.1, max_tokens=800,
    )

print(analyze_with_cot("EQ015", 88.5, 6.8, 18.3, "2024-01-10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 21.3 제조 도메인 프롬프트 최적화

# COMMAND ----------

# Lab: 프롬프트 엔지니어링 A/B 비교
import mlflow, json


# 프롬프트 버전 A (기본)
PROMPT_V1 = """불량 유형을 분류하세요: {defect_description}
카테고리: 치수불량, 표면결함, 재료결함, 조립오류, 기능불량"""

# 프롬프트 버전 B (Few-shot + system/user 역할 분리)
SYSTEM_V2 = """당신은 스마트팩토리 코리아의 품질 검사 전문가입니다.
다음 카테고리 중 하나로만 분류하세요. 카테고리명만 답하세요.


카테고리:
- 치수불량: 규격 치수 초과/미달
- 표면결함: 스크래치, 변색, 도장 불량
- 재료결함: 재질 불량, 이물질 혼입
- 조립오류: 부품 누락, 잘못된 조립 순서
- 기능불량: 작동 불량, 센서 오류


예시:
설명: "표면에 긁힌 흔적이 선명함" → 표면결함
설명: "나사 토크값 규격 미달 2.3Nm" → 치수불량


분류할 불량:
{defect_description}"""


USER_V2 = """분류할 불량:
{defect_description}"""

# 테스트 데이터
test_cases = [
    "플랜지 외경 50.3mm (규격: 50±0.1mm)",
    "표면 도장 박리 현상 발생",
    "체결 볼트 누락으로 조립 불완전"
]


username = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{username}/prompt_engineering_lab")


with mlflow.start_run(run_name="prompt_ab_test"):
    results_v1, results_v2 = [], []
    for case in test_cases:
        # V1: 단순 user 메시지만
        r1 = chat([{"role": "user", "content": PROMPT_V1.format(defect_description=case)}],
                  temperature=0.0, max_tokens=50)
        results_v1.append(r1.strip())


        # V2: system/user 역할 분리
        r2 = chat([{"role": "system", "content": SYSTEM_V2},
                   {"role": "user", "content": USER_V2.format(defect_description=case)}],
                  temperature=0.0, max_tokens=50)
        results_v2.append(r2.strip())


    mlflow.log_param("prompt_v1", "basic_category_list")
    mlflow.log_param("prompt_v2", "fewshot_system_user_split")
    mlflow.log_metric("test_cases", len(test_cases))


# 결과를 DataFrame으로 비교
import pandas as pd


expected = ["치수불량", "표면결함", "조립오류"]
df_ab = pd.DataFrame({
    "입력 불량 설명": test_cases,
    "정답": expected,
    "V1 (기본)": results_v1,
    "V2 (Few-shot+역할분리)": results_v2,
    "V1 정확": [r.strip() == e for r, e in zip(results_v1, expected)],
    "V2 정확": [r.strip() == e for r, e in zip(results_v2, expected)],
})
print(f"✅ 프롬프트 A/B 테스트 완료 | V1 정확도: {df_ab['V1 정확'].mean():.0%} | V2 정확도: {df_ab['V2 정확'].mean():.0%}")
display(df_ab)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC - **22~23장**: `04_ai_engineer/02_lab02_rag_maintenance.py` (Vector Search + RAG)