# Databricks notebook source
# 19장: Model Serving — 실시간 고장 예측 API 배포
# DBR 17.3 LTS (Spark 4.0.0 / Python 3.12.3)

# COMMAND ----------

# MAGIC %md
# MAGIC # 19장: Model Serving — 실시간 고장 예측 API 배포
# MAGIC
# MAGIC ## 실습 목표
# MAGIC - Unity Catalog 등록 모델을 Serving Endpoint로 배포
# MAGIC - Scale to Zero 설정으로 비용 최적화
# MAGIC - AutoCapture(Inference Table)로 예측 로그 수집
# MAGIC - 실시간 고장 예측 API 호출 테스트

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19.1 서빙 옵션 비교

# COMMAND ----------

import mlflow
from mlflow.deployments import get_deploy_client


mlflow.set_registry_uri("databricks-uc")
client = get_deploy_client("databricks")


endpoint_name = "smartfactory-pdm"


# 엔드포인트 존재 여부 확인 후 생성/업데이트
try:
    client.get_endpoint(endpoint_name)
    print(f"ℹ️ 엔드포인트 '{endpoint_name}' 이미 존재 — config 업데이트")
    client.update_endpoint(
        endpoint=endpoint_name,
        config={
            "served_entities": [
                {
                    "entity_name": "smartfactory.ml.failure_predictor",
                    "entity_version": "2",
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True,
                }
            ],
        },
    )
except Exception:
    print(f"🚀 엔드포인트 '{endpoint_name}' 신규 생성")
    client.create_endpoint(
        name=endpoint_name,
        config={
            "served_entities": [
                {
                    "entity_name": "smartfactory.ml.failure_predictor",
                    "entity_version": "2",
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True,
                }
            ],
        },
    )


print(f"✅ '{endpoint_name}' 배포 완료")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 19.2 실시간 추론 API 호출

# COMMAND ----------

import requests


# --- Serving 엔드포인트 URL 및 인증 ---
workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
endpoint_url = f"https://{workspace_host}/serving-endpoints/smartfactory-pdm/invocations"
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()


TIMEOUT_S = 300  # 콜드 스타트 대비 5분


def predict_failure(features: dict, threshold: float = 0.4) -> dict:
    """REST POST로 고장 예측 API 호출 후 위험 수준 딕셔너리 반환"""
    payload = {"dataframe_records": [features]}


    try:
        resp = requests.post(
            endpoint_url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()  # 4xx/5xx → HTTPError 발생
    except requests.exceptions.Timeout as e:
        raise TimeoutError(f"서빙 엔드포인트 응답 시간 초과 ({TIMEOUT_S}s): {e}")
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(f"네트워크 연결 오류: {e}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"서버 오류 (HTTP {resp.status_code}): {resp.text}")


    proba = resp.json()["predictions"][0]
    return {
        "failure_probability": round(proba, 4),
        "risk_level": "HIGH" if proba >= 0.7 else ("MEDIUM" if proba >= threshold else "LOW"),
        "recommendation": (
            "즉시 점검 필요"       if proba >= 0.7       else
            "48시간 내 점검 권고"  if proba >= threshold else
            "정상 모니터링 유지"
        ),
    }




# --- MES 호출 테스트 ---
result = predict_failure({
    "temp_avg_1h": 78.5, "temp_trend_4h": 3.2,
    "vib_avg_1h": 4.2, "vib_trend_4h": 1.1,
    "anomaly_count_1h": 5, "composite_anomaly_score": 0.78,
})
print(f"EQ015 고장 위험: {result['risk_level']} ({result['failure_probability']:.1%})")
print(f"권고: {result['recommendation']}")
