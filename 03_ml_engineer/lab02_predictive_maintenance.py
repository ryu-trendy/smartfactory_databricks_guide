# Databricks notebook source
# MAGIC %md
# MAGIC # 17장: 모델 개발 — XGBoost와 MLflow로 예지 정비 모델 만들기

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17.1 MLflow 실험 추적 구조

# COMMAND ----------

# MAGIC %pip install xgboost databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Cell 4
import mlflow, mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

# train_df / val_df / test_df 를 피쳐 행렬(X) + 레이블 벡터(y)로 분리
# FeatureLookup에서 요청한 피쳐명과 일치
from sklearn.model_selection import train_test_split
import numpy as np



feature_cols = [
    "temp_avg_1h", "temp_trend_4h", "vib_avg_1h",
    "vib_trend_4h", "anomaly_count_1h", "composite_anomaly_score",
]
label_col = "failure_within_48h"


fe = FeatureEngineeringClient()

label_df = spark.table("smartfactory.ml.pdm_labels").select(
    "equipment_id", "event_time", "failure_within_48h"
)

feature_lookups = [
    FeatureLookup(
        table_name="smartfactory.ml.equipment_features",
        lookup_key=["equipment_id"],
        timestamp_lookup_key="event_time",
        feature_names=feature_cols,
    )
]

training_set = fe.create_training_set(
    df=label_df,
    feature_lookups=feature_lookups,
    label=label_col,
)

training_df = training_set.load_df()


# 시간순 분할 시 양성 레이블이 앞부분에 집중 → train 세트가 전부 양성 → scale_pos_weight=nan
# 계층적 분할(stratified)로 각 세트의 클래스 비율을 보장
X_all = np.array(training_df.select(*feature_cols).fillna(0).collect()).astype(float)
y_all = np.array(training_df.select(label_col).collect()).astype(int)


X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.4, random_state=42, stratify=y_all
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


n_pos = int(y_train.sum())
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / n_pos   # 유한값 보장
print(f"X_train: {X_train.shape}  양성: {n_pos:,}  음성: {n_neg:,}  scale_pos_weight: {scale_pos_weight:.2f}")
print(f"X_val:   {X_val.shape}  /  X_test: {X_test.shape}")


print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print(f"피쳐 콜럼: {feature_cols}")


def train_and_evaluate(params: dict, run_name: str) -> dict:
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.set_tags({
            "feature_table": "smartfactory.ml.equipment_features",
            "model_type":    "XGBoost",
            "domain":        "predictive_maintenance",
        })


        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval   = xgb.DMatrix(X_val,   label=y_val)
        dtest  = xgb.DMatrix(X_test,  label=y_test)


        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=params.pop("num_boost_round", 300),
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )


        y_pred_proba = model.predict(dtest)
        threshold    = 0.4   # 재현율 우선
        y_pred       = (y_pred_proba >= threshold).astype(int)


        metrics = {
            "test_auc":    roc_auc_score(y_test, y_pred_proba),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1":     f1_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)


        # Feature Store 계보와 함께 모델 등록
        fe.log_model(
            model=model,
            artifact_path="model",
            flavor=mlflow.xgboost,
            training_set=training_set,
            registered_model_name="smartfactory.ml.failure_predictor",
        )


        print(f"[{run_name}] AUC={metrics['test_auc']:.4f}")
        return {"run_id": run.info.run_id, **metrics}


r1 = train_and_evaluate({
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 6, "learning_rate": 0.1,
    "scale_pos_weight": scale_pos_weight, "seed": 42,
}, "xgb_baseline")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 17.2 고장 예측 모델 MLflow 추적

# COMMAND ----------

# Lab 3-2: 고장 예측 모델 학습
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# 피처 로드 (Feature Store에서)
from databricks.feature_engineering import FeatureEngineeringClient


fe = FeatureEngineeringClient()


# → Spark 조인·재분할·타입 변환 불필요
# Cell 4의 scale_pos_weight도 그대로 사용
print(
    f"클래스 비율: scale_pos_weight={scale_pos_weight:.1f}, "
    f"X_train={X_train.shape}, X_test={X_test.shape}"
)


# MLflow 실험 실행
with mlflow.start_run(run_name="xgboost_v2_tuned") as run:
    params = {
        "n_estimators": 1000,
        "max_depth": 10,
        "learning_rate": 0.12,
        "scale_pos_weight": scale_pos_weight,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "gamma": 0,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "random_state": 42,
    }
    mlflow.log_params(params)


    # early_stopping_rounds: XGBoost 2.0+ 에서 fit() → 생성자로 이동
    model = XGBClassifier(**params, early_stopping_rounds=50)
    mlflow.xgboost.autolog()


    # X_train/y_train/X_test/y_test 는 Cell 8에서 이미 numpy 변환됨
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )


    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.3).astype(int)  # 재현율 우선 임계값


    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)


    mlflow.log_metric("test_auc", auc)
    mlflow.log_metric("test_precision", report["1"]["precision"])
    mlflow.log_metric("test_recall", report["1"]["recall"])
    mlflow.log_metric("test_f1", report["1"]["f1-score"])


    # Unity Catalog 등록: 입력/출력 signature 필수
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, model.predict_proba(X_train)[:, 1])
    mlflow.xgboost.log_model(
        model,
        "model",
        registered_model_name="smartfactory.ml.failure_predictor",
        signature=signature,
    )


    print(f"AUC: {auc:.4f}")
    print(f"재현율(Recall): {report['1']['recall']:.4f}")
    print(f"Run ID: {run.info.run_id}")
print("✅ 모델 학습 및 MLflow 등록 완료")


# COMMAND ----------

# MAGIC %md
# MAGIC # 18장: 모델 평가 — Champion/Challenger와 PSI 분석

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18.1 Champion/Challenger 승격 기준

# COMMAND ----------

from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score, recall_score
import mlflow
import xgboost as xgb


client = MlflowClient()


# alias 설정
client.set_registered_model_alias("smartfactory.ml.failure_predictor", "champion", "1")
client.set_registered_model_alias("smartfactory.ml.failure_predictor", "challenger", "2")


# Feature Store 래퍼를 우회하여 원본 XGBoost 모델 로드
def load_raw_model(model_name, alias):
    """fe.log_model 또는 mlflow.xgboost.log_model 모두 처리"""
    ver = client.get_model_version_by_alias(model_name, alias)
    try:
        return mlflow.xgboost.load_model(f"runs:/{ver.run_id}/model")
    except Exception:
        return mlflow.xgboost.load_model(f"runs:/{ver.run_id}/model/data/feature_store/raw_model")


champion = load_raw_model("smartfactory.ml.failure_predictor", "champion")
challenger = load_raw_model("smartfactory.ml.failure_predictor", "challenger")


CRITERIA = {"min_auc": 0.90, "min_recall": 0.80, "delta": 0.01}


def get_proba(model, X):
    """Booster 또는 XGBClassifier에서 확률값 추출"""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    else:
        return model.predict(xgb.DMatrix(X))


def evaluate_cc(champ_model, chall_model, X_test, y_test):
    threshold = 0.4
    champ_p = get_proba(champ_model, X_test)
    chall_p = get_proba(chall_model, X_test)


    champ_auc    = roc_auc_score(y_test, champ_p)
    chall_auc    = roc_auc_score(y_test, chall_p)
    chall_recall = recall_score(y_test, (chall_p >= threshold).astype(int))


    print(f"Champion AUC: {champ_auc:.4f}")
    print(f"Challenger AUC: {chall_auc:.4f}, Recall: {chall_recall:.4f}")


    should_promote = (
        chall_auc    >= CRITERIA["min_auc"]    and
        chall_recall >= CRITERIA["min_recall"] and
        chall_auc - champ_auc >= CRITERIA["delta"]
    )


    if should_promote:
        chall_ver = client.get_model_version_by_alias(
            "smartfactory.ml.failure_predictor", "challenger").version
        client.set_registered_model_alias(
            name="smartfactory.ml.failure_predictor",
            alias="champion", version=chall_ver,
        )
        # 승격 후 challenger alias 제거 (다음 후보 모델이 학습될 때까지 불필요)
        client.delete_registered_model_alias(
            name="smartfactory.ml.failure_predictor",
            alias="challenger",
        )
        print("Challenger → Champion 승격 완료! (challenger alias 제거)")
    else:
        print(f"미달 (AUC 개선: {chall_auc-champ_auc:.4f}). Champion 유지.")


evaluate_cc(champion, challenger, X_test, y_test)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 18.2 PSI 드리프트 탐지

# COMMAND ----------

import numpy as np


def calculate_psi(reference: np.ndarray, current: np.ndarray, n_bins=10) -> float:
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0]  = -np.inf
    bins[-1] = np.inf


    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current,   bins=bins)


    ref_pct = ref_counts / len(reference) + 1e-10
    cur_pct = cur_counts / len(current)   + 1e-10


    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# 학습 데이터 vs 최근 30일 비교
recent_df = spark.table("smartfactory.ml.equipment_features") \
    .filter("event_time >= DATEADD(DAY, -30, CURRENT_TIMESTAMP())") \
    .toPandas()


# 전체 6개 피처 PSI 계산 후 드리프트 심한 순으로 정렬
results = []
for feat in feature_cols:
    feat_idx = feature_cols.index(feat)
    psi = calculate_psi(X_train[:, feat_idx],
                        recent_df[feat].dropna().values)
    status = "stable" if psi < 0.10 else ("warning" if psi < 0.20 else "DRIFT")
    results.append((feat, psi, status))


results.sort(key=lambda x: x[1], reverse=True)


print("=" * 55)
print(f"{'순위':<4} {'피처':<28} {'PSI':<10} {'판정'}")
print("=" * 55)
drift_count = 0
for rank, (feat, psi, status) in enumerate(results, 1):
    flag = "🚨" if status == "DRIFT" else ("⚠️" if status == "warning" else "✅")
    print(f"{rank:<4} {feat:<28} {psi:<10.4f} {flag} {status}")
    if status == "DRIFT":
        drift_count += 1


print("=" * 55)
if drift_count > 0:
    print(f"\n🚨 DRIFT 피처 {drift_count}개 발견 → 재학습 파이프라인 실행 권고")
else:
    print("\n✅ 모든 피처 안정. 재학습 불필요.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 18.3 Champion/Challenger 전략

# COMMAND ----------

# Lab: Best Run 자동 선별 → 모델 등록 → Champion 지정 → 예측 검증
import mlflow
from mlflow.tracking import MlflowClient
import xgboost as xgb


client = MlflowClient()
MODEL_NAME = "smartfactory.ml.failure_predictor"


# 1. 최고 AUC 런 자동 선별 (현재 노트북 experiment에서)
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()


all_runs = mlflow.search_runs(
    experiment_names=[notebook_path],
    order_by=["metrics.test_auc DESC"],
    max_results=10
)


# 2. 모델 버전이 등록된 run 중 최고 AUC 선별
all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
registered_run_ids = {v.run_id: v for v in all_versions}


best_run = None
for _, run in all_runs.iterrows():
    if run['run_id'] in registered_run_ids:
        best_run = run
        break


if best_run is None:
    raise ValueError("모델 버전이 등록된 run을 찾을 수 없습니다.")


model_version = registered_run_ids[best_run['run_id']]
print(f"Best Run: {best_run['run_id'][:8]}... | AUC: {best_run['metrics.test_auc']:.4f}")
print(f"기존 등록 버전: v{model_version.version}")


# 3. Champion alias 지정
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=model_version.version
)
print(f"Champion alias → v{model_version.version}")


# 4. Champion 모델 로드 및 예측 검증 (Feature Store 래퍼 우회)
def load_raw_model(model_name, alias):
    """fe.log_model 또는 mlflow.xgboost.log_model 모두 처리"""
    ver = client.get_model_version_by_alias(model_name, alias)
    try:
        return mlflow.xgboost.load_model(f"runs:/{ver.run_id}/model")
    except Exception:
        return mlflow.xgboost.load_model(f"runs:/{ver.run_id}/model/data/feature_store/raw_model")


champion = load_raw_model(MODEL_NAME, "champion")


if hasattr(champion, 'predict_proba'):
    predictions = champion.predict_proba(X_test[:100])[:, 1]
else:
    predictions = champion.predict(xgb.DMatrix(X_test[:100]))


print(f"Champion 예측 샘플 (상위 5): {predictions[:5]}")
print("✅ Champion 모델 배포 완료")
