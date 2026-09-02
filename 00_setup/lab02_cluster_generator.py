# Databricks notebook source
# MAGIC %md
# MAGIC #3장 Databricks 환경 설정 — 워크스페이스부터 클러스터까지

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 첫 번째 클러스터 생성

# COMMAND ----------

# 클러스터 생성 권장 설정 (Databricks SDK 사용)
cluster_config = {
    "cluster_name":    "smartfactory-dev",
    "spark_version":   "18.x-scala2.13",    # DBR 18 LTS (Apache Spark 4.1.0, Scala 2.13)
    "node_type_id":    "m5d.large",         # AWS: 8GB RAM, 2 core
    "num_workers":      2,                  # 소규모 실습용 2대 (운영은 4~8대)
    "autotermination_minutes": 30,          # 30분 미사용 시 자동 종료 (비용 절감)
    "spark_conf": {
        "spark.databricks.io.cache.enabled": "true"  # 디스크 캐시로 쿼리 속도 향상
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 첫 번째 클러스터 생성

# COMMAND ----------

# 클러스터 생성 코드
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.compute import DataSecurityMode


w = WorkspaceClient()


cluster = w.clusters.create_and_wait(
    cluster_name=cluster_config["cluster_name"],
    spark_version=cluster_config["spark_version"],
    node_type_id=cluster_config["node_type_id"],
    num_workers=cluster_config["num_workers"],
    autotermination_minutes=cluster_config["autotermination_minutes"],
    spark_conf=cluster_config["spark_conf"],
    data_security_mode=DataSecurityMode.DATA_SECURITY_MODE_STANDARD,
)

# COMMAND ----------

# --- SQL Warehouse 생성 (26장 서빙 환경의 Statement Execution API용) ---
from databricks.sdk.service.sql import CreateWarehouseRequestWarehouseType


warehouse = w.warehouses.create_and_wait(
    name="smartfactory-agent-wh",
    cluster_size="2X-Small",                 # 에이전트 도구의 단건 쿼리 처리용 (최소 사양)
    max_num_clusters=1,                      # 실습용 1개 (운영은 2~4개)
    auto_stop_mins=15,                       # 15분 미사용 시 자동 중지
    warehouse_type=CreateWarehouseRequestWarehouseType.PRO,
    enable_serverless_compute=True,          # Serverless SQL Warehouse (권장)
)
print(f"✅ SQL Warehouse 생성 완료: {warehouse.name} (ID: {warehouse.id})")
print(f"   → 26장 Model Serving 엔드포인트 환경변수에 설정:")
print(f"     DATABRICKS_SQL_WAREHOUSE_ID = {warehouse.id}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 Cluster 환경 확인

# COMMAND ----------

# DBTITLE 1,3.3 Cluster 환경 확인
# Databricks Runtime 버전 확인
import sys
print(f"Python: {sys.version}")
print(f"Spark: {spark.version}")


# Unity Catalog 연결 확인
display(spark.sql("SHOW CATALOGS"))


# smartfactory 카탈로그 접근 확인
spark.sql("USE CATALOG smartfactory")
display(spark.sql("SHOW SCHEMAS"))
