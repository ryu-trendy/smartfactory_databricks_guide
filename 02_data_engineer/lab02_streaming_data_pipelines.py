# Databricks notebook source
# MAGIC %md
# MAGIC # 11장: Auto Loader — 클라우드 스토리지에서 실시간 데이터 수집

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11.2 Auto Loader 구현

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, col
from pyspark.sql.types import *


LANDING_PATH    = "/Volumes/smartfactory/default/data/landing/sensor_events/"
CHECKPOINT_PATH = "/Volumes/smartfactory/default/data/checkpoint/sensor_event/"


sensor_schema = StructType([
    StructField("event_id",       StringType(),    False), 
    StructField("equipment_id",   StringType(),    False),
    StructField("line_id",        StringType(),    True), 
    StructField("equipment_type", StringType(),    False),
    StructField("event_time",     TimestampType(), False), 
    StructField("temperature_c",  DoubleType(),    True),
    StructField("vibration_ms2",  DoubleType(),    True),
    StructField("pressure_bar",   DoubleType(),    True),
    StructField("rpm",            DoubleType(),    True),
    StructField("quality_score",  DoubleType(),    True),
    StructField("ingested_at",    TimestampType(), True), 
])


sensor_stream = (
    spark.readStream
    .format("cloudFiles")                           # Auto Loader 포맷
    .option("cloudFiles.format", "parquet")            # 소스 파일 형식
    .option("cloudFiles.schemaLocation", CHECKPOINT_PATH + "schema/")
    .option("cloudFiles.maxFilesPerTrigger", 1000)  # 트리거당 최대 파일 수
    .schema(sensor_schema)
    .load(LANDING_PATH)
    .withColumn("_ingestion_time", current_timestamp())
    .withColumn("_source_file",    col("_metadata.file_path"))
)


query = (
    sensor_stream.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH + "main/")
    .trigger(availableNow=True)   # 배치 모드: 현재 파일 모두 처리 후 종료


    .toTable("smartfactory.raw.sensor_events_autoloader")
)
query.awaitTermination()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 11.3 스키마 진화와 Quarantine 패턴

# COMMAND ----------

# Quarantine(격리) 패턴: 오류 데이터를 별도 테이블에 격리
from pyspark.sql.functions import col, current_timestamp


LANDING_PATH    = "/Volumes/smartfactory/default/data/landing/sensor_events/"
CHECKPOINT_PATH = "/Volumes/smartfactory/default/data/checkpoint/sensor_event/"


# 1. 소스 읽기 (기존 코드와 동일)
sensor_stream_with_rescue = (    
    spark.readStream    
    .format("cloudFiles")    
    .option("cloudFiles.format",              "parquet")    
    .option("cloudFiles.schemaLocation",      CHECKPOINT_PATH + "schema/")    
    .option("cloudFiles.schemaEvolutionMode", "rescue")  
    .option("rescuedDataColumn",              "_rescued_data")    
    .load(LANDING_PATH)    
    .withColumn("_ingestion_time", current_timestamp())
)


# 2. 마이크로 배치 단위로 두 테이블에 나눠서 쓰는 함수 정의
def route_quarantine_data(batch_df, batch_id):
    # 데이터를 메모리에 캐싱하여 여러 번 필터링할 때 재연산되는 것을 방지
    batch_df.cache()
    
    # 2-1. 정상 데이터 → Bronze 테이블
    batch_df.filter(col("_rescued_data").isNull()) \
        .write.format("delta").mode("append") \
        .saveAsTable("smartfactory.raw.nomal_sensor_events")
        
    # 2-2. 격리 데이터 → Quarantine 테이블
    batch_df.filter(col("_rescued_data").isNotNull()) \
        .write.format("delta").mode("append") \
        .saveAsTable("smartfactory.raw.quarantine_sensor_events")
        
    batch_df.unpersist() # 캐시 해제




query = (
    sensor_stream_with_rescue.writeStream
    .foreachBatch(route_quarantine_data)
    .option("checkpointLocation", CHECKPOINT_PATH + "quarantine/") 
    .trigger(availableNow=True) 
    .start())

# COMMAND ----------

# MAGIC %md
# MAGIC # 12장: Lakeflow Pipelines — 선언형 데이터 파이프라인 구축

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12.2 파이프라인 생성 — Python SDK

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.pipelines import PipelineCluster, NotebookLibrary

w = WorkspaceClient()
username = w.current_user.me().user_name
print(username)
SOURCE_PATH = f"/Workspace/Users/{username}/smartfactory_databricks_guide/02_data_engineer/lab02_sdp_pipeline.py"

pipeline = w.pipelines.create(
    name="smartfactory-sensor-pipeline-test",
    libraries=[PipelineLibrary(file=FileLibrary(path=SOURCE_PATH))],
    clusters=[
        PipelineCluster(label="default", num_workers=2,
                        node_type_id="m5d.large")
    ],
    catalog="smartfactory",
    target="raw",
    continuous=False,      # False=트리거 실행, True=지속 실행
    development=False,      # 개발 모드
)
pipeline_id = pipeline.pipeline_id
print(f"파이프라인 ID: {pipeline_id}")

# 증분 실행
update = w.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=False)
print(f"업데이트 시작: {update.update_id}")