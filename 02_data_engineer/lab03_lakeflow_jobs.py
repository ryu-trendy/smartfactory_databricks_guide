# Databricks notebook source
# MAGIC %md
# MAGIC # 13장: Lakeflow Jobs — 워크플로우 오케스트레이션

# COMMAND ----------

# DBTITLE 1,13.1 스마트팩토리 일간 파이프라인 DAG
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (    Task, NotebookTask, PipelineTask, TaskDependency,    CronSchedule, JobCluster, JobEmailNotifications, PauseStatus)
from databricks.sdk.service.compute import AutoScale, ClusterSpec, AwsAttributes, AwsAvailability, DataSecurityMode

w = WorkspaceClient()

username = w.current_user.me().user_name
base = f"/Workspace/Users/{username}/smartfactory_databricks_guide/02_data_engineer"

bronze_notebook  = f"{base}/lab03_bronze_pipeline"
silver_notebook  = f"{base}/lab03_silver_pipeline"
gold_notebook    = f"{base}/lab03_gold_pipeline"
feature_notebook = f"{base}/lab03_feature_pipeline"

job = w.jobs.create(    
                    name="smartfactory-daily-pipeline",    
                    schedule=CronSchedule(        
                                          quartz_cron_expression="0 0 6 * * ?",  # 매일 오전 6시        
                                          timezone_id="Asia/Seoul",        
                                          pause_status=PauseStatus.UNPAUSED,    
                                          ),    
                    job_clusters=[        
                                  JobCluster(            
                                             job_cluster_key="single-user-cluster",            
                                             new_cluster=ClusterSpec(                
                                                                     spark_version="18.x-scala2.13",                
                                                                     node_type_id="m5d.large",                
                                                                     autoscale=AutoScale(min_workers=2, max_workers=8),                
                                                                     aws_attributes=AwsAttributes(availability=AwsAvailability.ON_DEMAND),
                                                                     data_security_mode=DataSecurityMode.SINGLE_USER,           
                                                                     ),        
                                             )    
                                  ],    
                    tasks=[        
                           Task(            
                                task_key="ingest_sensor_data",            
                                notebook_task=NotebookTask(                
                                                           notebook_path=bronze_notebook,                    
                                                           ),            
                                job_cluster_key="single-user-cluster",            
                                timeout_seconds=1800,        
                                ),        
                           Task(            
                                task_key="sensor_clean_job",            
                                notebook_task=NotebookTask(                
                                                           notebook_path=silver_notebook,                   
                                                           ),            
                                job_cluster_key="single-user-cluster",            
                                depends_on=[TaskDependency(task_key="ingest_sensor_data")],        
                                ),        
                           Task(            
                                task_key="aggregate_oee",            
                                notebook_task=NotebookTask(                
                                                           notebook_path=gold_notebook,            
                                                           ),            
                                job_cluster_key="single-user-cluster",            
                                depends_on=[TaskDependency(task_key="sensor_clean_job")],        
                                ),        
                           Task(            
                                task_key="compute_features",            
                                notebook_task=NotebookTask(                
                                                           notebook_path=feature_notebook,            
                                                           ),            
                                job_cluster_key="single-user-cluster",            
                                depends_on=[TaskDependency(task_key="sensor_clean_job")],        
                                )
                           ],    
                    email_notifications=JobEmailNotifications(on_failure=[username]),    
                    max_concurrent_runs=1,)
print(f"Job 생성: {job.job_id}")