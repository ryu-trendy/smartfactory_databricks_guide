# Databricks notebook source
# MAGIC %md
# MAGIC #2장 Unity Catalog — 데이터 거버넌스의 시작

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 스마트팩토리 코리아 카탈로그 설정

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Unity Catalog 기본 구조 생성
# MAGIC -- 이 SQL은 Databricks SQL 또는 노트북 %sql 셀에서 실행합니다
# MAGIC
# MAGIC -- 1단계: 카탈로그 생성
# MAGIC CREATE CATALOG IF NOT EXISTS smartfactory
# MAGIC COMMENT '스마트팩토리 코리아 데이터 플랫폼 전용 카탈로그';
# MAGIC
# MAGIC -- 2단계: 스키마(레이어) 생성
# MAGIC CREATE SCHEMA IF NOT EXISTS smartfactory.raw
# MAGIC COMMENT 'Bronze 레이어: 원시 데이터, 가공 없이 저장';
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS smartfactory.processed
# MAGIC COMMENT 'Silver 레이어: 정제·검증된 데이터';
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS smartfactory.analytics
# MAGIC COMMENT 'Gold 레이어: 집계·분석용 데이터';
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS smartfactory.ml
# MAGIC COMMENT 'ML 레이어: 피처 스토어, 추론 로그';
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS smartfactory.ai
# MAGIC COMMENT 'AI 레이어: 벡터 인덱스, 에이전트 로그';
# MAGIC
# MAGIC -- 현재 카탈로그 확인
# MAGIC SHOW SCHEMAS IN smartfactory;
# MAGIC
# MAGIC -- volume 생성
# MAGIC CREATE VOLUME IF NOT EXISTS smartfactory.default.data;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Unity Catalog 권한 체계 실전 구성

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 1. 그룹 생성 및 사용자 추가
# MAGIC -- ※ Unity Catalog 환경에서 그룹 생성은 SQL로 지원되지 않습니다.
# MAGIC --    Account 콘솔(accounts.cloud.databricks.com) 또는 SCIM을 통해
# MAGIC --    Account-level 그룹(data_analysts, data_engineers, ml_engineers)을
# MAGIC --    먼저 생성하고 멤버를 등록한 뒤 아래 GRANT 명령을 실행하세요.
# MAGIC
# MAGIC -- 3. 카탈로그 수준 권한
# MAGIC GRANT USE CATALOG ON CATALOG smartfactory TO data_analysts;
# MAGIC GRANT USE CATALOG ON CATALOG smartfactory TO data_engineers;
# MAGIC GRANT ALL PRIVILEGES ON CATALOG smartfactory TO data_engineers;
# MAGIC
# MAGIC -- 4. 스키마 수준 권한
# MAGIC GRANT USE SCHEMA, SELECT ON SCHEMA smartfactory.analytics TO data_analysts;
# MAGIC GRANT ALL PRIVILEGES ON SCHEMA smartfactory.processed TO data_engineers;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 역할 기반 접근 제어(RBAC)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 역할별 권한 부여 (Databricks SQL에서 실행)
# MAGIC -- 데이터 엔지니어링 팀
# MAGIC GRANT USE CATALOG ON CATALOG smartfactory        TO data_engineers;
# MAGIC GRANT USE SCHEMA  ON SCHEMA  smartfactory.raw    TO data_engineers;
# MAGIC GRANT SELECT, MODIFY, CREATE TABLE
# MAGIC                   ON SCHEMA  smartfactory.raw    TO data_engineers;
# MAGIC GRANT USE SCHEMA  ON SCHEMA  smartfactory.processed TO data_engineers;
# MAGIC GRANT SELECT, MODIFY, CREATE TABLE
# MAGIC                   ON SCHEMA  smartfactory.processed TO data_engineers;
# MAGIC
# MAGIC -- 데이터 사이언스 팀 (raw 직접 접근 불가 - processed를 통해서만)
# MAGIC GRANT USE CATALOG ON CATALOG smartfactory           TO data_scientist;
# MAGIC GRANT USE SCHEMA  ON SCHEMA  smartfactory.processed TO data_scientist;
# MAGIC GRANT SELECT      ON SCHEMA  smartfactory.processed TO data_scientist;
# MAGIC GRANT USE SCHEMA  ON SCHEMA  smartfactory.ml        TO data_scientist;
# MAGIC GRANT SELECT, MODIFY, CREATE TABLE
# MAGIC                   ON SCHEMA  smartfactory.ml        TO data_scientist;
# MAGIC
# MAGIC -- 분석가 (analytics만 읽기)
# MAGIC GRANT USE CATALOG ON CATALOG smartfactory           TO data_analysts;
# MAGIC GRANT USE SCHEMA  ON SCHEMA  smartfactory.analytics TO data_analysts;
# MAGIC GRANT SELECT      ON SCHEMA  smartfactory.analytics TO data_analysts;