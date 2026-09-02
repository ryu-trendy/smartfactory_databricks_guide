# Databricks notebook source
# MAGIC %md
# MAGIC # 10장: Unity Catalog — 데이터 거버넌스

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10.2 컬럼 마스킹 — 민감 데이터 자동 보호

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 컬럼 마스킹 함수 정의
# MAGIC CREATE OR REPLACE FUNCTION smartfactory.analytics.mask_employee_id(
# MAGIC     employee_id STRING
# MAGIC ) RETURNS STRING
# MAGIC RETURN CASE
# MAGIC     WHEN IS_MEMBER('de-team') OR IS_MEMBER('admin')
# MAGIC     THEN employee_id                          -- DE팀과 관리자는 실제 사번 조회
# MAGIC     ELSE CONCAT(LEFT(employee_id, 2), '***')  -- 그 외는 앞 2자리만 표시 (예: EM***)
# MAGIC END;
# MAGIC
# MAGIC -- 테이블 생성 시 마스킹 정책 적용
# MAGIC CREATE TABLE smartfactory.analytics.maintenance_log (
# MAGIC     work_order_id  STRING,
# MAGIC     employee_id    STRING MASK smartfactory.analytics.mask_employee_id,
# MAGIC     equipment_id   STRING,
# MAGIC     work_type      STRING,
# MAGIC     completed_at   TIMESTAMP
# MAGIC );
# MAGIC
# MAGIC -- 테스트: 역할에 따라 다른 결과가 반환됨
# MAGIC SELECT employee_id FROM smartfactory.analytics.maintenance_log LIMIT 5;
# MAGIC -- DE팀 조회 결과:  EM12345 (실제 사번)
# MAGIC -- 분석가 조회 결과: EM***   (마스킹)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10.4 감사 로그(Audit Log) 활용

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 감사 로그 조회 (system.access.audit)
# MAGIC -- 최근 24시간 데이터 접근 이벤트
# MAGIC SELECT
# MAGIC     event_time,
# MAGIC     user_identity.email  AS user_email,
# MAGIC     action_name,         -- getTable, executeQuery, updateTable 등
# MAGIC     request_params.table_full_name AS table_name,
# MAGIC     source_ip_address,
# MAGIC     response.status_code AS status
# MAGIC FROM system.access.audit
# MAGIC WHERE event_date >= DATEADD(DAY, -1, CURRENT_DATE())
# MAGIC   AND action_name IN ('getTable', 'executeQuery', 'updateTable', 'deleteTable')
# MAGIC ORDER BY event_time DESC
# MAGIC LIMIT 100;
# MAGIC
# MAGIC -- 비정상 접근 패턴 감지: 야간 접근
# MAGIC SELECT
# MAGIC     user_identity.email AS user_email,
# MAGIC     COUNT(*) AS access_count,
# MAGIC     MIN(event_time) AS first_access,
# MAGIC     MAX(event_time) AS last_access
# MAGIC FROM system.access.audit
# MAGIC WHERE (HOUR(event_time) >= 22 OR HOUR(event_time) <= 6)  -- 야간 시간대
# MAGIC   AND event_date >= DATEADD(DAY, -7, CURRENT_DATE())
# MAGIC   AND action_name = 'executeQuery'
# MAGIC GROUP BY user_identity.email
# MAGIC HAVING COUNT(*) > 10  -- 10회 이상 접근
# MAGIC ORDER BY access_count DESC;
# MAGIC
# MAGIC -- 특정 민감 테이블 접근 모니터링
# MAGIC SELECT
# MAGIC     DATE(event_time) AS access_date,
# MAGIC     user_identity.email AS user,
# MAGIC     COUNT(*) AS query_count
# MAGIC FROM system.access.audit
# MAGIC WHERE request_params.table_full_name = 'smartfactory.ml.pdm_labels'
# MAGIC   AND event_date >= DATEADD(DAY, -30, CURRENT_DATE())
# MAGIC GROUP BY DATE(event_time), user_identity.email
# MAGIC ORDER BY access_date DESC, query_count DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10.5 데이터 계약(Data Contract)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 데이터 계약을 테이블 속성으로 표현
# MAGIC ALTER TABLE smartfactory.analytics.oee_daily
# MAGIC SET TBLPROPERTIES (
# MAGIC     -- SLA 계약
# MAGIC     'contract.sla.refresh_time'    = '06:00 KST',
# MAGIC     'contract.sla.max_delay_hours' = '2',
# MAGIC     -- 품질 계약
# MAGIC     'contract.quality.null_cols'   = 'none',
# MAGIC     'contract.quality.min_rows'    = '500',
# MAGIC     -- 스키마 계약
# MAGIC     'contract.schema.version'      = '1.2.0',
# MAGIC     'contract.schema.compatibility' = 'backward',
# MAGIC     -- 소유권
# MAGIC     'contract.owner'               = 'data-engineering@company.com',
# MAGIC     'contract.consumers'           = 'analytics,ml-engineering'
# MAGIC );
# MAGIC
# MAGIC -- 데이터 계약 검증 함수
# MAGIC
# MAGIC
# MAGIC -- 데이터 계약 검증 함수 (순수 SQL — Python UDF에서는 spark 세션 접근 불가)
# MAGIC
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION smartfactory.analytics.validate_oee_contract()
# MAGIC RETURNS TABLE (check_name STRING, status STRING, detail STRING)
# MAGIC RETURN
# MAGIC   -- 1. 오늘 데이터 존재 여부 검증
# MAGIC   SELECT 
# MAGIC     '데이터 존재' AS check_name,
# MAGIC     CASE WHEN cnt > 0 THEN 'PASS' ELSE 'FAIL' END AS status,
# MAGIC     CASE WHEN cnt > 0 
# MAGIC          THEN CONCAT('오늘 ', cnt, '건 존재')
# MAGIC          ELSE '오늘 데이터 없음' 
# MAGIC     END AS detail
# MAGIC   FROM (SELECT COUNT(*) AS cnt 
# MAGIC         FROM smartfactory.analytics.oee_daily 
# MAGIC         WHERE CAST(production_date AS DATE) = CURRENT_DATE())
# MAGIC   UNION ALL
# MAGIC   -- 2. quality NULL 검증
# MAGIC   SELECT
# MAGIC     'NULL 검사' AS check_name,
# MAGIC     CASE WHEN cnt = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
# MAGIC     CASE WHEN cnt = 0 
# MAGIC          THEN 'quality NULL 없음'
# MAGIC          ELSE CONCAT('quality NULL ', cnt, '건') 
# MAGIC     END AS detail
# MAGIC   FROM (SELECT COUNT(*) AS cnt 
# MAGIC         FROM smartfactory.analytics.oee_daily 
# MAGIC         WHERE quality IS NULL)
# MAGIC   UNION ALL
# MAGIC   -- 3. 최소 행 수 검증 (contract.quality.min_rows = 500)
# MAGIC   SELECT
# MAGIC     '최소 행 수' AS check_name,
# MAGIC     CASE WHEN cnt >= 500 THEN 'PASS' ELSE 'FAIL' END AS status,
# MAGIC     CONCAT('총 ', cnt, '건 (기준: 500건 이상)') AS detail
# MAGIC   FROM (SELECT COUNT(*) AS cnt 
# MAGIC         FROM smartfactory.analytics.oee_daily);
# MAGIC
# MAGIC -- 계약 검증 실행
# MAGIC SELECT * FROM smartfactory.analytics.validate_oee_contract();
# MAGIC