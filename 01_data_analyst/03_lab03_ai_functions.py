# Databricks notebook source
# 8장: AI SQL 함수와 고급 분석 — LLM을 SQL 속으로
# DBR 17.3 LTS (Spark 4.0.0 / Python 3.12.3)

# COMMAND ----------

# MAGIC %md
# MAGIC # 8장: AI SQL 함수와 고급 분석
# MAGIC
# MAGIC Databricks Foundation Models API를 SQL 안에서 직접 호출합니다.
# MAGIC - `ai_query()`: 커스텀 엔드포인트 또는 Foundation Models 호출
# MAGIC - `ai_classify()`: 텍스트 분류
# MAGIC - `ai_summarize()`: 텍스트 요약
# MAGIC - `ai_extract()`: 구조화된 정보 추출
# MAGIC
# MAGIC **전제 조건:** SQL Warehouse에서 실행 (Serverless 권장)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.1.1 ai_query() 함수 — 텍스트 생성

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ai_query(): 텍스트 생성 (LLM 호출)
# MAGIC -- 센서 이상 이벤트에 대한 자연어 설명 생성
# MAGIC SELECT
# MAGIC     equipment_id,
# MAGIC     event_time,
# MAGIC     temperature_c,
# MAGIC     vibration_ms2,
# MAGIC     ai_query(
# MAGIC         -- 사용할 LLM 엔드포인트
# MAGIC         'databricks-meta-llama-3-1-8b-instruct',
# MAGIC         -- 프롬프트 (설비 정보와 센서값 포함)
# MAGIC         CONCAT(
# MAGIC             '설비 ', equipment_id, '에서 이상이 감지되었습니다. ',
# MAGIC             '온도: ', ROUND(temperature_c, 1), '°C, ',
# MAGIC             '진동: ', ROUND(vibration_ms2, 2), ' mm/s. ',
# MAGIC             '이 상태를 30자 이내로 간결하게 설명해 주세요.'
# MAGIC         )
# MAGIC     ) AS anomaly_description
# MAGIC FROM smartfactory.processed.sensor_clean
# MAGIC WHERE vibration_ms2 > 5.0
# MAGIC   AND event_time >= DATEADD(DAY, -7, CURRENT_DATE())
# MAGIC LIMIT 20;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.1.2 ai_classify() 함수

# COMMAND ----------

# ──────────────────────────────────────────────────────────────
# 정비 기록 테이블 생성 (smartfactory.processed.maintenance_clean)
# ──────────────────────────────────────────────────────────────
# 목적: ai_classify / ai_extract 실습용 한국어 정비 일지 데이터 생성
# 핵심 컬럼: work_description (자연어 텍스트 — 교체 부품, 작업 시간, 담당 기술자 포함)
# 생성 규모: 90일 × 일평균 2.5건 ≈ 약 225건


from datetime import datetime, date, timedelta
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, TimestampType
import random


random.seed(42)


# 대상 설비 목록: (설비ID, 설비유형, 라인ID)
EQUIPMENT_LIST = (
    [(f"EQ{i:03d}", "CNC", "LINE01") for i in range(1, 11)] +
    [(f"EQ{i:03d}", "AOI", "LINE03") for i in range(11, 21)]
)


# 고장 유형별 정비 일지 템플릿 및 교체 부품 사전
# → ai_classify가 분류할 5가지 카테고리와 1:1 대응
CATEGORY_CONFIG = {
    "기계 마모": {
        "templates": [
            "{eq} 연결부 베어링 마모 감지로 교체 완료.",
            "{eq} 기어 마모로 이상 소음 발생 교체 진행.",
        ],
        "parts": ["6204 베어링", "헬리컨 기어 세트", "V벨트", "체인 스프로켓"],
    },
    "전기 고장": {
        "templates": [
            "{eq} 모터 과열로 인버터 교체 완료.",
            "{eq} 전기 계통 단락 발생 포시 교체.",
        ],
        "parts": ["IGBT 모듈", "10A 퓨즈", "DC 릴레이", "전압 변환기"],
    },
    "오염/누유": {
        "templates": [
            "{eq} 절삭유 누유 감지로 씨링 교체.",
            "{eq} 에어필터 오염으로 교체 수행.",
        ],
        "parts": ["오일씨 세트", "에어필터", "오링 팡킹", "절삭유 필터"],
    },
    "제어 오류": {
        "templates": [
            "{eq} PLC 통신 오류로 제어보드 교체.",
            "{eq} 센서 드리프트 감지 센서 교체.",
        ],
        "parts": ["온도센서 PT100", "PLC 배터리", "인코더", "근접센서"],
    },
    "예방 정비": {
        "templates": [
            "{eq} 월간 정기 점검 윤활유 교체.",
            "{eq} 분기 소모품 교체 완료.",
        ],
        "parts": ["그리스 카트리지", "에어 필터 엘리먼트", "고무 패드", "절연 테이프"],
    },
}
TECHNICIANS = ["김영호", "이철수", "박민준", "최재혁", "정현우"]


# 전체 (카테고리, 템플릿) 조합 리스트 — 랜덤 선택용
FAILURE_SCENARIOS = [
    (cat, tmpl)
    for cat, cfg in CATEGORY_CONFIG.items()
    for tmpl in cfg["templates"]
]






def generate_maintenance_clean(days: int = 90) -> list:
    """90일치 정비 기록 생성 (하루 1~4건 랜덤 발생)"""
    records = []
    today  = date.today()
    rec_no = 1
    for d in range(days):
        work_date = today - timedelta(days=days - d)
        # 하루 1~4건 랜덤 발생
        for _ in range(random.choices([1, 2, 3, 4], weights=[3, 4, 2, 1])[0]):
            eq_id, eq_type, line_id = random.choice(EQUIPMENT_LIST)
            cat, tmpl = random.choice(FAILURE_SCENARIOS)
            part = random.choice(CATEGORY_CONFIG[cat]["parts"])
            dur  = random.choice([20, 30, 45, 60, 90, 120])
            tech = random.choice(TECHNICIANS)
            desc = tmpl.format(eq=eq_id) + f" 교체 부품: {part}, 작업 시간: {dur}분, 담당 기술자: {tech}"
            records.append((
                f"MAINT_{eq_id}_{work_date.strftime('%Y%m%d')}_{rec_no:04d}",
                eq_id, eq_type, line_id, desc,
                work_date, dur, tech, cat,
                datetime.now(),
            ))
            rec_no += 1
    return records


# 데이터 생성 실행
records = generate_maintenance_clean(days=90)
print(f"생성된 레코드 수: {len(records):,} (90일치)")


# 테이블 스키마 정의
schema = StructType([
    StructField("maintenance_id",    StringType(),  False),  # 정비 고유 ID
    StructField("equipment_id",      StringType(),  False),
    StructField("equipment_type",    StringType(),  True),
    StructField("line_id",           StringType(),  True),
    StructField("work_description",  StringType(),  True),   # ai_classify/ai_extract 대상 텍스트
    StructField("work_date",         DateType(),    False),
    StructField("work_duration_min", IntegerType(), True),   # 작업 시간(분)
    StructField("technician",        StringType(),  True),   # 담당 기술자
    StructField("failure_category",  StringType(),  True),   # 정답 라벨 (ai_classify 검증용)
    StructField("created_at",        TimestampType(), True),
])


# Delta 테이블로 저장
df_maint = spark.createDataFrame(records, schema)
df_maint.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("smartfactory.processed.maintenance_clean")


print("✅ maintenance_clean 저장 완료!")
# 카테고리별 건수 확인 (ai_classify 결과와 비교 가능)
display(
    df_maint.groupBy("failure_category").count()
            .orderBy("count", ascending=False)
)


# COMMAND ----------

# MAGIC %sql
# MAGIC -- ai_classify: 텍스트를 지정된 카테고리로 분류
# MAGIC -- 정비 기록을 고장 유형으로 자동 분류
# MAGIC SELECT
# MAGIC     maintenance_id,
# MAGIC     equipment_id,
# MAGIC     work_description,
# MAGIC     ai_classify(
# MAGIC         work_description,  -- 분류할 텍스트
# MAGIC         ARRAY(             -- 분류 카테고리
# MAGIC             '전기 고장',
# MAGIC             '기계 마모',
# MAGIC             '오염/누유',
# MAGIC             '제어 오류',
# MAGIC             '예방 정비'
# MAGIC         )
# MAGIC     ) AS failure_category
# MAGIC FROM smartfactory.processed.maintenance_clean
# MAGIC WHERE work_date >= DATEADD(DAY, -90, CURRENT_DATE());
# MAGIC
# MAGIC -- 분류 결과 집계 (파레토 분석)
# MAGIC WITH classified AS (
# MAGIC     SELECT
# MAGIC         equipment_type,
# MAGIC         ai_classify(
# MAGIC             work_description,
# MAGIC             ARRAY('전기 고장', '기계 마모', '오염/누유', '제어 오류', '예방 정비')
# MAGIC         ) AS failure_category
# MAGIC     FROM smartfactory.processed.maintenance_clean
# MAGIC     WHERE work_date >= DATEADD(DAY, -30, CURRENT_DATE())
# MAGIC )
# MAGIC SELECT
# MAGIC     equipment_type,
# MAGIC     failure_category,
# MAGIC     COUNT(*) AS count,
# MAGIC     ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY equipment_type), 1) AS pct
# MAGIC FROM classified
# MAGIC GROUP BY equipment_type, failure_category
# MAGIC ORDER BY equipment_type, count DESC;
# MAGIC

# COMMAND ----------

# PySpark에서 AI 함수 사용
from pyspark.sql.functions import expr, col

df_anomaly = spark.sql("""
    SELECT equipment_id, equipment_type, line_id,
           AVG(temperature_c) AS avg_temp,
           AVG(vibration_ms2) AS avg_vib,
           COUNT(*) AS anomaly_count
    FROM smartfactory.processed.sensor_clean
    WHERE is_anomaly = TRUE
    GROUP BY equipment_id, equipment_type, line_id
    HAVING anomaly_count >= 5
""")

df_with_ai = df_anomaly.withColumn(
    "ai_risk_level",
    expr("""
        ai_classify(
            CONCAT('설비:', equipment_type, ', 평균온도:', ROUND(avg_temp,1),
                   '°C, 평균진동:', ROUND(avg_vib,3), 'm/s², 이상횟수:', anomaly_count),
            ARRAY('HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK')
        )
    """)
)

display(df_with_ai.orderBy(col("anomaly_count").desc()).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.1.3 ai_extract() 함수

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ai_extract: 텍스트에서 구조화된 정보 추출
# MAGIC -- 정비 기록에서 교체 부품과 작업 시간 추출
# MAGIC SELECT
# MAGIC     maintenance_id,
# MAGIC     work_description,
# MAGIC     ai_extract(
# MAGIC         work_description,
# MAGIC         ARRAY('교체_부품', '작업_시간_분', '담당_기술자')
# MAGIC     ) AS extracted_info
# MAGIC FROM smartfactory.processed.maintenance_clean
# MAGIC WHERE work_date >= DATEADD(DAY, -7, CURRENT_DATE())
# MAGIC   AND work_description LIKE '%교체%'
# MAGIC LIMIT 50;
# MAGIC
# MAGIC
# MAGIC -- STRUCT 필드 접근으로 구조화 데이터 추출
# MAGIC WITH extracted AS (
# MAGIC     SELECT
# MAGIC         maintenance_id,
# MAGIC         equipment_id,
# MAGIC         ai_extract(
# MAGIC             work_description,
# MAGIC             ARRAY('교체_부품', '작업_시간_분')
# MAGIC         ) AS info
# MAGIC     FROM smartfactory.processed.maintenance_clean
# MAGIC     WHERE work_date >= DATEADD(DAY, -30, CURRENT_DATE())
# MAGIC )
# MAGIC SELECT
# MAGIC     maintenance_id,
# MAGIC     equipment_id,
# MAGIC     info.`교체_부품`    AS replaced_part,
# MAGIC     info.`작업_시간_분` AS work_minutes
# MAGIC FROM extracted
# MAGIC WHERE info.`교체_부품` IS NOT NULL;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.2 고급 분석 기법 — 윈도우 함수 이상 감지

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 이동 평균과 표준편차를 활용한 이상 감지
# MAGIC WITH stats AS (
# MAGIC     SELECT
# MAGIC         equipment_id,
# MAGIC         event_time,
# MAGIC         temperature_c,
# MAGIC         -- 1시간(360개 레코드) 이동 평균
# MAGIC         AVG(temperature_c) OVER (
# MAGIC             PARTITION BY equipment_id
# MAGIC             ORDER BY event_time
# MAGIC             ROWS BETWEEN 359 PRECEDING AND CURRENT ROW
# MAGIC         ) AS temp_1h_avg,
# MAGIC         -- 1시간 이동 표준편차
# MAGIC         STDDEV(temperature_c) OVER (
# MAGIC             PARTITION BY equipment_id
# MAGIC             ORDER BY event_time
# MAGIC             ROWS BETWEEN 359 PRECEDING AND CURRENT ROW
# MAGIC         ) AS temp_1h_std
# MAGIC     FROM smartfactory.processed.sensor_clean
# MAGIC     WHERE event_time >= DATEADD(DAY, -7, CURRENT_DATE())
# MAGIC )
# MAGIC SELECT
# MAGIC     equipment_id,
# MAGIC     event_time,
# MAGIC     temperature_c,
# MAGIC     temp_1h_avg,
# MAGIC     ROUND((temperature_c - temp_1h_avg) / NULLIF(temp_1h_std, 0), 2) AS z_score,
# MAGIC     CASE
# MAGIC         WHEN ABS((temperature_c - temp_1h_avg) / NULLIF(temp_1h_std, 0)) > 3.0
# MAGIC         THEN 'ANOMALY'
# MAGIC         ELSE 'NORMAL'
# MAGIC     END AS anomaly_flag
# MAGIC FROM stats
# MAGIC WHERE ABS((temperature_c - temp_1h_avg) / NULLIF(temp_1h_std, 0)) > 3.0
# MAGIC ORDER BY event_time;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.3 실습 Lab — AI 품질 분류 파이프라인

# COMMAND ----------

# MAGIC %sql
# MAGIC -- AI 품질 분류 파이프라인
# MAGIC -- Step 1: 원본 불량 데이터 확인
# MAGIC SELECT inspection_id, equipment_id, defect_type, severity, inspection_time
# MAGIC FROM smartfactory.processed.quality_inspections
# MAGIC WHERE passed = false
# MAGIC   AND DATE(inspection_time) >= DATEADD(DAY, -7, CURRENT_DATE())
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Step 2: ai_classify로 불량 유형 분류 (영어 코드 → 한국어 카테고리)
# MAGIC CREATE OR REPLACE TABLE smartfactory.analytics.defect_classified AS
# MAGIC SELECT
# MAGIC   inspection_id,
# MAGIC   equipment_id,
# MAGIC   defect_type,
# MAGIC   severity,
# MAGIC   inspection_time,
# MAGIC   ai_classify(
# MAGIC     CONCAT('불량유형: ', defect_type, ', 심각도: ', severity),
# MAGIC     ARRAY('치수불량', '표면결함', '재료결함', '조립오류', '기능불량')
# MAGIC   ) AS defect_category,
# MAGIC   ai_query('databricks-meta-llama-3-3-70b-instruct',
# MAGIC     CONCAT('설비 검사에서 ', defect_type, ' 유형의 ', severity, ' 등급 불량이 발견되었습니다. 이를 한 문장으로 요약하세요.')
# MAGIC   ) AS defect_summary
# MAGIC FROM smartfactory.processed.quality_inspections
# MAGIC WHERE passed = false
# MAGIC   AND DATE(inspection_time) >= DATEADD(DAY, -7, CURRENT_DATE())
# MAGIC LIMIT 1000; -- 비용 제어

# COMMAND ----------

# MAGIC %sql
# MAGIC FROM smartfactory.analytics.defect_classified

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- Step 3: 불량 유형별 파레토 분석
# MAGIC SELECT
# MAGIC   defect_category,
# MAGIC   COUNT(*) AS count,
# MAGIC   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct,
# MAGIC   SUM(COUNT(*)) OVER (ORDER BY COUNT(*) DESC) AS cumulative
# MAGIC FROM smartfactory.analytics.defect_classified
# MAGIC GROUP BY defect_category
# MAGIC ORDER BY count DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC - **9장**: `01_data_analyst/04_agent_analyst.py` (SQL 분석 에이전트)