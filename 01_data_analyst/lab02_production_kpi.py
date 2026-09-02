# Databricks notebook source
# 7장: OEE 대시보드 구축 — SQL로 설비 효율을 시각화하다
# DBR 17.3 LTS (Spark 4.0.0 / Python 3.12.3)

# COMMAND ----------

# MAGIC %md
# MAGIC # 7장: OEE 대시보드 구축
# MAGIC
# MAGIC ## OEE 공식
# MAGIC - **가용률(Availability)** = 실제 가동 시간 / 계획 가동 시간
# MAGIC - **성능률(Performance)** = 실제 생산량 / 이론 생산량
# MAGIC - **품질률(Quality)** = 양품 수 / 실제 생산량
# MAGIC - **OEE = 가용률 × 성능률 × 품질률**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.1 OEE 개념과 계산 공식

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE VIEW smartfactory.analytics.v_equipment_hourly AS
# MAGIC SELECT
# MAGIC     equipment_id,
# MAGIC     line_id,
# MAGIC     equipment_type,
# MAGIC     DATE(event_time)  AS production_date,
# MAGIC     HOUR(event_time)  AS production_hour,
# MAGIC     COUNT(*)          AS sensor_count,        -- 10분당 1건: 최대 6건/시간
# MAGIC     AVG(temperature_c) AS avg_temp,
# MAGIC     MAX(vibration_ms2) AS max_vib,
# MAGIC     AVG(pressure_bar)  AS avg_current,
# MAGIC     -- 가동 판별: 전류 2A 이상이면 가동 중
# MAGIC     SUM(CASE WHEN pressure_bar >= 2.0 THEN 1 ELSE 0 END)
# MAGIC         AS running_minutes_10m,               -- 가동 간격 수 (×10분 = 분)
# MAGIC     -- 이상 판별: 진동 임계값 초과
# MAGIC     SUM(CASE WHEN vibration_ms2 > 5.0 THEN 1 ELSE 0 END)
# MAGIC         AS anomaly_count
# MAGIC FROM smartfactory.processed.sensor_clean
# MAGIC GROUP BY 1, 2, 3, 4, 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.1.1 일별 OEE 계산

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 일별 OEE 집계 (Gold 레이어)
# MAGIC CREATE OR REPLACE TABLE smartfactory.analytics.oee_daily
# MAGIC USING DELTA
# MAGIC COMMENT 'Gold: 일별 OEE 집계'
# MAGIC AS
# MAGIC WITH daily_base AS (
# MAGIC     SELECT
# MAGIC         equipment_id,
# MAGIC         line_id,
# MAGIC         equipment_type,
# MAGIC         production_date,
# MAGIC         SUM(sensor_count)            AS total_records,
# MAGIC         SUM(running_minutes_10m * 10) AS total_running_min,  -- 가동 시간(분)
# MAGIC         SUM(anomaly_count)           AS total_anomalies
# MAGIC     FROM smartfactory.analytics.v_equipment_hourly
# MAGIC     GROUP BY equipment_id, line_id, equipment_type, production_date
# MAGIC ),
# MAGIC oee_calc AS (
# MAGIC     SELECT
# MAGIC         *,
# MAGIC         -- 가용률: 실제 가동 시간 / 계획 가동 시간(8시간 2교대 = 960분)
# MAGIC         ROUND(
# MAGIC             LEAST(total_running_min / 960.0, 1.0) * 100, 1
# MAGIC         ) AS availability_pct,
# MAGIC         -- 성능률: 이상 없는 시간의 비율로 근사
# MAGIC         ROUND(
# MAGIC             (1.0 - total_anomalies / GREATEST(total_records, 1)) * 95.0, 1
# MAGIC         ) AS performance_pct,
# MAGIC         97.5 AS quality_pct  -- 이 장에서는 임시값 사용
# MAGIC     FROM daily_base
# MAGIC )
# MAGIC SELECT
# MAGIC     equipment_id, line_id, equipment_type, production_date,
# MAGIC     availability_pct, performance_pct, quality_pct,
# MAGIC     ROUND(
# MAGIC         (availability_pct / 100.0) *
# MAGIC         (performance_pct  / 100.0) *
# MAGIC         (quality_pct      / 100.0) * 100, 1
# MAGIC     ) AS oee_pct,
# MAGIC     total_running_min / 60.0   AS running_hours,
# MAGIC     total_anomalies
# MAGIC FROM oee_calc;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.2 6대 손실 분석

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 손실 유형별 분석 (파레토 분석)
# MAGIC WITH loss_analysis AS (
# MAGIC     SELECT
# MAGIC         equipment_id,
# MAGIC         line_id,
# MAGIC         production_date,
# MAGIC         -- 가용률 손실 (100% - 가용률)
# MAGIC         (100.0 - availability_pct) AS availability_loss,
# MAGIC         -- 성능 손실
# MAGIC         availability_pct / 100.0 * (100.0 - performance_pct)
# MAGIC             AS performance_loss,
# MAGIC         -- 품질 손실
# MAGIC         availability_pct / 100.0 * performance_pct / 100.0
# MAGIC             * (100.0 - quality_pct) AS quality_loss
# MAGIC     FROM smartfactory.analytics.oee_daily
# MAGIC     WHERE production_date >= DATEADD(DAY, -30, CURRENT_DATE())
# MAGIC )
# MAGIC SELECT
# MAGIC     line_id,
# MAGIC     ROUND(AVG(availability_loss), 1)  AS `가용률_손실`,
# MAGIC     ROUND(AVG(performance_loss),  1)  AS `성능_손실`,
# MAGIC     ROUND(AVG(quality_loss),      1)  AS `품질_손실`,
# MAGIC     ROUND(AVG(availability_loss + performance_loss + quality_loss), 1) AS `총_손실`,
# MAGIC     CASE
# MAGIC         WHEN AVG(availability_loss) >= AVG(performance_loss)
# MAGIC          AND AVG(availability_loss) >= AVG(quality_loss) THEN '가용률 개선 필요'
# MAGIC         WHEN AVG(performance_loss) >= AVG(quality_loss) THEN '성능 개선 필요'
# MAGIC         ELSE '품질 개선 필요'
# MAGIC     END AS `우선_개선_영역`
# MAGIC FROM loss_analysis
# MAGIC GROUP BY line_id
# MAGIC ORDER BY `총_손실` DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.3 AI/BI 대시보드 구성

# COMMAND ----------

# MAGIC %sql
# MAGIC -- OEE 트렌드 차트용 쿼리 (Databricks 대시보드 데이터셋 생성시 실행)
# MAGIC -- 1. OEE 경보 설비 목록 (테이블: OEE < 65% 설비, 즉시 조치 필요)
# MAGIC SELECT
# MAGIC   equipment_id,
# MAGIC   line_id,
# MAGIC   ROUND(oee_pct, 1) AS oee_pct,
# MAGIC   ROUND(availability_pct, 1) AS availability_pct,
# MAGIC   ROUND(performance_pct, 1) AS performance_pct,
# MAGIC   ROUND(quality_pct, 1) AS quality_pct
# MAGIC FROM
# MAGIC   smartfactory.analytics.oee_daily
# MAGIC WHERE
# MAGIC   production_date = DATEADD(DAY, -1, CURRENT_DATE())
# MAGIC   AND oee_pct < 65
# MAGIC ORDER BY
# MAGIC   oee_pct ASC;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 2. 설비별 OEE 히트맵 (X=production_date, Y=equipment_id, Color=oee_pct)
# MAGIC SELECT
# MAGIC   equipment_id,
# MAGIC   production_date,
# MAGIC   ROUND(oee_pct, 1) AS oee_pct
# MAGIC FROM
# MAGIC   smartfactory.analytics.oee_daily
# MAGIC WHERE
# MAGIC   production_date >= DATEADD(DAY, -30, CURRENT_DATE())
# MAGIC ORDER BY
# MAGIC   equipment_id,
# MAGIC   Production_date;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 3. 라인별 일별 OEE 추이 (선 그래프: X=production_date, Y=avg_oee_pct, Color=line_id)
# MAGIC SELECT
# MAGIC   production_date,
# MAGIC   line_id,
# MAGIC   ROUND(AVG(oee_pct), 1) AS avg_oee_pct
# MAGIC FROM
# MAGIC   smartfactory.analytics.oee_daily
# MAGIC WHERE
# MAGIC   production_date >= DATEADD(DAY, -30, CURRENT_DATE())
# MAGIC   AND line_id IN ('LINE01', 'LINE02', 'LINE03')
# MAGIC GROUP BY
# MAGIC   production_date,
# MAGIC   line_id
# MAGIC ORDER BY
# MAGIC   production_date,
# MAGIC   Line_id;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 4. 손실 원인 분포 (파이 차트: 가용률/성능/품질 손실 비율, 30일 평균)
# MAGIC SELECT
# MAGIC   '가용률 손실' AS loss_type,
# MAGIC   ROUND(AVG(100 - availability_pct), 1) AS loss_pct
# MAGIC FROM
# MAGIC   smartfactory.analytics.oee_daily
# MAGIC WHERE
# MAGIC   production_date >= DATEADD(DAY, -30, CURRENT_DATE())
# MAGIC UNION ALL
# MAGIC SELECT
# MAGIC   '성능 손실',
# MAGIC   ROUND(AVG(100 - performance_pct), 1)
# MAGIC FROM
# MAGIC   smartfactory.analytics.oee_daily
# MAGIC WHERE
# MAGIC   production_date >= DATEADD(DAY, -30, CURRENT_DATE())
# MAGIC UNION ALL
# MAGIC SELECT
# MAGIC   '품질 손실',
# MAGIC   ROUND(AVG(100 - quality_pct), 1)
# MAGIC FROM
# MAGIC   smartfactory.analytics.oee_daily
# MAGIC WHERE
# MAGIC   production_date >= DATEADD(DAY, -30, CURRENT_DATE());

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 5. 전일 KPI 카운터 위젯용 (평균 OEE / 최저 OEE / 경보 건수)
# MAGIC SELECT
# MAGIC   ROUND(AVG(oee_pct), 1) AS avg_oee,
# MAGIC   ROUND(MIN(oee_pct), 1) AS min_oee,
# MAGIC   COUNT(
# MAGIC     CASE
# MAGIC       WHEN oee_pct < 65 THEN 1
# MAGIC     END
# MAGIC   ) AS alert_count
# MAGIC FROM
# MAGIC   smartfactory.analytics.oee_daily
# MAGIC WHERE
# MAGIC   production_date = DATEADD(DAY, -1, CURRENT_DATE());

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.4 실습 Lab — OEE 실시간 모니터링

# COMMAND ----------

# OEE 골드 테이블 생성 및 시각화
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pyspark.sql import functions as F




# Gold 테이블 생성


spark.sql("""CREATE OR REPLACE TABLE smartfactory.analytics.oee_daily
          CLUSTER BY (equipment_id, production_date)
          AS
          WITH base AS (
                SELECT    p.equipment_id,
                    p.production_date,
                    e.equipment_type,
                    e.line_id,
                    SUM(480 - p.downtime_min) / SUM(480) AS availability,
                    SUM(p.actual_qty) / NULLIF(SUM(p.planned_qty), 0) AS performance,
                    SUM(CASE WHEN q.passed = true THEN 1 ELSE 0 END) / COUNT(q.passed) AS quality,
                    SUM(p.actual_qty) AS total_actual_qty  
                FROM smartfactory.processed.production_logs p  
                JOIN smartfactory.processed.equipment_master e 
                  ON p.equipment_id = e.equipment_id  
                LEFT JOIN smartfactory.processed.quality_inspections q    
                  ON p.equipment_id = q.equipment_id    
                 AND CAST(DATE(q.inspection_time) AS STRING) = p.production_date  
                GROUP BY p.equipment_id, p.production_date, e.equipment_type, e.line_id
                )
          SELECT  *,
                  ROUND(availability * performance * quality * 100, 2) AS oee_pct,
                  ROUND(availability * 100, 2) AS availability_pct,
                  ROUND(performance * 100, 2) AS performance_pct,
                  ROUND(quality * 100, 2) AS quality_pct,
                  current_timestamp() AS updated_at
          FROM base""")
# Plotly 시각화
oee_pdf = (spark.table("smartfactory.analytics.oee_daily")
               .filter(F.col("production_date") >= F.date_sub(F.current_date(), 30))
               .toPandas())
oee_pdf["production_date"] = pd.to_datetime(oee_pdf["production_date"])




fig = make_subplots(rows=2, cols=2,
    subplot_titles=("OEE 추이", "설비별 평균 OEE", "가용성·성능·품질 분해", "일간 생산량"))




# ── 패널 1: OEE 추이 (상위 5개 설비) ────────────────────
for eq_id in sorted(oee_pdf["equipment_id"].unique())[:5]:
    df_eq = oee_pdf[oee_pdf["equipment_id"] == eq_id]
    fig.add_trace(go.Scatter(x=df_eq["production_date"], y=df_eq["oee_pct"],
        name=eq_id, mode="lines+markers"), row=1, col=1)
fig.add_hline(y=82, line_dash="dash", line_color="red",
              annotation_text="목표 82%", row=1, col=1)




# ── 패널 2: 설비별 평균 OEE 바 차트 ────────────────────
oee_avg = (oee_pdf.groupby("equipment_id", as_index=False)["oee_pct"]
                 .mean().sort_values("equipment_id"))
fig.add_trace(go.Bar(x=oee_avg["equipment_id"], y=oee_avg["oee_pct"].round(1),
    marker_color="steelblue", showlegend=False), row=1, col=2)
fig.add_hline(y=82, line_dash="dash", line_color="red", row=1, col=2)




# ── 패널 3: OEE 3요소 분해 (일별 전체 평균) ─────────────
daily_kpi = (oee_pdf.groupby("production_date", as_index=False)
                   [["availability_pct", "performance_pct", "quality_pct"]].mean())
for kpi_col, name, color in [("availability_pct", "가용성", "royalblue"),
                              ("performance_pct",  "성능",   "seagreen"),
                              ("quality_pct",      "품질",   "darkorange")]:
    fig.add_trace(go.Scatter(x=daily_kpi["production_date"], y=daily_kpi[kpi_col].round(1),
        name=name, mode="lines", line=dict(color=color)), row=2, col=1)




# ── 패널 4: 일간 총 생산량 바 차트 ───────────────────
daily_prod = oee_pdf.groupby("production_date", as_index=False)["total_actual_qty"].sum()
fig.add_trace(go.Bar(x=daily_prod["production_date"], y=daily_prod["total_actual_qty"],
    marker_color="mediumseagreen", showlegend=False, name="생산량"), row=2, col=2)




fig.update_layout(height=800, title_text="스마트팩토리 코리아 - OEE 대시보드")
fig.show()
print("OEE Gold 테이블 생성 및 시각화 완료")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 다음 단계
# MAGIC - **8장**: `01_data_analyst/03_lab03_ai_functions.py` (AI SQL 함수)