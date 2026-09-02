# Databricks notebook source
# MAGIC %md
# MAGIC #22장 | Vector Search — 정비 매뉴얼을 의미로 검색하다
# MAGIC

# COMMAND ----------

# MAGIC %pip install --quiet markdownify
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import requests
from functools import reduce
from markdownify import markdownify as md_convert
from pyspark.sql import DataFrame
import pandas as pd
from pyspark.sql.functions import md5, concat_ws, col, lit, pandas_udf
from pyspark.sql.types import StringType


# --- 1. 볼륨 생성 ---
spark.sql("CREATE VOLUME IF NOT EXISTS smartfactory.ai.manuals")
print("✅ Volume 생성 완료: /Volumes/smartfactory/ai/manuals/")


# --- 매뉴얼 정의 (U.S. Government Public Domain 문서 3건) ---
# 본 실습에서는 미국 연방정부 기관 OSHA와 NIOSH가 공개한 산업안전 문서를 사용합니다. 
# 문서는 Public Domain 자료이며, 실습에서는 원본 PDF를 재배포하지 않고 각 기관의 공식 URL에서 직접 다운로드합니다.
MANUALS = {
    "osha_3170.pdf": {
        "url": "https://www.osha.gov/sites/default/files/publications/OSHA3170.pdf",
        "doc_id": "OSHA-3170",
        "title": "Safeguarding Equipment and Protecting Employees from Amputations",
        "equipment_type": "MACHINE",    # 기계 안전장치 특화 (guards, barriers, interlocks)
        "doc_type": "safety_guide",     # 안전 가이드라인
        "provider": "U.S. OSHA",
        "license": "Public Domain",
    },


    "niosh_2011_156.pdf": {
        "url": "https://stacks.cdc.gov/view/cdc/5923/cdc_5923_DS1.pdf",
        "doc_id": "NIOSH-2011-156",
        "title": "Using Lockout and Tagout Procedures to Prevent Injury and Death During Machine Maintenance",
        "equipment_type": "GENERAL",    # 일반 산업 설비
        "doc_type": "procedure",        # LOTO 절차서
        "provider": "CDC / NIOSH",
        "license": "Public Domain",
    },


    "niosh_83_125.pdf": {
        "url": "https://stacks.cdc.gov/view/cdc/11169/cdc_11169_DS1.pdf",
        "doc_id": "NIOSH-83-125",
        "title": "Guidelines for Controlling Hazardous Energy During Maintenance and Servicing",
        "equipment_type": "GENERAL",    # 일반 산업 설비
        "doc_type": "safety_guide",     # 에너지 제어 가이드
        "provider": "CDC / NIOSH",
        "license": "Public Domain",
    },
}




# --- 2. PDF 다운로드 ---
volume_base = "/Volumes/smartfactory/ai/manuals"
for filename, meta in MANUALS.items():
    print(f"📄 다운로드 중: {filename} ...", end=" ")
    resp = requests.get(meta["url"], allow_redirects=True)
    resp.raise_for_status()
    with open(f"{volume_base}/{filename}", "wb") as f:
        f.write(resp.content)
    print(f"✅ {len(resp.content) / (1024*1024):.1f} MB")
print(f"💾 전체 {len(MANUALS)}건 다운로드 완료\n")


# --- 3. HTML 테이블 → 마크다운 변환 UDF ---
@pandas_udf(StringType())
def html_to_markdown(texts: pd.Series) -> pd.Series:
    """content 내 HTML 테이블을 마크다운으로 변환 (Arrow 배치 처리)"""
    def convert(text):
        if text is None or '<table' not in text:
            return text
        return md_convert(text, strip=['img'])
    return texts.apply(convert)


# --- 4. 각 PDF별 ai_parse_document() → 페이지별 그룹화 → 메타데이터 부여
all_dfs = []


for filename, meta in MANUALS.items():
    pdf_path = f"{volume_base}/{filename}"
    print(f"파싱 중: {filename} ...", end=" ")


    df_pages = spark.sql(f"""
        WITH raw AS (
            SELECT ai_parse_document('{pdf_path}') AS parsed
        ),
        elements AS (
            SELECT
                CASE elem.type
                    WHEN 'title' THEN CONCAT('# ', elem.content)
                    WHEN 'section_header' THEN CONCAT('## ', elem.content)
                    WHEN 'caption' THEN CONCAT('*', elem.content, '*')
                    ELSE elem.content
                END AS content,
                elem.bbox[0].page_id AS page_id
            FROM raw
            LATERAL VIEW EXPLODE(
                from_json(
                    parsed:document.elements::STRING,
                    'ARRAY<STRUCT<id:BIGINT, type:STRING, content:STRING, confidence:DOUBLE, description:STRING, bbox:ARRAY<STRUCT<coord:ARRAY<BIGINT>, page_id:BIGINT>>>>'
                )
            ) AS elem
            WHERE elem.content IS NOT NULL AND LENGTH(elem.content) > 0
        )
        SELECT
            page_id,
            CONCAT_WS('\\n', COLLECT_LIST(content)) AS page_content
        FROM elements
        GROUP BY page_id
        ORDER BY page_id
    """)


    pg_count = df_pages.count()
    df_doc = (df_pages
        .withColumnRenamed("page_content", "content")
        .withColumns({
            "doc_id": lit(meta["doc_id"]),
            "title": lit(meta["title"]),
            "equipment_type": lit(meta["equipment_type"]),
            "doc_type": lit(meta["doc_type"]),
            "content": html_to_markdown(col("content")),
            "chunk_id": md5(concat_ws("_", lit(meta["doc_id"]), col("page_id").cast("string"))),
            "provider": lit(meta["provider"]),
            "license": lit(meta["license"]),
        })
    )
    all_dfs.append(df_doc)
    print(f"✅ {pg_count}페이지")


# --- 5. 전체 합치기 + Delta 저장 + CDF 활성화
df_final = reduce(DataFrame.unionByName, all_dfs)
df_final = df_final.select("chunk_id", "doc_id", "title", "equipment_type", "doc_type", "page_id", "content", "provider", "license")


total_chunks = df_final.count()
df_final.write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("smartfactory.ai.manual_chunks")


spark.sql(
    "ALTER TABLE smartfactory.ai.manual_chunks "
    "SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
)


print(f"\n✅ smartfactory.ai.manual_chunks 저장 완료 ({total_chunks}청크, CDF 활성화)")
display(df_final.groupBy("equipment_type", "doc_type", "title").count().orderBy("equipment_type"))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 22.2 Vector Search 인덱스 생성
# MAGIC

# COMMAND ----------

# MAGIC %pip install --quiet databricks-ai-search
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,AI Search 엔드포인트 생성
from databricks.ai_search.client import AISearchClient


client = AISearchClient()


# --- 1. AI Search 엔드포인트 생성 ---
ENDPOINT_NAME = "smartfactory-vs"
try:
    client.create_endpoint(name=ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"AI Search 엔드포인트 '{ENDPOINT_NAME}' 생성 중... (약 5~10분 소요)")
except Exception as e:
    if "already exists" in str(e):
        print(f"✅ 엔드포인트 '{ENDPOINT_NAME}' 이미 존재")
    else:
        raise e

# COMMAND ----------

# --- 2. Delta Sync 인덱스 생성 (TRIGGERED 모드) ---
INDEX_NAME = "smartfactory.ai.manual_index"
try:
    index = client.create_delta_sync_index(
        endpoint_name=ENDPOINT_NAME,
        index_name=INDEX_NAME,
        source_table_name="smartfactory.ai.manual_chunks",
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="content",
        embedding_model_endpoint_name="databricks-qwen3-embedding-0-6b",
    )
    print(f"✅ 인덱스 '{INDEX_NAME}' 생성 완료 5~10분 대기 후 다음 코드를 실행합니다.")
except Exception as e:
    if "already exists" in str(e):
        print(f"✅ 인덱스 '{INDEX_NAME}' 이미 존재")
    else:
        raise e

# COMMAND ----------

# --- 3. TRIGGERED 모드: 수동 동기화 ---
import time


index = client.get_index(index_name=INDEX_NAME)


# 신규 생성 직후는 SETTING_UP_TABLES 상태이므로 초기 설정 완료 후 sync 호출
try:
    index.sync()
    print("🔄 인덱스 동기화 시작 (TRIGGERED)")
except Exception as e:
    if "SETTING_UP_TABLES" in str(e) or "not ready to sync" in str(e):
        print("⏳ 인덱스 초기 설정 중... 자동 동기화 완료 대기")
    else:
        raise e


# --- 4. 인덱스 동기화 완료 대기 (폴링) ---
while True:
    idx_status = index.describe().get("status", {}).get("ready", False)
    if idx_status:
        print("✅ 인덱스 동기화 완료 — 검색 가능")
        break
    print("⏳ 동기화 대기 중...")
    time.sleep(30)


# --- 5. 검색 테스트: 필터 없이 vs 필터 적용 비교 ---


print("\n[한국어 검색] '정비 전 안전 잠금 절차' → 영어 LOTO 문서 검색:")
results_kr1 = index.similarity_search(
    query_text="정비 전 안전 잠금 절차",
    columns=["chunk_id", "title", "doc_type", "equipment_type", "content"],
    num_results=3,
)
for c in results_kr1.get("result", {}).get("data_array", []):
    print(f"  [{c[1]}] ({c[2]}) score: {c[-1]:.4f}")
    print(f"    {c[4][:80]}...")


print("\n[한국어 + 필터] '에너지 차단 절차' (doc_type='procedure'):")
results_kr2 = index.similarity_search(
    query_text="에너지 차단 절차",
    columns=["chunk_id", "title", "doc_type", "equipment_type", "content"],
    num_results=3,
    filters={"doc_type": "procedure"},
)
for c in results_kr2.get("result", {}).get("data_array", []):
    print(f"  [{c[1]}] ({c[2]}) score: {c[-1]:.4f}")
    print(f"    {c[4][:80]}...")


print("\n[한국어 + 필터] '기계 안전 방호망 절단 사고 예방' (equipment_type='MACHINE'):")
results_kr3 = index.similarity_search(
    query_text="기계 안전 방호망 절단 사고 예방",
    columns=["chunk_id", "title", "doc_type", "equipment_type", "content"],
    num_results=3,
    filters={"equipment_type": "MACHINE"},
)
for c in results_kr3.get("result", {}).get("data_array", []):
    print(f"  [{c[1]}] ({c[2]}) score: {c[-1]:.4f}")
    print(f"    {c[4][:80]}...")


# COMMAND ----------

# MAGIC %md
# MAGIC # 23장 | RAG 시스템 — 문서와 실시간 데이터를 결합한 AI 답변

# COMMAND ----------

# MAGIC %md
# MAGIC ## 23.1 RAG 아키텍처

# COMMAND ----------

# DBTITLE 1,LangChain + AI Search 설치
# MAGIC %pip install --quiet databricks-langchain
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

from databricks_langchain import ChatDatabricks
from databricks_langchain import DatabricksVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# --- 1. Vector Store + Retriever ---
vectorstore = DatabricksVectorSearch(
    index_name="smartfactory.ai.manual_index",
    columns=["chunk_id", "doc_id", "title", "equipment_type", "doc_type", "page_id", "provider"]
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# --- 2. LLM ---
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct",
                     temperature=0.1, max_tokens=800)


# --- 3. RAG 프롬프트 ---
RAG_PROMPT = ChatPromptTemplate.from_template('''
당신은 스마트팩토리 코리아의 설비 정비 AI 어시스턴트입니다.
아래 참고 문서를 기반으로만 답변하세요.
문서에 없는 내용은 "매뉴얼에 명시되지 않은 내용입니다"라고 답하세요.


[안전 원칙]
- 전기 작업 전 LOTO 절차를 반드시 언급하세요
- 개인보호장비(PPE) 착용 필요 시 명시하세요


[참고 문서]
{context}


[질문]
{question}


[답변] (한국어로, 단계별로):
''')


# --- 4. 문서 포맷팅 ---
def format_docs(docs) -> str:
    return "\n\n".join(
        f"[문서ID: {d.metadata.get('doc_id','N/A')} | "
        f"제목: {d.metadata.get('title','알 수 없음')} | "
        f"유형: {d.metadata.get('doc_type','N/A')} | "
        f"설비: {d.metadata.get('equipment_type','N/A')} | "
        f"페이지: {d.metadata.get('page_id','N/A')} | "
        f"출처: {d.metadata.get('provider','N/A')}]\n{d.page_content}"
        for d in docs
    )


# --- 5. LCEL RAG 체인 (한국어 직접 검색 → 다국어 임베딩 → 한국어 답변) ---
rag_chain = (
    {
        "context": retriever | format_docs,  # 다국어 임베딩으로 한국어 직접 검색 (번역 제거)
        "question": RunnablePassthrough(),
    }
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)


# --- 7. 테스트 ---
test_questions = [
    # NIOSH-2011-156: LOTO 절차서 (doc_type: procedure)
    "정비 전 안전 잠금(Lockout/Tagout) 절차는 어떻게 되나요?",
    # OSHA-3170: 기계 안전장치 (doc_type: safety_guide, equipment_type: MACHINE)
    "기계 안전장치(guard) 설치 요구사항은 무엇인가요?",
    # NIOSH-83-125: 위험 에너지 제어 (doc_type: safety_guide, equipment_type: GENERAL)
    "위험 에너지 제어 절차에서 잔여 에너지 해소 방법은?",
]


for q in test_questions:
    print(f"\n{'='*60}")
    print(f"🔍 원문 질문(KO): {q}")
    print(f"🌐 벡터 검색 → 한국어 답변 생성 중...\n")
    print(rag_chain.invoke(q))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 23.2 하이브리드 RAG — 매뉴얼 + 실시간 센서
# MAGIC  > 23.1 코드에 이어 진행됩니다.

# COMMAND ----------

from langchain_core.runnables import RunnableLambda


# --- 1. 센서 데이터 조회 함수 ---
def get_sensor_context(equipment_id: str, anomaly_only: bool = False) -> str:
    """sensor_clean 테이블에서 센서 데이터 조회
    anomaly_only=True: 가장 최근 anomaly 이벤트 조회 (시나리오 시연용)
    anomaly_only=False: 최신 레코드 조회 (실제 운영용)
    """
    anomaly_filter = "AND is_anomaly = true" if anomaly_only else ""
    sensor_df = spark.sql(f'''
        SELECT equipment_id, equipment_type, temperature_c, vibration_ms2,
               pressure_bar, rpm, is_anomaly, event_time
        FROM smartfactory.processed.sensor_clean
        WHERE equipment_id = '{equipment_id}' {anomaly_filter}
        ORDER BY event_time DESC LIMIT 1
    ''')
    if sensor_df.isEmpty():
        return "현재 센서 데이터 없음"
    row = sensor_df.first()
    anomaly_status = "⚠️ ANOMALY 감지" if row['is_anomaly'] else "✅ 정상"
    return (
        f"현재 설비 상태 ({row['equipment_id']} / {row['equipment_type']}):\n"
        f"  시각: {row['event_time']}\n"
        f"  온도: {row['temperature_c']:.1f}°C | 진동: {row['vibration_ms2']:.2f}mm/s\n"
        f"  압력: {row['pressure_bar']:.1f}bar | RPM: {row['rpm']:.0f}\n"
        f"  상태: {anomaly_status}"
    )


# --- 2. 하이브리드 RAG 프롬프트 ---
HYBRID_PROMPT = ChatPromptTemplate.from_template('''
당신은 스마트팩토리 코리아의 설비 정비 AI 어시스턴트입니다.
아래 [실시간 설비 상태]와 [안전 매뉴얼]을 모두 고려하여 답변하세요.
매뉴얼에 없는 내용은 "매뉴얼에 명시되지 않은 내용입니다"라고 답하세요.


[안전 원칙]
- ANOMALY 감지 시 정비 전 LOTO 절차를 반드시 언급하세요
- 고온·고진동 상태에서는 잔여 에너지 해소의 중요성을 강조하세요
- 개인보호장비(PPE) 착용 필요 시 명시하세요


[실시간 설비 상태]
{sensor_context}


[안전 매뉴얼]
{manual_context}


[질문]
{question}


[답변] (한국어로, 상황 반영하여 단계별로):
''')


# --- 3. 하이브리드 RAG 함수 (LCEL 컴포넌트 재사용) ---
def hybrid_rag_query(question: str, equipment_id: str, anomaly_only: bool = True) -> str:
    """센서 데이터 + 매뉴얼 검색을 결합한 하이브리드 RAG"""
    # 센서 컨텍스트 조회 (anomaly_only=True: anomaly 이벤트 우선 조회)
    sensor_context = get_sensor_context(equipment_id, anomaly_only=anomaly_only)


    # 다국어 임베딩으로 한국어 직접 검색 (번역 불필요)
    docs = retriever.invoke(question)
    manual_context = format_docs(docs)


    # 하이브리드 프롬프트로 답변 생성
    chain = HYBRID_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "sensor_context": sensor_context,
        "manual_context": manual_context,
        "question": question,
    })


# --- 4. 테스트: CNC 설비 anomaly 발생 시 안전 절차 문의 ---
print("=" * 60)
print("🏭 하이브리드 RAG 테스트: 센서 이상 + 안전 절차 결합")
print("=" * 60)


# 센서 상태 확인 (anomaly_only=True: 가장 최근 anomaly 이벤트 조회)
print(f"\n📊 {get_sensor_context('EQ005', anomaly_only=True)}\n")


# 하이브리드 RAG 질의
result = hybrid_rag_query(
    question="이 설비에 이상이 감지되었습니다. 정비 전 안전 절차와 잔여 에너지 해소 방법을 알려주세요.",
    equipment_id="EQ005"
)
print(result)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 23.3 유지보수 RAG 시스템

# COMMAND ----------

test_scenarios = [
    {
        "equipment_id": "EQ005",
        "question": "이 설비에 이상이 감지되었습니다. 정비 전 LOTO 절차와 주의사항을 알려주세요.",
    },
    {
        "equipment_id": "EQ007",
        "question": "이 설비의 안전장치(guard) 상태를 점검해야 합니다. 요구사항은?",
    },
]


for scenario in test_scenarios:
    print(f"\n{'='*60}")
    print(f"🏭 설비: {scenario['equipment_id']}")
    print(f"🔍 질문: {scenario['question']}")
    print(f"{'-'*60}")
    result = hybrid_rag_query(
        question=scenario["question"],
        equipment_id=scenario["equipment_id"],
    )
    print(result)


print(f"\n{'='*60}")
print("✅ Lab 4-2 하이브리드 RAG 실습 완료")
