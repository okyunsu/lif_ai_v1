# Taxo Service

EU Taxonomy alignment analysis service for ESG reports.

## 개요

`taxo-service`는 ESG 보고서에서 추출된 텍스트를 분석하여, EU Taxonomy(유럽연합 택소노미)의 6대 환경 목표 중 어떤 항목에 해당하는지 자동 분류하고 정렬도(Alignment Score)를 계산하는 마이크로서비스입니다.

## 주요 기능

- ESG 텍스트 분석을 통한 EU Taxonomy 환경 목표 분류
- 정렬도(Alignment Score) 계산
- DNSH(Do No Significant Harm) 기준 위배 여부 판단

## 기술 스택

- FastAPI
- PyTorch
- Transformers (BERT)
- Docker

## 설치 및 실행

### 로컬 개발 환경

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --reload --port 9000
```

### Docker 환경

```bash
# 도커 이미지 빌드
docker build -t taxo-service .

# 컨테이너 실행
docker run -p 9000:9000 taxo-service
```

## API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:

- Swagger UI: `http://localhost:9000/docs`
- ReDoc: `http://localhost:9000/redoc`

## 주요 API 엔드포인트

### 단일 텍스트 분석

```
POST /api/taxonomy/analyze-alignment
```

요청 예시:
```json
{
  "text": "당사는 2030년까지 전체 에너지의 80%를 재생에너지로 전환할 계획입니다.",
  "report_id": "report-123"
}
```

응답 예시:
```json
{
  "taxonomy_goals": ["기후 변화 완화"],
  "confidence": 0.91,
  "alignment_score": 78.2,
  "dnsh_violation": false
}
```

### 다중 텍스트 분석

```
POST /api/taxonomy/analyze-alignment/batch
```

요청 예시:
```json
{
  "texts": [
    "당사는 2030년까지 전체 에너지의 80%를 재생에너지로 전환할 계획입니다.",
    "수자원 보호를 위해 폐수 처리 시스템을 개선하였습니다."
  ],
  "report_id": "report-123"
}
```

응답 예시:
```json
{
  "results": [
    {
      "taxonomy_goals": ["기후 변화 완화"],
      "confidence": 0.91,
      "alignment_score": 78.2,
      "dnsh_violation": false
    },
    {
      "taxonomy_goals": ["수자원 및 해양 자원의 지속가능한 이용 및 보호"],
      "confidence": 0.85,
      "alignment_score": 72.5,
      "dnsh_violation": false
    }
  ]
}
```

## 모델 정보

현재 구현은 BERT 기반의 다국어 모델을 사용합니다. 실제 배포 시에는 EU Taxonomy 분류를 위해 특별히 파인튜닝된 모델을 사용해야 합니다. 