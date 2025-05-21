
import uuid
import json
import csv
import os
from keybert import KeyBERT
from konlpy.tag import Okt
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, util
from app.domain.model.esg_issue_dto import ESGIssue

OUTPUT_DIR = "/app/app/domain/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRI_KEYWORD_MAP = {
    "기후변화": "GRI 302: Energy",
    "탄소중립": "GRI 305: Emissions",
    "탄소배출": "GRI 305: Emissions",
    "온실가스": "GRI 305: Emissions",
    "수자원": "GRI 303: Water",
    "다양성": "GRI 405: Diversity",
    "인권": "GRI 412: Human Rights",
    "윤리": "GRI 205: Anti-corruption",
    "정보보호": "GRI 418: Customer Privacy",
    "보안": "GRI 418: Customer Privacy"
}

GRI_TOPICS = {
    "GRI 302: Energy": "에너지 소비, 효율 개선, 재생 에너지 사용 등",
    "GRI 305: Emissions": "온실가스 배출, 탄소중립, 탄소배출권 등",
    "GRI 303: Water": "수자원 절약, 용수 관리, 오염 방지 등",
    "GRI 306: Waste": "폐기물 감축, 자원순환, 재활용 등",
    "GRI 412: Human Rights": "노동 인권, 강제노동, 차별금지, 결사의 자유",
    "GRI 405: Diversity": "다양성과 포용, 성별 다양성, 평등 기회",
    "GRI 205: Anti-corruption": "윤리경영, 부패 방지, 내부 고발 시스템",
    "GRI 418: Customer Privacy": "개인정보보호, 보안, 정보 유출 방지",
    "GRI 403: Occupational Health": "산업안전, 건강 보호, 사고 예방",
    "GRI 201: Economic Performance": "경제 성과, 매출, 투자자 관계"
}

sbert_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

def extract_keywords_from_text(text: str, model) -> list:
    okt = Okt()
    nouns = ' '.join(okt.nouns(text))
    return [kw for kw, _ in model.extract_keywords(nouns, top_n=3)]

def map_keywords_to_gri(keywords: list[str]) -> str | None:
    for kw in keywords:
        for pattern, gri_code in GRI_KEYWORD_MAP.items():
            if pattern in kw:
                return gri_code
    return None

def match_to_gri_by_similarity(text: str) -> str | None:
    target_embedding = sbert_model.encode(text.strip(), convert_to_tensor=True)
    max_sim = 0.0
    matched_topic = None
    for gri_code, gri_desc in GRI_TOPICS.items():
        gri_embedding = sbert_model.encode(gri_desc, convert_to_tensor=True)
        sim = float(util.pytorch_cos_sim(target_embedding, gri_embedding))
        if sim > max_sim:
            max_sim = sim
            matched_topic = gri_code
    return matched_topic if max_sim > 0.4 else None

def map_keywords_to_gri_or_semantic(keywords: list[str], full_text: str) -> str | None:
    mapped = map_keywords_to_gri(keywords)
    if mapped:
        return mapped
    return match_to_gri_by_similarity(full_text)

def save_issues_to_json(issues: list[ESGIssue], filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([issue.dict() for issue in issues], f, ensure_ascii=False, indent=2)
    return path

def save_issues_to_csv(issues: list[ESGIssue], filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "keywords", "mapped_gri", "score", "source_file"])
        for issue in issues:
            writer.writerow([
                issue.id, issue.text, ', '.join(issue.keywords),
                issue.mapped_gri or "", issue.score, issue.source_file
            ])
    return path

def extract_esg_issues_from_pdf(file_path: str) -> list[ESGIssue]:
    text = extract_text(file_path)
    paragraphs = list(set(p.strip() for p in text.split('\n') if len(p.strip()) > 50))
    esg_keywords = ['기후', '탄소', '에너지', '수자원', '인권', '공급망', '정보보호', '보안', '윤리', '재생', '환경', '다양성']
    filtered_paragraphs = [p for p in paragraphs if any(k in p for k in esg_keywords)]

    kw_model = KeyBERT(model='distiluse-base-multilingual-cased-v2')
    results = []

    for para in filtered_paragraphs:
        keywords = extract_keywords_from_text(para, kw_model)
        mapped = map_keywords_to_gri_or_semantic(keywords, para)
        issue = ESGIssue(
            id=str(uuid.uuid4()),
            text=para,
            keywords=keywords,
            mapped_gri=mapped,
            score=0.85,
            source_file=file_path.split("/")[-1]
        )
        results.append(issue)
    return results
