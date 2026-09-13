# Minimal RAG Benchmark

## 목적
SciFact에서 BM25와 dense retrieval을 직접 비교했다.

## 범위와 정직한 표현
검색·평가 로직은 NumPy로 직접 구현했고, 임베딩 모델과 LLM은 기존 모델을 사용했다.
따라서 “완전한 RAG from scratch”보다 **“without RAG frameworks”**라고 표현하는 편이 정확하다.

## 파이프라인
query → BM25/dense → Top-k abstracts → LLM → grounded answer

## 결과
BM25와 dense 각각의 Recall@5, MRR@10, 중앙 검색 시간을 기록한다.

## 실패 사례 3개
어떤 질문에서 어느 검색기가 실패했고 왜 그런지 직접 분석한다.

## 설계 판단
왜 SciFact인지, 왜 LangChain·FAISS·벡터 DB를 사용하지 않았는지 설명한다.

## 실행 방법과 한계
