# Shopping Search with VLM and Attribute Information

상품 이미지와 속성 텍스트를 OpenCLIP으로 임베딩하고, FAISS 검색과 대화 요약·후속 질문을 결합해 사용자가 원하는 상품을 더 적은 검색 반복으로 찾도록 돕는 연구 프로토타입입니다.

## 구현 포인트

- OpenCLIP 이미지·속성 임베딩의 late fusion과 L2 정규화를 적용한 카탈로그 인덱싱
- FAISS cosine 검색과 k-means 군집 정보를 결합한 후보 탐색
- 대화 요약, 검색, 중복 질문 검사를 분리한 Streamlit 기반 반복 검색 흐름
- 상대경로 기반 인덱스 아티팩트와 환경변수 기반 API 설정으로 실행 환경 이식성 확보

**Tech stack:** Python, OpenCLIP, PyTorch, FAISS, scikit-learn, Streamlit, OpenAI Responses API

## 시스템 구성

### 준비 단계

1. 상품 이미지와 같은 이름의 속성 `.txt` 파일을 OpenCLIP ViT-H/14로 임베딩합니다.
2. 이미지와 텍스트 임베딩을 late fusion(평균 후 L2 정규화)합니다.
3. FAISS cosine-similarity 인덱스를 생성합니다.
4. k-means로 13개 상품 군집을 만들고, 대표 속성 문장을 군집 설명으로 저장합니다.

### 탐색 단계

1. 누적 대화를 한 문장의 검색 의도로 요약합니다.
2. 요약을 OpenCLIP 텍스트 임베딩으로 변환하고 FAISS에서 후보를 검색합니다.
3. 상위 후보의 주요 군집과 기존 대화를 바탕으로 중복되지 않는 후속 질문을 생성합니다.
4. 사용자의 답변을 누적해 다시 검색합니다.

## 논문 결과

- 실제 온라인 쇼핑 상품 이미지·속성 쌍 53,393개
- 13개 카테고리, 4개 크롤링 출처
- 참가자 10명 × 상품 5개 = 총 50개 검색 과제
- 이미지만 임베딩: 원하는 상품 발견까지 평균 4.96회
- 이미지+속성 임베딩: 평균 2.82회
- 평균 검색 반복 약 43% 감소

수치는 논문에 보고된 사용자 실험 결과입니다. 원 데이터·인덱스·실험 로그가 포함되어 있지 않아 현재 저장소만으로 자동 재현되지는 않습니다.

## 저장소 구조

```text
.
├── app.py                    # Streamlit 대화형 검색 앱
├── paper/                    # 발표 논문 PDF
├── src/shopping_search/      # 인덱싱·검색·LLM 모듈
├── scripts/build_index.py    # 상품 인덱스 생성
├── tests/                    # 임베딩 결합 단위 테스트
├── .env.example              # 환경변수 이름 예시
├── data/README.md            # 데이터 형식
└── artifacts/                # 생성 인덱스(버전 관리 제외)
```

## 보안 주의

API 키는 코드나 설정 파일에 저장하지 않고 운영 환경의 `OPENAI_API_KEY`로만 주입합니다. `.env`는 Git에서 제외되며 `.env.example`에는 변수 이름과 예시값만 포함됩니다. 자세한 공개·취약점 제보 기준은 [SECURITY.md](SECURITY.md)를 참고하세요.

LLM 호출은 `OpenAI()`가 환경변수의 키를 읽고 Responses API의 `client.responses.create(...)`를 사용합니다. 외부 상품 설명과 사용자 대화는 JSON 데이터로 구분하고, 그 안의 지시를 따르지 않도록 시스템 지침을 적용했습니다.

## 설치

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -e .
```

GPU를 사용할 경우 CUDA 버전에 맞는 PyTorch를 먼저 설치하세요. Windows에서 `faiss-cpu` 설치가 실패하면 WSL 또는 conda 환경이 더 간단할 수 있습니다.

## 데이터 준비

각 이미지 옆에 같은 stem을 가진 UTF-8 텍스트 파일을 둡니다.

```text
data/catalog/
├── product_0001.jpg
├── product_0001.txt
├── product_0002.jpg
└── product_0002.txt
```

텍스트에는 논문과 같이 상품명, 가격, 제조사/브랜드, 원산지, 색상, 카테고리 등 사용 가능한 속성을 기록합니다. 자세한 내용은 [data/README.md](data/README.md)를 참고하세요.

## 인덱스 생성

논문의 이미지+속성 설정:

```bash
python scripts/build_index.py \
  --image-root data/catalog \
  --output-dir artifacts/index \
  --fusion image-text \
  --clusters 13
```

이미지만 사용하는 기준 실험은 `--fusion image`로 생성합니다.

## 앱 실행

먼저 환경변수를 설정합니다. `.env.example`에는 형식만 있으며 앱이 `.env`를 자동 로드하지는 않습니다.

PowerShell 예시:

```powershell
$env:OPENAI_API_KEY="새로 발급한 키"
$env:OPENAI_MODEL="사용할 수 있는 모델 ID"
$env:SHOPPING_IMAGE_ROOT="data/catalog"
$env:SHOPPING_INDEX_DIR="artifacts/index"
streamlit run app.py
```

`OPENAI_MODEL`은 계정/프로젝트에서 사용할 수 있는 모델 ID를 명시적으로 지정하도록 했습니다. 코드에 특정 모델을 고정하지 않아 모델 제공 상태가 바뀌어도 환경 설정만 변경하면 됩니다.

## 테스트

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## 데이터·법적 주의

- 쇼핑몰 크롤링 데이터는 사이트 이용약관, 저작권, 로봇 정책을 확인해야 합니다.
- 상품 이미지와 설명을 이 저장소에 재배포하기 전 원 출처의 허용 범위를 확인하세요.
- 사용자 실험을 다시 수행할 경우 대화 로그와 참여자 동의 처리 계획이 필요합니다.

## 논문

[VLM과 속성 정보를 활용한 쇼핑몰 검색 시스템](paper/VLM과%20속성%20정보를%20활용한%20쇼핑몰%20검색%20시스템%20논문집버전.pdf)

## 라이선스

아직 라이선스를 선택하지 않았습니다. 라이선스 파일이 추가되기 전까지 코드와 논문의 저작권은 저자에게 있으며, 재사용 허가는 별도로 받아야 합니다.
