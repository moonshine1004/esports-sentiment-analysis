# e스포츠 오심에 관한 관한 시청자 인식 감정분석: 유튜브 댓글을 중심으로

e스포츠 오심 경기에 대한 시청자 반응을 분석하기 위한 연구용 Python 프로젝트입니다.

YouTube 댓글과 대댓글을 수집한 뒤, 사전에 정의한 분류체계에 따라 GPT 기반 분류와 인간 코딩을 수행합니다. 인간 코딩 결과를 기준으로 GPT 분류 성능을 평가하고, 전체 댓글에서 정서, 평가 대상, 태도, 조롱·냉소 표현의 분포를 분석합니다.

> 이 저장소는 현재 연구 진행 중인 코드를 포함합니다. 인간 코딩과 분석 설계에 따라 일부 코드 및 출력 구조가 변경될 수 있습니다.

---

## 1. 연구 개요

본 프로젝트는 e스포츠 경기에서 발생한 시스템 오류와 이에 대한 시청자 반응이 어떤 방식으로 나타나는지를 분석합니다.

현재 분석 자료는 다음 세 유형의 영상에서 수집됩니다.

| 사례 ID | 사례 |
|---|---|
| `01` | 룬 설정 오류 관련 영상 |
| `02` | 강타 재사용 대기시간 오류 관련 영상 |
| `03` | 두 오류가 포함된 경기 하이라이트 영상 |

각 댓글과 대댓글은 독립적인 분석 단위로 처리합니다. 대댓글의 경우 부모 댓글은 의미 해석을 위한 문맥으로만 사용하고, 실제 분류는 대댓글 자체를 대상으로 수행합니다.

---

## 2. 분석 항목

모든 댓글은 다음 네 가지 항목으로 분류합니다.

### 2.1 Sentiment

| 레이블 | 의미 |
|---|---|
| `positive` | 긍정적 평가 댓글 |
| `neutral` | 사실 전달 중심의 감정표현이 약한 댓글 |
| `negative` | 부정적 평가 댓글 |

### 2.2 Target

| 레이블 | 의미 |
|---|---|
| `referee` | 경기 중 오류 발생 여부를 인지하고 조치하는 판정 주체 개인 |
| `league` | 공식 경기 운영, 제도적 책임을 담당하는 운영 주체 |
| `player` | 경기력 평가 대상으로 언급되는 경기 참여자, 팀 |
| `game_system` | 인간 행위자에게 환원되지 않는 게임 시스템의 기술적·절차적 구조 |
| `none` | 평가가 위 대상에 명확히 귀속되지 않는 경우 |

### 2.3 Stance

| 레이블 | 의미 |
|---|---|
| `blame` | 특정 대상에 대한 비판, 문제 제기, 책임 귀속 |
| `support` | 특정 대상에 대한 옹호, 방어, 책임 부정 |
| `neutral_fact` | 사실 전달 및 설명 |
| `other` | 위 세 범주로 분류하기 어려운 경우 |

### 2.4 Sarcasm and mockery

| 레이블 | 의미 |
|---|---|
| `true` | 비꼼, 냉소, 희화화, 과장, 밈적 조롱을 통해 평가를 드러내는 경우 |
| `false` | 그러한 표현이 없는 경우 |


---

## 3. 인간 코딩 설계

인간 코딩 표본은 전체 댓글에서 사례와 댓글 유형을 기준으로 층화 추출합니다.

---

## 4. GPT 분류 설계

GPT 출력은 Pydantic 기반의 사전 정의된 구조화 출력 스키마를 사용합니다.
출력 항목은 다음과 같습니다.
```text
sentiment
target
stance
is_sarcasm_mockery
```
각 분석 단위는 현재 설정에서 1회 분류합니다.

GPT 분류 모델, 프롬프트 버전 및 실행 조건은 `.env`에서 설정합니다. 연구 재현성을 위해 최종 분석에서는 동일한 모델과 프롬프트 버전을 유지해야 합니다.

---

## 5. 프로젝트 구조

```text
esports-sentiment/
├─ src/
│  ├─ config.py
│  ├─ labels.py
│  ├─ models.py
│  ├─ text_utils.py
│  ├─ 01_collect_youtube_comments.py
│  ├─ 02_preprocess_comments.py
│  ├─ 03_sample_human_coding.py
│  ├─ 04_create_coder_workbooks.py
│  ├─ 05_compute_coder_agreement.py
│  ├─ 06_create_consensus_workbook.py
│  ├─ 07_finalize_gold_labels.py
│  ├─ 08_run_gpt_repeated_classification.py
│  ├─ 09_finalize_gpt_predictions.py
│  ├─ 10_evaluate_gpt.py
│  ├─ 11_analyze_gpt_results.py
│  ├─ 12_visualize_gpt_results.py
│  └─ 13_visualize_evaluation.py
│
├─ prompts/
│  ├─ classification_prompt_v1.txt
│  └─ classification_prompt_v2.txt
│
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ human_coding/
│  └─ results/
│
├─ .env.example
├─ .gitignore
├─ requirements.txt
└─ README.md
```

---

## 6. 실행 환경

### 6.1 권장 환경

- Python 3.10 이상
- Windows PowerShell 또는 명령 프롬프트
- Anaconda 또는 Miniconda
- YouTube Data API 키
- OpenAI API 키

### 6.2 Conda 환경 생성

```powershell
conda create -n esports-nlp python=3.11
conda activate esports-nlp
```

### 6.3 패키지 설치

```powershell
python -m pip install -r requirements.txt
```

주요 패키지는 다음과 같습니다.

```text
google-api-python-client
matplotlib
numpy
openai
openpyxl
pandas
pydantic
python-dotenv
scikit-learn
scipy
```

---

## 7. 전체 분석 파이프라인

전체 파이프라인은 다음과 같이 구성됩니다.

```text
YouTube 댓글 수집
        ↓
텍스트 전처리
        ↓
인간 코딩 표본 추출
        ↓
코더별 Excel 생성
        ↓
인간 코딩
        ↓
코더 간 신뢰도 계산
        ↓
불일치 합의 코딩
        ↓
최종 인간 기준 레이블 생성
        ↓
GPT 분류
        ↓
GPT 결과 확정
        ↓
인간 기준 레이블과 GPT 성능 비교
        ↓
전체 댓글 분석 및 시각화
```

인간 코딩과 GPT 분류는 독립적으로 진행할 수 있습니다.

---

## 8. 단계별 사용 방법

### 8.1 YouTube 댓글 수집

```powershell
python src\01_collect_youtube_comments.py
```

주요 출력:

```text
data/raw/youtube_comments_raw.csv
data/raw/youtube_comments_raw.xlsx
data/raw/collection_errors.csv
data/raw/collection_metadata.json
```

원댓글과 대댓글을 모두 수집합니다.

---

### 8.2 댓글 전처리

```powershell
python src\02_preprocess_comments.py
```

주요 출력:

```text
data/interim/comments_preprocessed.csv
data/interim/comments_preprocessed.xlsx
```

전처리 과정에서는 URL과 불필요한 공백을 정리합니다. 웃음, 울음 표현, 이모지 및 문장부호는 분석 의미를 보존하기 위해 유지합니다.

대댓글에는 부모 댓글 문맥이 `parent_text`로 연결됩니다.

---

### 8.3 인간 코딩 표본 추출

```powershell
python src\03_sample_human_coding.py
```

주요 출력:

```text
data/human_coding/human_sample_master.csv
data/human_coding/human_sample_master.xlsx
```

전체 자료에서 500건을 추출하고, 그중 코더 2가 코딩할 공통 표본 150건을 지정합니다.

---

### 8.4 코더별 Excel 파일 생성

```powershell
python src\04_create_coder_workbooks.py
```

주요 출력:

```text
data/human_coding/coder1_coding.xlsx
data/human_coding/coder2_coding.xlsx
```

- `coder1_coding.xlsx`: 500건
- `coder2_coding.xlsx`: 공통 표본 150건

코더는 `analysis_text`를 실제 분석 대상으로 사용합니다. `parent_text`는 대댓글의 문맥을 확인할 때만 사용합니다.

---

### 8.5 인간 코딩

각 코더는 다음 항목을 독립적으로 입력합니다.

```text
sentiment
target
stance
is_sarcasm_mockery
coder_note
```

드롭다운에 정의된 값만 사용해야 합니다.

---

### 8.6 코더 간 신뢰도 계산

두 코더의 코딩이 완료된 뒤 실행합니다.

```powershell
python src\05_compute_coder_agreement.py
```

주요 출력:

```text
data/human_coding/coder_agreement.csv
data/human_coding/coding_disagreements.xlsx
```

공통 표본 150건을 대상으로 다음 값을 산출합니다.

```text
원시 일치율
Cohen's kappa
```

결과는 전체, 사례별, 댓글 유형별로 저장됩니다.

---

### 8.7 합의 코딩 파일 생성

```powershell
python src\06_create_consensus_workbook.py
```

주요 출력:

```text
data/human_coding/consensus_coding.xlsx
```

두 코더가 불일치한 항목을 검토하고 `consensus_*` 열에 최종 합의 레이블을 입력합니다.

---

### 8.8 최종 인간 기준 레이블 생성

합의 코딩이 완료된 뒤 실행합니다.

```powershell
python src\07_finalize_gold_labels.py
```

주요 출력:

```text
data/human_coding/gold_labels.csv
data/human_coding/gold_labels.xlsx
```

이 파일은 GPT 성능평가의 인간 기준 자료로 사용됩니다.

---

### 8.9 GPT 분류 실행

```powershell
python src\08_run_gpt_repeated_classification.py
```

주요 출력:

```text
data/results/gpt_predictions_runs.csv
data/results/gpt_predictions_runs.xlsx
```

실제 전체 분류 전에는 코드의 `TEST_LIMIT`을 작은 값으로 설정해 시험 실행하는 것이 권장됩니다.

예:

```python
TEST_LIMIT = 2
```

시험 결과를 확인한 뒤 전체 실행 시 다음과 같이 변경합니다.

```python
TEST_LIMIT = 0
```

중단된 실행은 성공적으로 저장된 분석 단위를 건너뛰고 이어서 실행할 수 있습니다. 단, 모델 또는 프롬프트가 변경된 경우 기존 결과와 새 결과를 혼합하지 않아야 합니다.

---

### 8.10 GPT 최종 예측 파일 생성

```powershell
python src\09_finalize_gpt_predictions.py
```

주요 출력:

```text
data/results/gpt_predictions_final.csv
data/results/gpt_predictions_final.xlsx
```

---

### 8.11 GPT 분류 성능평가

인간 기준 레이블이 완성된 뒤 실행합니다.

```powershell
python src\10_evaluate_gpt.py
```

주요 출력:

```text
data/results/gpt_evaluation_summary.csv
data/results/gpt_evaluation_per_class.csv
data/results/gpt_evaluation_confusion.csv
data/results/gpt_evaluation_errors.csv
data/results/gpt_evaluation.xlsx
```

산출 지표:

```text
Accuracy
Macro-Precision
Macro-Recall
Macro-F1
Weighted-F1
클래스별 Precision
클래스별 Recall
클래스별 F1-score
혼동행렬
오분류 사례
```

클래스 불균형을 고려해 Macro-F1을 주요 평가 지표로 사용합니다.

---

### 8.12 전체 GPT 분류 결과 분석

```powershell
python src\11_analyze_gpt_results.py
```

주요 출력:

```text
data/results/gpt_label_distribution.csv
data/results/gpt_case_crosstab.csv
data/results/gpt_case_chi_square.csv
data/results/gpt_case_residuals.csv
data/results/gpt_descriptive_analysis.xlsx
```

이 단계에서는 다음 내용을 분석합니다.

- 전체 레이블 빈도와 비율
- 사례별 레이블 분포
- 원댓글·대댓글별 분포
- 사례와 레이블 간 카이제곱 검정
- Cramér’s V
- 조정 표준화 잔차
- Benjamini-Hochberg 다중검정 보정

이 분석은 탐색적 분석을 포함합니다. 특히 사례 ID `03`은 두 오류가 함께 포함된 하이라이트 영상이므로, 사례 간 추론통계 결과는 연구설계와 자료 구조를 고려해 제한적으로 해석해야 합니다.

---

### 8.13 전체 GPT 결과 시각화

```powershell
python src\12_visualize_gpt_results.py
```

주요 출력:

```text
data/results/figures/figure_01_sentiment.png
data/results/figures/figure_02_target.png
data/results/figures/figure_03_stance.png
data/results/figures/figure_04_is_sarcasm_mockery.png
```

각 그래프에는 통합데이터와 사례별 결과가 함께 표시됩니다.

막대 높이는 실제 댓글 건수이며, 막대 끝에는 다음 형식으로 건수와 집단 내부 비율이 표시됩니다.

```text
1,312(70%)
```

---

### 8.14 성능평가 결과 시각화

인간 코딩과 GPT 성능평가가 완료된 뒤 실행합니다.

```powershell
python src\13_visualize_evaluation.py
```

주요 출력 폴더:

```text
data/results/figures/evaluation/
```

생성되는 그림에는 다음 내용이 포함됩니다.

- 과업별 전체 성능
- 사례별 Macro-F1
- 댓글 유형별 Macro-F1
- 클래스별 Precision, Recall, F1
- 과업별 혼동행렬

---

## 9. 주요 결과 파일

| 파일 | 설명 |
|---|---|
| `comments_preprocessed.csv` | 전처리된 전체 분석 자료 |
| `human_sample_master.csv` | 인간 코딩 표본 500건 |
| `coder_agreement.csv` | 공통 150건의 코더 간 일치도 |
| `gold_labels.csv` | 최종 인간 기준 레이블 |
| `gpt_predictions_final.csv` | 전체 댓글의 최종 GPT 예측 |
| `gpt_evaluation_summary.csv` | GPT 전체 성능평가 |
| `gpt_evaluation_per_class.csv` | 클래스별 성능평가 |
| `gpt_evaluation_errors.csv` | GPT 오분류 사례 |
| `gpt_descriptive_analysis.xlsx` | 전체 GPT 결과 분석 |
| `figures/` | 논문 작성용 시각화 결과 |

---

## 10. 데이터 공개 범위

이 저장소에는 플랫폼 이용조건과 연구윤리, 개인정보 보호 및 재배포 가능성을 검토한 뒤 다음 자료를 포함하지 않습니다.

```text
YouTube 원댓글 및 대댓글
댓글 작성자 정보
YouTube API 키
OpenAI API 키
인간 코딩 파일
합의 코딩 파일
GPT 원시 응답
분석 결과 파일
```

---

## 11. 재현성

연구 결과의 재현을 위해 다음 정보를 함께 기록합니다.

- Python 버전
- 패키지 버전
- GPT 모델 ID
- 프롬프트 버전
- 프롬프트 해시
- 무작위 추출 시드
- 인간 코딩 표본 수
- 코더 간 신뢰도
- 데이터 수집 시점
