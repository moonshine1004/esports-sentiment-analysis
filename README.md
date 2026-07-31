# e스포츠 오심에 관한 시청자 인식 감정분석: 유튜브 댓글을 중심으로

e스포츠 오심 경기에 대한 시청자 반응을 분석하기 위한 연구용 Python 프로젝트입니다.

> 이 저장소는 현재 연구 진행 중인 코드를 포함합니다. 연구 진행 상황에 따라 일부 코드 및 출력 구조가 변경될 수 있습니다.

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
| `league` | 공식 경기 운영, 규정, 제도적 책임 및 공식 대응을 담당하는 운영 주체 |
| `player` | 경기 참여 선수 또는 팀 |
| `game_system` | 버그, 서버, 클라이언트, 룬, 스킬, 재사용 대기시간 등 게임 시스템의 기술적·절차적 구조 |
| `other_commenter` | 다른 댓글 작성자, 대댓글 상대, 댓글창 이용자, 팬덤 또는 커뮤니티 구성원 |
| `other` | 방송진, 해설자, 코치 등 기타 대상이거나 중심 대상을 위 범주로 특정하기 어려운 경우 |

### 2.3 Stance

태도는 대상과 독립적으로 분류하지 않습니다. 먼저 댓글에서 가장 중심적으로 평가되는 대상을 선택한 뒤, 해당 대상에 대한 태도를 분류합니다.

| 레이블 | 의미 |
|---|---|
| `blame` | 선택한 대상에 대한 비판, 문제 제기, 책임 귀속 |
| `support` | 선택한 대상에 대한 옹호, 방어, 칭찬 또는 책임 부정 |
| `neutral_fact` | 선택한 대상에 대한 평가 없이 사실, 질문 또는 설명을 제시하는 경우 |
| `other` | 선택한 대상에 대한 태도를 위 세 범주로 분류하기 어려운 경우 |

### 2.4 Sarcasm and mockery

| 레이블 | 의미 |
|---|---|
| `true` | 비꼼, 냉소, 희화화, 과장, 밈적 조롱을 통해 평가를 드러내는 경우 |
| `false` | 그러한 표현이 없는 경우 |

---

## 3. GPT 분류 설계

GPT 출력은 Pydantic 기반의 사전 정의된 구조화 출력 스키마를 사용합니다.

출력 항목은 다음과 같습니다.

```text
sentiment
target_attitude.target
target_attitude.stance
is_sarcasm_mockery
reason
```

`reason`에는 대상과 태도를 중심으로 분류 근거를 한국어 한 문장으로 출력합니다. 최종 CSV에서는 다음 열로 평탄화하여 저장합니다.

```text
gpt_sentiment
gpt_target
gpt_stance
gpt_is_sarcasm_mockery
gpt_reason
```

GPT 분류 모델, 서비스 처리 방식, 프롬프트 버전 및 실행 조건은 `.env`에서 설정합니다. 연구 재현성을 위해 최종 분석에서는 동일한 모델 스냅샷과 프롬프트 버전을 유지해야 합니다.

---

## 4. 프로젝트 구조

```text
esports-sentiment/
├─ src/
│  ├─ config.py
│  ├─ labels.py
│  ├─ models.py
│  ├─ text_utils.py
│  ├─ test_common_settings.py
│  ├─ 01_collect_youtube_comments.py
│  ├─ 02_preprocess_comments.py
│  ├─ 03_sample_human_coding.py
│  ├─ 04_create_coder_workbooks.py
│  ├─ 05_finalize_gold_labels.py
│  ├─ 06_run_gpt_classification.py
│  ├─ 07_finalize_gpt_predictions.py
│  ├─ 08_evaluate_gpt.py
│  ├─ 09_visualize_gpt_results.py
│  ├─ 10_visualize_gpt_evaluation.py
│  ├─ 11_visualize_confusion_matrices.py
│
├─ prompts/
│  ├─ classification_prompt_v4.txt
│  └─ legacy/
│     ├─ classification_prompt_v1.txt
│     ├─ classification_prompt_v2.txt
│     └─ classification_prompt_v3.txt
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

## 5. 실행 환경

### 5.1 권장 환경

- Python 3.10 이상
- Windows PowerShell 또는 명령 프롬프트
- Anaconda 또는 Miniconda
- YouTube Data API 키
- OpenAI API 키

### 5.2 Conda 환경 생성

```powershell
conda create -n esports-nlp python=3.11
conda activate esports-nlp
```

### 5.3 패키지 설치

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
```

### 5.4 환경 변수 설정

`.env.example`을 복사해 `.env` 파일을 만들고 필요한 값을 입력합니다.

```powershell
Copy-Item .env.example .env
```

주요 설정 항목은 다음과 같습니다.

```text
YOUTUBE_API_KEY
OPENAI_API_KEY
HUMAN_SAMPLE_MAX=300
OPENAI_MODEL
OPENAI_SERVICE_TIER
REPEAT_COUNT=1
PROMPT_VERSION=v3.0
```

---

## 6. 전체 분석 파이프라인

전체 파이프라인은 다음과 같이 구성됩니다.

```text
YouTube 댓글 수집
        ↓
텍스트 전처리
        ↓
인간 코딩 표본 추출
        ↓
인간 코딩 기준 레이블 생성
        ↓
GPT 분류
        ↓
GPT 결과 확정
        ↓
인간 기준 레이블과 GPT 성능 비교
        ↓
전체 GPT 분류 결과 시각화
        ↓
성능평가 지표 시각화
```

---

## 7. 단계별 사용 방법

### 7.1 YouTube 댓글 수집

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

### 7.2 댓글 전처리

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

### 7.3 인간 코딩 표본 추출

```powershell
python src\03_sample_human_coding.py
```

주요 출력:

```text
data/human_coding/human_sample_master.csv
data/human_coding/human_sample_master.xlsx
```

전체 자료에서 사례와 댓글 유형을 기준으로 300건을 층화 추출합니다.

---

### 7.4 인간 코딩 Excel 파일 생성

```powershell
python src\04_create_coder_workbooks.py
```

주요 출력:

```text
data/human_coding/coder1_coding.xlsx
```

---

### 7.5 인간 코딩

코더는 다음 항목을 입력합니다.

```text
sentiment
target
stance
is_sarcasm_mockery
coder_note
```

드롭다운에 정의된 값만 사용해야 합니다.

---

### 7.6 인간 코딩 기준 레이블 생성

인간 코딩이 완료된 뒤 실행합니다.

```powershell
python src\05_finalize_gold_labels.py
```

주요 출력:

```text
data/human_coding/gold_labels.csv
data/human_coding/gold_labels.xlsx
```


---

### 7.7 GPT 분류 실행

```powershell
python src\06_run_gpt_classification.py
```

주요 출력:

```text
data/results/gpt_predictions_runs.csv
data/results/gpt_predictions_runs.xlsx
```

```python
TEST_LIMIT = 0
```

중단된 실행은 성공적으로 저장된 분석 단위를 건너뛰고 이어서 실행할 수 있습니다. 단, 모델 또는 프롬프트가 변경된 경우 기존 결과와 새 결과를 혼합하지 않아야 합니다.

---

### 7.8 GPT 최종 예측 파일 생성

```powershell
python src\07_finalize_gpt_predictions.py
```

주요 출력:

```text
data/results/gpt_predictions_final.csv
data/results/gpt_predictions_final.xlsx
```

최종 파일에는 정서, 대상, 대상에 대한 태도, 조롱·냉소 여부와 함께 GPT의 분류 이유가 `gpt_reason` 열로 저장됩니다.

---

### 7.9 GPT 분류 성능평가

인간 코딩 기준 레이블과 GPT 최종 예측 파일이 완성된 뒤 실행합니다.

```powershell
python src\08_evaluate_gpt.py
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
---

## 8. 성능평가 지표

본 프로젝트의 성능평가는 인간 코딩 레이블을 기준값으로, GPT 레이블을 예측값으로 사용합니다. 정확도, 정밀도, 재현율, F1-score 및 다중범주 macro 평균의 기본 정의는 Sokolova와 Lapalme(2009, pp. 429–430, Tables 2–3)를 따릅니다. Weighted-F1의 support 가중 방식과 실제 계산 규칙은 프로젝트에서 사용하는 `scikit-learn`의 `precision_recall_fscore_support` 정의를 따릅니다.

### 8.1 Accuracy

정확도는 전체 평가 댓글 중 인간 기준 레이블과 GPT 예측 레이블이 일치한 비율입니다.

$$
\mathrm{Accuracy}
=\frac{1}{N}\sum_{n=1}^{N}I(y_n=\hat{y}_n)
=\frac{\sum_{k=1}^{K}C_{kk}}{N}
$$

여기서 $I(\cdot)$는 조건이 참이면 1, 거짓이면 0을 반환하는 지시함수입니다. 이 수식은 이진 분류에서 다음 식과 같습니다.

$$
\mathrm{Accuracy}
=\frac{TP+TN}{TP+TN+FP+FN}
$$

정확도는 전체 분류 성공률을 직관적으로 보여주지만, 특정 범주가 대부분을 차지하는 불균형 자료에서는 다수 범주만 잘 예측해도 높은 값이 나올 수 있습니다. 따라서 본 프로젝트에서는 정확도를 단독으로 해석하지 않고 Macro-F1과 Weighted-F1을 함께 제시합니다.

## 8.2 클래스별 Precision

범주 $k$의 정밀도는 GPT가 범주 $k$로 예측한 댓글 중 실제 인간 코딩에서도 범주 $k$였던 댓글의 비율입니다.

$$
\mathrm{Precision}_k
=\frac{TP_k}{TP_k+FP_k}
$$

정밀도가 높다는 것은 GPT가 특정 범주를 선택했을 때 그 예측이 실제 인간 코딩과 일치할 가능성이 높다는 의미입니다.

## 8.3 클래스별 Recall

범주 $k$의 재현율은 인간 코딩에서 실제로 범주 $k$에 속하는 댓글 중 GPT가 범주 $k$로 올바르게 찾아낸 댓글의 비율입니다.

$$
\mathrm{Recall}_k
=\frac{TP_k}{TP_k+FN_k}
$$

재현율이 높다는 것은 해당 범주에 속하는 댓글을 GPT가 누락하지 않고 포착하는 성능이 높다는 의미입니다.

## 8.4 클래스별 F1-score

범주 $k$의 F1-score는 정밀도와 재현율의 조화평균입니다.

$$
F1_k
=2\times\frac{\mathrm{Precision}_k\times\mathrm{Recall}_k}
{\mathrm{Precision}_k+\mathrm{Recall}_k}
$$

동일한 수식을 혼동행렬의 항으로 나타내면 다음과 같습니다.

$$
F1_k
=\frac{2TP_k}{2TP_k+FP_k+FN_k}
$$

F1-score는 특정 범주를 정확하게 예측하는 정도와 실제 범주를 빠짐없이 포착하는 정도를 함께 반영합니다.

## 8.5 Macro-Precision과 Macro-Recall

Macro 평균은 각 범주의 성능을 동일한 비중으로 평균합니다. 따라서 표본이 적은 범주도 표본이 많은 범주와 동일한 영향력을 갖습니다.

$$
\mathrm{Macro\text{-}Precision}
=\frac{1}{K}\sum_{k=1}^{K}\mathrm{Precision}_k
$$

$$
\mathrm{Macro\text{-}Recall}
=\frac{1}{K}\sum_{k=1}^{K}\mathrm{Recall}_k
$$

현재 평가 코드에서는 인간 기준 자료에 실제로 출현한 범주만 `observed_labels`에 포함해 Macro-Precision, Macro-Recall 및 Macro-F1을 계산합니다. 따라서 여기서 $K$는 해당 평가 범위에서 인간 코딩에 실제로 관찰된 범주 수입니다.

## 8.6 Macro-F1

Macro-F1은 각 범주의 F1-score를 동일한 비중으로 평균합니다.

$$
\mathrm{Macro\text{-}F1}
=\frac{1}{K}\sum_{k=1}^{K}F1_k
$$

본 프로젝트의 Macro-F1은 Macro-Precision과 Macro-Recall을 다시 조화평균한 값이 아니라, `scikit-learn`의 `average="macro"` 규칙에 따라 클래스별 F1-score를 직접 산술평균한 값입니다.

Macro-F1은 소수 범주의 성능 저하를 민감하게 반영하므로, 클래스 불균형으로 인해 특정 범주가 제대로 분류되지 않는 문제를 확인하는 데 사용합니다.

## 8.7 Weighted-F1

Weighted-F1은 각 범주의 F1-score에 인간 기준 자료의 범주별 support 비율을 가중치로 적용한 평균입니다.

$$
\mathrm{Weighted\text{-}F1}
=\sum_{k=1}^{K}\frac{n_k}{N}F1_k
$$

여기서 $n_k/N$은 전체 인간 기준 자료에서 범주 $k$가 차지하는 비율입니다. 표본이 많은 범주의 성능이 최종 점수에 더 크게 반영되므로, 실제 자료의 범주 분포를 고려한 전체 성능을 나타냅니다.

현재 평가 코드에서는 `labels.py`에 정의된 전체 범주를 대상으로 `average="weighted"`를 적용합니다. 인간 기준 자료에 출현하지 않은 범주는 support가 0이므로 Weighted-F1에 실질적인 가중치를 갖지 않습니다.

본 연구는 범주별 비중 차이가 큰 자료의 전체 성능을 평가하기 위해 Weighted-F1을 주요 지표로 사용하고, 소수 범주를 포함한 균등한 범주별 성능을 확인하기 위해 Macro-F1을 병행하여 제시합니다.

---

## 9. 시각화


### 9.1 전체 GPT 결과 시각화

```powershell
python src\09_visualize_gpt_results.py
```

주요 출력:

```text
data/results/figures/figure_01_sentiment.png
data/results/figures/figure_02_target.png
data/results/figures/figure_03_stance.png
data/results/figures/figure_04_is_sarcasm_mockery.png
data/results/figures/figure_05_target_by_stance.png
```

정서, 대상, 태도, 조롱·냉소 여부 그래프에는 통합데이터와 사례별 결과가 함께 표시됩니다.

---

### 9.2 성능평가 지표 시각화

```powershell
python src\10_visualize_gpt_evaluation.py
```

주요 출력:

```text
data/results/figures/evaluation/figure_01_overall_metrics.png
data/results/figures/evaluation/gpt_evaluation_plot_data.csv
```

하나의 묶음 세로 막대그래프에서 다음 네 과업을 비교합니다.

```text
정서 극성
평가 대상
대상에 대한 태도
조롱·냉소 여부
```

각 과업에는 다음 다섯 성능지표가 함께 표시됩니다.

```text
Accuracy
Macro Precision
Macro Recall
Macro F1
Weighted F1
```

---

## 10. 데이터 공개 범위

이 저장소에는 플랫폼 이용조건과 연구윤리, 개인정보 보호 및 재배포 가능성을 검토한 뒤 다음 자료를 포함하지 않습니다.

```text
YouTube 원댓글 및 대댓글
댓글 작성자 정보
YouTube API 키
OpenAI API 키
인간 코딩 파일
GPT 원시 응답
분석 결과 파일
```

---

## 11. 재현성

연구 결과의 재현을 위해 다음 정보를 함께 기록합니다.

- Python 버전
- 패키지 버전
- GPT 모델 ID 및 고정 스냅샷
- OpenAI 서비스 처리 방식
- 프롬프트 버전
- 프롬프트 해시
- 구조화 출력 스키마
- 무작위 추출 시드
- 인간 코더 수
- 인간 코딩 표본 수
- 대상과 태도의 결합 분류 규칙
- 데이터 수집 시점


---

본 프로젝트의 실제 계산은 `scikit-learn`의 `accuracy_score`, `precision_recall_fscore_support`, `confusion_matrix`를 사용합니다. 특히 Macro-F1은 클래스별 F1-score의 비가중 산술평균이며, Weighted-F1은 인간 기준 support로 가중한 클래스별 F1-score의 평균입니다.
