# Structure.md — 프로젝트 폴더 구조 안내

> "이 파일이 뭐 하는 파일인지, 어디에 뭘 만들어야 하는지" 헷갈리지 않도록 정리한 문서.
> 코드 구조가 바뀔 때마다 이 문서도 같이 업데이트할 것.

---

## 전체 구조

```
(프로젝트 루트)
├── README.md
├── Structure.md                
└── Simulator/
    ├── main.py 
    ├── schema.py              
    ├── Generation/
    │   ├── config.py            
    │   ├── normal_generator.py  
    │   ├── phishing_generator.py  
    │   └── dataset_builder.py    
    └── Detection/
        ├── train_eval.py         
        └── sensitivity_analysis.py 
```

---

## Generation과 Detection 분리 목적

**"Generation-Detection Separation" 원칙**(AMLNet 3.3절 근거) 때문.

- `Generation/` = 가짜 데이터를 만드는 부분
- `Detection/` = 그 데이터로 모델을 학습 및 평가하는 부분

핵심 규칙: `Detection/` 안의 어떤 코드도 `Generation/config.py`를 import하면 안 됨.
데이터를 만들 때 쓴 파라미터(θ값, 지인:기관 비중 등)를 학습 코드가 알고 있으면, 그 정보가 실수로 feature처럼 새어 들어가서 "생성 규칙을 암기하는" 모델이 될 위험이 있음. 두 폴더를 코드 상으로도 분리해두면 이런 실수가 구조적으로 원천 차단되므로 두 폴더를 분리 생성함.

---

## 폴더/파일별 역할

### `Simulator/main.py` 

역할: 전체 파이프라인의 진입점. Generation과 Detection을 순서대로 호출하는 최상위 스크립트.

하는 일 (예정):
- 민감도분석 모드: `sensitivity_analysis.py`를 호출해서, config.py의 각 파라미터를 하나씩 바꿔가며 여러 데이터셋 생성 → 학습 → 결과 비교표 출력
- 최종 모드: 팀이 확정한 값으로 데이터셋 1개 생성 → 학습 → 최종 성능 출력

주의: `main.py`는 `Generation/config.py`를 직접 import해도 됨(생성 단계를 총괄 지휘하는 역할이기 때문). 문제가 되는 건 `Detection/` 내부 코드가 config를 보는 것이므로 `main.py`에서는 고려하지 않아도 됨.

---

### 'Simulator/schema.py

역할: Dataset의 column 정의를 위한 class 선언 파일.

---

### `Generation/config.py` 

역할: 마스터표에서 확정한 "정상 baseline 가중치"(A~G그룹) 전체를 담는 설정 파일. 코드가 아닌 값들의 저장소 역할.

포함된 것:
- `[확정]` 표시: 바로 시뮬레이터에 쓸 수 있는 값들 (A그룹 단일확률, B그룹 카테고리비중, D그룹 전환간격 등)
- `[보류]` 표시: 아직 근거 부족·팀 미결정이라 `None`으로 남겨둔 TODO 항목들 (재발신/재연락, 활동시간대, 통신매개총량 등)

현재 미해결 사항 (이 파일 안에 주석으로도 표시돼 있음):
1. 재발신/재연락 정상값 — 마스터표 서술 자체부터 다시 채워야 함
2. 지인:기관 / 개인:기업 비중을 "연동"할지 "독립"으로 둘지
3. 클래스 불균형 비율(1~5%) — AMLNet 사례(0.16%)보다 후한 편일 수 있어 재검토 필요
4. 유형별 비중 — (A)/(B) 논의사항 중 (B) 선택 시에만 사용, 협박형 비중 근거 없음

---

### `Generation/normal_generator.py` 

역할: 정상 이벤트 1건을 생성하는 함수를 담는 파일.

만들 함수 (예정): `generate_normal_event(subgroup, config)`
- Algorithm1의 9~10번 줄(하위집단 지인/기관 결정 → 그 규칙으로 이벤트 생성)에 대응
- `config.py`의 F그룹(예외율), 개인기업_비중_후보 등을 참조
- **[보류] 항목(재발신/재연락 등)은 팀 논의 끝나기 전까지 이 함수에서 빈 값(NaN)이나 placeholder로 처리해야 함**

---

### `Generation/phishing_generator.py` 

역할: 피싱 이벤트 1건을 생성하는 함수를 담는 파일.

만들 함수 (예정): `generate_phishing_event(type, sophistication, config)`
- Algorithm1의 5~7번 줄(유형선택 → 값-옵션선택 → 생성)에 대응
- `config.py`의 `TYPE_RATIO`, `SMS_TO_CALL_GAP` 등을 참조
- **(A)/(B) 논의사항에서 (B)로 최종 결정될 경우에만, 유형(대출/기관/지인사칭/협박)별로 다른 θ값을 적용하는 분기 로직이 들어감. (A)로 결정되면 이 함수 구조가 달라져야 함 — 팀 논의 후 재작성 필요**

---

### `Generation/dataset_builder.py` 

역할: `normal_generator`와 `phishing_generator`를 실제로 n번 호출해서, 완성된 데이터셋(표) 하나를 만드는 파일. Algorithm1 전체를 구현하는 곳.

만들 함수 (예정): `build_dataset(n, phishing_rate, config)`
- Algorithm1의 1~18번 줄 전체(for문, if/else 분기, 라벨링)를 그대로 코드로 옮김
- 반환값: pandas DataFrame (사건ID, feature 15~19개, 피싱여부 라벨, 유형라벨)

---

### `Detection/train_eval.py` 

역할: 데이터셋 하나를 받아서, 분할→학습→평가까지 한 번에 처리하는 파일.

만들 함수 (예정):
- `split_data(df)` — 층화(stratify) 방식으로 학습/검증/테스트 분할
- `train_model(train, val)` — 트리기반 모델(XGBoost 등) 학습
- `evaluate(model, test)` — accuracy/precision/recall/F1/feature importance 반환

**이 파일은 `Generation/config.py`를 import하지 않아야 함.** 

---

### `Detection/sensitivity_analysis.py` 

역할: config.py의 스윕 파라미터(짧게/중간/길게, 지인기관비중 A~E 등)를 하나씩 바꿔가며 `dataset_builder` + `train_eval`을 반복 실행하고, 결과를 비교표로 정리하는 파일.

만들 함수 (예정): `run_sensitivity(param_name, candidate_values, fixed_config)`
- "한 번에 하나씩만 바꾸고 나머지는 고정"이라는 원칙을 그대로 구현
- 결과: 파라미터값별 성능 비교 DataFrame → 변동폭이 큰(취약한) 파라미터 식별용

---

## Algorithm1과 파일의 대응 관계 (참고용)

| Algorithm1 줄 | 담당 파일 |
|---|---|
| Require (n, phishing_rate, config) | `main.py`에서 값을 정해서 `dataset_builder.build_dataset()`에 전달 |
| 1~4, 8, 11, 12~17 (전체 루프·분기·라벨링) | `dataset_builder.py` |
| 5~7 (피싱 이벤트 생성) | `phishing_generator.py` |
| 9~10 (정상 이벤트 생성) | `normal_generator.py` |
| 18 (반환) | `dataset_builder.py`의 리턴값 |

---

## 제안 작업 순서

1. `normal_generator.py`, `phishing_generator.py` 작성 (config.py의 `[확정]` 항목만 우선 반영, `[보류]`는 placeholder)
2. `dataset_builder.py` 작성 — Algorithm1 그대로 구현, 소량(10~20건) 생성해서 눈으로 검증
3. `train_eval.py` 작성 — 층화분할+RandomForest 학습
4. `sensitivity_analysis.py` 작성
5. `main.py`에서 전체 연결
6. 팀 미결정 사항([보류] 항목들, (A)/(B) 결정) 회의 후 config.py 업데이트 → 위 순서 재실행