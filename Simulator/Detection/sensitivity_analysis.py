# 민감도 분석 진행을 위한 파일
from typing import List, Optional
from Simulator.Generation.dataset_builder import build_dataset
from Simulator.schema_utils import get_feature_columns
from Simulator.Detection.train_eval import train_model, evaluate, prepare_categorical
from sklearn.model_selection import StratifiedKFold


def run_sensitivity(
    param_name: str,                    # 스윕 대상인 feature 이름. config.py 변수명이 아니라 generator의 _get_soph() 조회 key(한글)  ex) "URL_존재확률"
    candidate_values: List[str],        # param_name에 하나씩 대입해볼 값 목록. sweep_target="sophistication"이면 [LOW, MID, HIGH], "subgroup_ratio_key"면 ["A".."E"]
    fixed_config,                       # config.py 모듈. param_name 외 나머지 값은 전부 여기서 그대로(고정) 가져다 씀. Detection/ 안에서 config.py import 하지 않기 위한 parameter.
    sweep_target: str = "sophistication",  # 무엇을 스윕할지: "sophistication"(_get_soph() 기반 feature) | "subgroup_ratio_key"(RELATION_TYPE_RATIO, A~E)
    base_sophistication: Optional[str] = None,  # param_name을 제외한 다른 feature들에 공통 적용할 sophistication. 스윕 대상인 feature 외의 다른 feature들이 LOW/MID/HIGH 중 어떤 값을 쓸지 결정. (보통, MID)
    n: int = 10000,                     # 후보값마다 생성할 사건(row) 개수. 후보값끼리 동일해야 공정 비교 가능
    phishing_rate: Optional[float] = None,  # build_dataset()에 넘길 클래스 불균형 비율(보통 config.CLASS_IMBALANCE)
    subgroup_ratio_key: str = "B",      # 정상 이벤트 하위집단(지인/기관) 비중 키. sweep_target="sophistication"일 때만 유효(고정값으로 사용)
    gen_seed: int = 42,                 # build_dataset()용 random_state. 후보값 사이에 동일해야 "파라미터만 바뀌었다"고 말할 수 있음
    fold_seed: int = 7,                 # StratifiedKFold용 random_state. 마찬가지로 후보값 사이에 고정 필요
    k: int = 5,                         # K-Fold 분할 개수
):
    # 한 feature에 대한 민감도 분석 진행을 위한 함수.
    # K-Fold 학습/평가 -> 결과 비교표 반환.

    # 스윕 대상 외 feature에 쓸 sophistication. None이면 mid.
    base_soph = base_sophistication or "mid"
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=fold_seed)

    summary = []
    for value in candidate_values:
        # dataset 생성을 위한 build_dataset() 호출.
        # candidate_values가 [LOW, MID, HIGH]라면 각각의 sophistication 값에 대해 총 3번의 for문 반복.
        # sophistication에 {param_name: value} + __base__를 넘기면, 스윕 대상만 value,
        # 나머지 feature는 base_soph(_get_soph의 __base__)로 고정됨.
        if sweep_target == "subgroup_ratio_key":
            # RELATION_TYPE_RATIO(A~E) 스윕: sophistication이 아니라 build_dataset()의
            # subgroup_ratio_key 인자 자체가 바뀜. sophistication은 지인/기관 비중과
            # 무관한 축이라 "mid"로 고정. (base_soph와 무관)
            df = build_dataset(
                n,
                phishing_rate=phishing_rate,
                subgroup_ratio_key=value,
                random_state=gen_seed,
                config=fixed_config,
                sophistication="mid",
            )
        else:
            # 기존 방식: _get_soph() 조회 key(param_name)에 해당하는 feature의
            # sophistication만 candidate_values로 바꿔가며 스윕.
            df = build_dataset(
                n,
                phishing_rate=phishing_rate,
                subgroup_ratio_key=subgroup_ratio_key,
                random_state=gen_seed,
                config=fixed_config,
                sophistication={"__base__": base_soph, param_name: value},
            )

        df = prepare_categorical(df)  # K-Fold로 나뉘기 전에 category dtype 한 번만 확정
        X = df[get_feature_columns()] # 학습에 사용할 feature columns 관련 데이터만 추출.
        y = df["is_phishing"] # label column.

        
        fold_results = [] # 이 후보값(value) 하나에 대한 K-Fold 결과를 모으는 곳.

        for train_idx, val_idx in skf.split(X, y):
            # train_idx : train fold로 쓸 행들의 위치(0부터 세는 positional 정수) 배열.
            # val_idx   : validation fold로 쓸 행들의 위치 배열.

            train_fold = df.iloc[train_idx] # train fold에 해당하는 data
            val_fold = df.iloc[val_idx] # validation fold에 해당하는 data

            # val_fold는 학습에 안 쓰고, 오직 평가에만 사용 -> 표준 K-Fold 방식.
            # threshold는 train_fold에서 고르고 val_fold에 고정 적용.
            model = train_model(train_fold)
            fold_results.append(evaluate(model, val_fold, threshold_df=train_fold))

        summary.append({
            "param_name": param_name,
            "value": value,
            "fold_results": fold_results,  # K개 fold의 evaluate() 결과 목록. 평균/표준편차 집계는 evaluate() 반환 형태 확정 후 추가.
        })

    return summary
