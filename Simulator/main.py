# Simulator 코드 작성을 위한 main.py
# 역할: 전체 파이프라인의 진입점. mode에 따라 Generation과 Detection을 순서대로 호출.

import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from Simulator.Generation import config
from Simulator.Generation.dataset_builder import build_dataset
from Simulator.Detection.train_eval import split_data, train_model, evaluate, prepare_categorical
from Simulator.Detection.sensitivity_analysis import run_sensitivity, run_stress_sensitivity
from Simulator.Detection.feature_ablation import run_feature_selection_pipeline
from Simulator.Detection.optuna_apply import run_optuna_tuning_and_compare
from Simulator.schema import Track
from Simulator.schema_utils import get_feature_columns

# 최종 dataset 저장 폴더: 실행 위치(cwd)와 무관하게 항상 프로젝트 루트 기준으로 고정.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATASET_DIR = _PROJECT_ROOT / "DataSet"
_TUNE_RESULTS_DIR = _PROJECT_ROOT / "optuna_results"


# 최종 모드 실행 파라미터. "이번 최종 실행을 어떻게 돌릴지"에 대한 파이프라인 설정값이라 main.py에 지역 상수로 둠.
N = 100000                  # 생성할 사건 수
SUBGROUP_RATIO_KEY = "D"   # 정상 이벤트 하위집단(지인/기관) 비중 키 (민감도 분석 결과 B->D로 확정)
GEN_SEED = 42              # build_dataset()용 random_state

# DataSet/final_dataset 캐시 무효화용 버전. N/GEN_SEED/SUBGROUP_RATIO_KEY/FINAL_SOPHISTICATION은
# 바뀌면 자동으로 감지되지만(fingerprint 비교), "생성 로직 자체"가 바뀌는 경우(코드는 바뀌었는데
# 위 파라미터 값은 그대로인 경우)는 자동으로 감지가 안 되므로, 그럴 때 이 숫자를 수동으로 올려야
# get_or_generate_final_dataset()이 저장된 dataset을 재사용하지 않고 강제로 다시 생성함.
# TODO: 파생 feature ablation 검증이 끝나서 add_candidate_features()를 generate_final_dataset()에
#       연결하게 되면, 그 시점에 이 값을 1 -> 2로 올릴 것(그래야 파생 feature 없이 저장된
#       기존 dataset이 재사용되지 않고 파생 feature 포함 버전으로 다시 생성됨).
DATASET_GEN_VERSION = 1

# 최종 dataset 생성 시 각 feature에 적용할 sophistication. 민감도 분석(필요 시 K=10/stress 모드
# 재검증까지 거쳐) 결과로 확정한 값. build_dataset()의 sophistication 인자로 그대로 넘어감
FINAL_SOPHISTICATION = {
    # 하나
    config.SOPH_NUMBER_TYPE_BAND: "mid",
    config.SOPH_FIRST_CONTACT: "high",
    config.SOPH_REPEAT_GAP: "low",           # K=10 재검증 완료
    config.SOPH_REPEAT_CONTACT: "mid",
    # 지민
    config.SOPH_NUM_IN_MSG: "mid",           # stress 모드로 재검증
    config.SOPH_INNER_NUM_DIFFERS: "mid",
    config.SOPH_MSG_OFFICIAL_MATCH: "high",
    config.SOPH_SMS_TO_CALL_GAP: "mid",      # K=10 재검증 완료
    # 다은
    config.SOPH_URL_RATE: "mid",             # stress 모드로 재검증
    config.SOPH_APP_INSTALL: "low",
    config.SOPH_SEQUENTIAL_CALLERS: "mid",
    config.SOPH_SMS_TO_CALL: "high",
}

# Track별 접근 가능 feature 시나리오 (누적 구조: 통신사가 가장 넓은 범위에 접근 가능).
# 최종 dataset 1개를 그대로 두고, 시나리오별로 사용하는 컬럼만 달라짐.
TRACK_SCENARIOS = {
   # "A": [Track.CARRIER, Track.DEVICE, Track.DEVICE_STRUCTURAL],  # 통신사: A+B+C 전부 접근 가능 (현재로써는 접근 가능한 데이터 X)
    "B": [Track.DEVICE],                                          # 단말: B만 접근 가능
    "C": [Track.DEVICE, Track.DEVICE_STRUCTURAL],                 # 단말+구조적신호: B+C 접근 가능
}


# 민감도분석 대상 feature 목록. run_sensitivity()의 param_name으로 그대로 넘어감.
# config.py의 SOPH_* 상수와 일치해야 함(= generator_utils.get_soph() 조회 key).
# 하드코딩 문자열 대신 상수를 써서, 오타 시 여기서 바로 NameError로 드러나게 함.
SENSITIVITY_PARAMS = [
    # 하나
    config.SOPH_NUMBER_TYPE_BAND,
    config.SOPH_FIRST_CONTACT,
    config.SOPH_REPEAT_GAP,          # 정상 전용 (재연락 간격 배율/theta 크기)
    config.SOPH_REPEAT_CONTACT,      # 정상+피싱 공용 (사건 내 반복 접촉 존재확률)
    # 지민
    config.SOPH_NUM_IN_MSG,
    config.SOPH_INNER_NUM_DIFFERS,
    config.SOPH_MSG_OFFICIAL_MATCH,
    config.SOPH_SMS_TO_CALL_GAP,     # 정상 전용
    # 다은
    config.SOPH_URL_RATE,            # 정상 전용
    config.SOPH_APP_INSTALL,
    config.SOPH_SEQUENTIAL_CALLERS,
    config.SOPH_SMS_TO_CALL,         # 피싱 전용
]

# mode="each_sen"일 때 확인할 feature 1개. 바꾸고 싶으면 이 줄만 수정(config.SOPH_* 중 하나).
EACH_SENSITIVITY_PARAM = config.SOPH_SMS_TO_CALL_GAP

# mode="each_sen"일 때 쓸 K. 기본 5(표준). 재검증하려면 이 줄만 10으로 수정 ->
# sensitivity_results/<param_name>_k10.json으로 기존 K=5 결과와 별도 저장됨.
EACH_SENSITIVITY_K = 10


# mode="stress" 대상 시나리오 목록. run_stress_sensitivity()의 scenario_key/level_overrides로 그대로 넘어감.
# config.py의 STRESS_*_KEY / STRESS_*_LEVELS 상수와 짝을 맞춰서 등록.
# "순수가정"으로 정한 feature(예: num_in_msg, url_rate)의 절대값 자체가 틀렸을 때 성능이
# 얼마나 흔들리는지 확인하기 위한 것으로, low/mid/high 3-preset 비교인 SENSITIVITY_PARAMS와는 별개.
STRESS_SCENARIOS = {
    config.STRESS_NUM_IN_MSG_KEY: config.STRESS_NUM_IN_MSG_LEVELS,
    config.STRESS_URL_RATE_KEY: config.STRESS_URL_RATE_LEVELS,
}

# mode="stress"일 때 확인할 시나리오 1개. 바꾸고 싶으면 이 줄만 수정(STRESS_SCENARIOS의 key 중 하나).
STRESS_SCENARIO_KEY = config.STRESS_URL_RATE_KEY

# mode="stress"일 때 쓸 K. 기본 5(표준).
STRESS_K = 5

# mode="ablation"일 때 쓸 다중공선성/데이터 누수 판정 임계값. feature_ablation.py의
# 기본값(0.8/0.95)을 그대로 쓰고 싶으면 None으로 둠 -> 바꾸고 싶을 때만 이 줄 수정.
ABLATION_MULTICOLLINEARITY_THRESHOLD = None
ABLATION_LEAKAGE_THRESHOLD = None

# mode="tune"일 때 Optuna trial 횟수 / validation 비율(train_set 중 permutation importance와
# 동일한 방식으로 train_sub/val_sub를 나눠서, val_sub로만 탐색 평가 -> test_set은 안 건드림).
TUNE_N_TRIALS = 50
TUNE_VAL_SIZE = 0.2


# ------------------------------------------------------------
# 최종 dataset 캐싱 메커니즘 (generate_final_dataset / get_or_generate_final_dataset)
# ------------------------------------------------------------
# 문제: run_final()이 매번 build_dataset()으로 처음부터 다시 생성하면, N=100000 기준
#      몇 초씩 낭비됨. 그런데 GEN_SEED 등이 고정이라 매번 만들어도 "완전히 동일한" dataset이
#      나오니, 이미 저장된 파일이 있으면 그냥 그걸 읽는 게 더 낫다.
# 해결: 저장할 때(save_final_dataset) dataset과 함께 "이번에 쓴 생성 파라미터"를
#      final_dataset.meta.json에 지문(fingerprint)으로 같이 남긴다. 다음에 get_or_
#      generate_final_dataset()이 불릴 때, 지금 설정의 지문과 저장된 지문을 비교해서:
#        - 지문이 같음 -> 파라미터가 하나도 안 바뀌었다는 뜻 -> parquet 파일을 그대로 읽어서 반환
#        - 지문이 다름(또는 파일이 아예 없음) -> N/시드/subgroup/sophistication 중 뭔가
#          바뀌었다는 뜻 -> generate_final_dataset()으로 새로 생성해서 덮어씀
# 한계: 이 지문은 "숫자로 바뀌는 파라미터"만 감지한다. 파생 feature를 여기 새로 연결하는
#      것처럼, 파라미터 값은 그대로인데 "생성 함수 안의 코드 자체"가 바뀌는 경우는 지문에
#      안 잡힌다 -> 그럴 때는 DATASET_GEN_VERSION 숫자를 수동으로 올려서 지문을 강제로
#      바꿔야 재사용되지 않고 새로 생성된다(이것도 못 하면 코드는 바뀌었는데 예전 dataset이
#      계속 재사용되는 채로 조용히 넘어갈 수 있음).
# mode="generate"는 이 캐시를 무시하고 generate_final_dataset()을 직접 불러 항상 강제로
# 새로 만든다(예: 캐시가 있어도 일부러 재현성만 다시 확인하고 싶을 때).
def _dataset_fingerprint() -> dict:
    """지금 이 실행의 생성 파라미터 지문. get_or_generate_final_dataset()이 저장된 dataset을
    재사용해도 되는지 판단하는 기준으로 씀 -> N/GEN_SEED/SUBGROUP_RATIO_KEY/FINAL_SOPHISTICATION
    중 하나라도 달라지면 값이 달라져서 자동으로 재생성됨."""
    return {
        "version": DATASET_GEN_VERSION,
        "n": N,
        "gen_seed": GEN_SEED,
        "subgroup_ratio_key": SUBGROUP_RATIO_KEY,
        "sophistication": FINAL_SOPHISTICATION,
    }


def save_final_dataset(df, out_dir: Path = _DATASET_DIR) -> None:
    """생성 직후(카테고리 변환/분할 전)의 원본 dataset을 DataSet/ 폴더에 CSV+Parquet 둘 다로 저장.
    같은 이름으로 덮어씀(재실행 시 항상 최신 dataset만 유지, 버전별 파일 누적 안 함).
    이번 생성에 쓴 파라미터 지문도 final_dataset.meta.json으로 같이 저장 -> 나중에
    get_or_generate_final_dataset()이 재사용 가능 여부를 판단하는 데 씀."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "final_dataset.csv", index=False, encoding="utf-8-sig")  # utf-8-sig: Excel에서 한글 깨짐 방지
    df.to_parquet(out_dir / "final_dataset.parquet", index=False)  # dtype(카테고리 등) 그대로 보존
    with open(out_dir / "final_dataset.meta.json", "w", encoding="utf-8") as f:
        json.dump(_dataset_fingerprint(), f, ensure_ascii=False, indent=2)


def generate_final_dataset():
    """최종 확정한 config 값으로 dataset 1개를 새로 생성해서 DataSet/ 폴더에 저장하고 반환
    (기존 저장분이 있어도 무조건 다시 만듦 -> mode="generate"에서 "강제로 새로 만들고 싶을 때" 씀).
    GEN_SEED/FINAL_SOPHISTICATION/SUBGROUP_RATIO_KEY가 고정이라 몇 번을 다시 돌려도 완전히
    동일한 dataset이 나옴(재현성 보장). 평소엔 이 함수를 직접 부르기보다
    get_or_generate_final_dataset()을 쓰는 걸 권장(불필요한 재생성 방지)."""
    df = build_dataset(
        n=N,
        phishing_rate=config.CLASS_IMBALANCE,
        subgroup_ratio_key=SUBGROUP_RATIO_KEY,
        random_state=GEN_SEED,
        config=config,
        sophistication=FINAL_SOPHISTICATION,
    )
    save_final_dataset(df)  # Detection 쪽 가공(category 변환/분할) 전, 생성 직후의 원본을 저장
    # 파생 feature 13개는 전부 ablation 검증 대기 중이라 여기서는 아직 추가하지 않음
    # (Detection/derived_features.py의 add_candidate_features() 참고). 검증 끝나고 여기에
    # 연결하게 되면 DATASET_GEN_VERSION을 올려서 기존 캐시를 무효화할 것.
    return df


def get_or_generate_final_dataset():
    """DataSet/final_dataset.parquet + final_dataset.meta.json이 있고, 그 지문이 지금
    _dataset_fingerprint()와 완전히 같으면 저장된 dataset을 그대로 읽어서 반환(재생성 생략).
    하나라도 없거나 지문이 다르면(N/시드/sophistication 변경, 또는 DATASET_GEN_VERSION을
    수동으로 올린 경우) generate_final_dataset()으로 새로 생성."""
    parquet_path = _DATASET_DIR / "final_dataset.parquet"
    meta_path = _DATASET_DIR / "final_dataset.meta.json"

    if parquet_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            saved_fingerprint = json.load(f)
        if saved_fingerprint == _dataset_fingerprint():
            return pd.read_parquet(parquet_path)

    return generate_final_dataset()


def run_final():
    """최종 모드: get_or_generate_final_dataset()으로 dataset 확보(저장된 게 있고 파라미터가
    그대로면 재사용, 아니면 새로 생성) -> split_data() 2분할(train/test) -> optuna_results/
    best_params.json이 있으면 그 모델 종류(XGBoost/LightGBM 중 승자)+하이퍼파라미터를 사용
    (없으면 train_model() 기본값=XGBoost) -> Track 시나리오(B/C)별로 feature만 다르게 골라
    train_model()/evaluate() 반복."""
    df = get_or_generate_final_dataset()

    df = prepare_categorical(df)  # train/test로 나뉘기 전에 category dtype 한 번만 확정

    train_set, test_set = split_data(df) # train/test 2분할

    model_type, tuned_hyperparams = load_tuned_hyperparams()  # 없으면 (None, None) -> 기본값 사용
    model_type = model_type or "XGBoost"

    results = {}  # {시나리오명("B"/"C"): evaluate() 결과}
    for scenario_name, tracks in TRACK_SCENARIOS.items():
        # TRACK_SCENARIOS 개수만큼 반복. (Track별로 다른 모델 생성 및 평가 진행)
        cols = get_feature_columns(track_filter=tracks)
        model = train_model(train_set, feature_cols=cols, hyperparams=tuned_hyperparams, model_type=model_type)
        # threshold는 train에서 고르고, 점수는 test에 고정 적용 (낙관 편향 방지)
        results[scenario_name] = evaluate(
            model, test_set, feature_cols=cols, threshold_df=train_set
        )
    return results


def run_sensitivity_mode():
    """민감도분석 모드: config.py 스윕 파라미터를 하나씩 바꿔가며 run_sensitivity() 반복 호출 -> 비교표 출력."""
    result = {}  # {param_name: run_sensitivity()의 반환값(summary 리스트)} 형태로 feature별 결과를 보관

    for param_name in SENSITIVITY_PARAMS:
        # run_sensitivity() 안에서 sensitivity_results/<param_name>.json으로 자동 저장됨.
        result[param_name] = run_sensitivity(
            param_name=param_name,
            candidate_values=["low", "mid", "high"],
            fixed_config=config,
            phishing_rate=config.CLASS_IMBALANCE,
            n=N,
            subgroup_ratio_key=SUBGROUP_RATIO_KEY,
        )

    result["subgroup_ratio_key"] = run_sensitivity_ratio_mode()

    return result


def run_sensitivity_ratio_mode():
    """지인:기관 하위집단 비중(RELATION_TYPE_RATIO, A~E)만 따로 민감도분석 진행.
    sophistication 기반이 아니라 build_dataset()의 subgroup_ratio_key 자체를 바꾸는 축이라
    SENSITIVITY_PARAMS(SOPH_* 스윕)와는 별개로 분리."""
    # run_sensitivity() 안에서 sensitivity_results/subgroup_ratio_key.json으로 자동 저장됨.
    summary = run_sensitivity(
        param_name="subgroup_ratio_key",
        candidate_values=list(config.RELATION_TYPE_RATIO.keys()),  # ["A", "B", "C", "D", "E"]
        fixed_config=config,
        sweep_target="subgroup_ratio_key",
        phishing_rate=config.CLASS_IMBALANCE,
        n=N,
    )
    return summary


def run_sensitivity_each_feature_mode(param_name, k=5):
    """각 feature에 대해 진행 가능하도록 하는 함수(run_sensitivity_mode()는 전체 feature에 대해서.)
    k: 기본 5(표준). 재검증 등으로 늘리고 싶으면(예: 10) 여기로 넘기면 됨 ->
       sensitivity_results/<param_name>_k<k>.json으로 기존 K=5 결과와 별도 저장됨."""
    # run_sensitivity() 안에서 sensitivity_results/<param_name>.json(또는 _k<k>.json)으로 자동 저장됨.
    summary = run_sensitivity(
        param_name=param_name,
        candidate_values=["low", "mid", "high"],
        fixed_config=config,
        phishing_rate=config.CLASS_IMBALANCE,
        n=N,
        subgroup_ratio_key=SUBGROUP_RATIO_KEY,
        k=k,
    )

    return summary


def run_stress_mode(scenario_key=None, k=None):
    """가정-파괴(stress) 모드: scenario_key 1개에 대해 run_stress_sensitivity() 호출.
    sophistication은 "mid" 고정, config.py의 STRESS_*_LEVELS에 정의된 level별로 확률/θ 절대값만 스윕."""
    if scenario_key is None:
        scenario_key = STRESS_SCENARIO_KEY
    if k is None:
        k = STRESS_K
    if scenario_key not in STRESS_SCENARIOS:
        raise ValueError(
            f"알 수 없는 stress scenario_key: {scenario_key!r}. STRESS_SCENARIOS 중 하나여야 함: {list(STRESS_SCENARIOS)}"
        )
    # run_stress_sensitivity() 안에서 sensitivity_results/<scenario_key>.json(또는 _k<k>.json)으로 자동 저장됨.
    summary = run_stress_sensitivity(
        scenario_key=scenario_key,
        level_overrides=STRESS_SCENARIOS[scenario_key],
        fixed_config=config,
        description=f"{scenario_key} 가정-파괴(stress) 검증",
        phishing_rate=config.CLASS_IMBALANCE,
        n=N,
        subgroup_ratio_key=SUBGROUP_RATIO_KEY,
        k=k,
    )
    return summary


def run_ablation_mode():
    """파생 feature 선정 파이프라인 모드: baseline(원본만) vs full(원본+13개) 비교 ->
    train set 기준 다중공선성 점검(+데이터 누수 플래그) -> Embedded method(SHAP 참고) +
    Permutation Importance 기반 가지치기 -> 최종 feature set으로 재학습 -> test set 평가.
    결과는 feature_selection_results/feature_selection_pipeline.json에 저장됨."""
    kwargs = dict(
        fixed_config=config,
        n=N,
        phishing_rate=config.CLASS_IMBALANCE,
        subgroup_ratio_key=SUBGROUP_RATIO_KEY,
        sophistication=FINAL_SOPHISTICATION,  # 실제 최종 dataset과 동일한 조건에서 검증되도록
    )
    if ABLATION_MULTICOLLINEARITY_THRESHOLD is not None:
        kwargs["multicollinearity_threshold"] = ABLATION_MULTICOLLINEARITY_THRESHOLD
    if ABLATION_LEAKAGE_THRESHOLD is not None:
        kwargs["leakage_threshold"] = ABLATION_LEAKAGE_THRESHOLD
    return run_feature_selection_pipeline(**kwargs)


def load_tuned_hyperparams(out_dir: Path = _TUNE_RESULTS_DIR):
    """optuna_results/best_params.json이 있으면 (model_type, tuned_hyperparams) 튜플을 반환,
    없으면 (None, None)(= train_model()이 기본 XGBoost 하이퍼파라미터를 그대로 씀).
    "winner" 필드(XGBoost/LightGBM 비교 결과)를 기준으로 이긴 쪽 하이퍼파라미터만 꺼내옴."""
    path = out_dir / "best_params.json"
    if not path.exists():
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    winner = payload.get("winner")
    if winner is None:
        return None, None
    return winner, payload[winner]["tuned_hyperparams"]


def run_tuning_mode():
    """Optuna 하이퍼파라미터 탐색 모드: get_or_generate_final_dataset()으로 dataset 확보
    -> train/test 분할(test는 안 건드림) -> train_set 안에서 다시 train_sub/val_sub 분할
    (feature_ablation.py와 동일한 방식) -> val_sub 기준으로 XGBoost/LightGBM 둘 다 Optuna 탐색
    후 더 나은 쪽(winner)을 선택.
    결과는 optuna_results/best_params.json에 저장되고, run_final()이 다음 실행부터
    이 파일을 자동으로 읽어서(winner 모델 + 그 하이퍼파라미터) 반영함."""
    df = get_or_generate_final_dataset()
    df = prepare_categorical(df)
    train_set, test_set = split_data(df)  # test_set은 여기서 전혀 안 씀

    train_sub, val_sub = train_test_split(
        train_set, test_size=TUNE_VAL_SIZE, random_state=42, stratify=train_set["is_phishing"]
    )

    feature_cols = get_feature_columns()
    X_train, y_train = train_sub[feature_cols], train_sub["is_phishing"]
    X_val, y_val = val_sub[feature_cols], val_sub["is_phishing"]

    result = run_optuna_tuning_and_compare(X_train, y_train, X_val, y_val, n_trials=TUNE_N_TRIALS)

    _TUNE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_TUNE_RESULTS_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main(mode: str, param_name=None, k=None, scenario_key=None):
    # param_name/k: mode="each_sen"일 때만 사용. scenario_key/k: mode="stress"일 때만 사용.
    # 둘 다 직접 Python으로 호출할 때만 넘기는 선택 인자이고, CLI(-m each_sen / -m stress)로는
    # 안 받음 -> 코드 상단의 EACH_SENSITIVITY_PARAM/EACH_SENSITIVITY_K, STRESS_SCENARIO_KEY/
    # STRESS_K를 바꿔서 확인할 feature(또는 시나리오)/K를 정함(안 넘기면 그 상수값을 그대로 씀).
    # mode="ablation"은 K-Fold가 아니라 train/test 고정 분할 기반이라 k를 받지 않음 ->
    # 임계값을 바꾸려면 ABLATION_MULTICOLLINEARITY_THRESHOLD/ABLATION_LEAKAGE_THRESHOLD를 수정.
    if mode == "generate":
        result = generate_final_dataset()
    elif mode == "final":
        result = run_final()
    elif mode == "sen":
        result = run_sensitivity_mode()
    elif mode == "each_sen":
        if param_name is None:
            param_name = EACH_SENSITIVITY_PARAM
        if k is None:
            k = EACH_SENSITIVITY_K
        # param_name이 SENSITIVITY_PARAMS(유효한 SOPH_* 값 목록)에 있는지 방어적으로 한 번 더 확인.
        if param_name not in SENSITIVITY_PARAMS:
            raise ValueError(
                f"알 수 없는 param_name: {param_name!r}. SENSITIVITY_PARAMS 중 하나여야 함: {SENSITIVITY_PARAMS}"
            )
        result = run_sensitivity_each_feature_mode(param_name, k=k)
    elif mode == "ratio":
        result = run_sensitivity_ratio_mode()
    elif mode == "stress":
        result = run_stress_mode(scenario_key=scenario_key, k=k)
    elif mode == "ablation":
        result = run_ablation_mode()
    elif mode == "tune":
        result = run_tuning_mode()
    else:
        raise ValueError(f"알 수 없는 mode: {mode}")

    if mode == "generate":
        # result가 DataFrame(N=100000행)이라 그대로 print()하면 너무 장황함 -> 요약만 출력.
        print(f"dataset 생성 완료: shape={result.shape}, 저장 위치={_DATASET_DIR}")
    else:
        print(result)
    return result


# 실행 방법 (프로젝트 루트에서):
#   python -X utf8 -m Simulator.main              -> 기본값(mode=final)
#   python -X utf8 -m Simulator.main -m generate  -> dataset 생성 + DataSet/ 폴더 저장만(학습/평가 없음).
#                                                     final과 완전히 동일한 GEN_SEED/FINAL_SOPHISTICATION을
#                                                     쓰므로 나중에 -m final을 돌려도 100% 동일한 dataset이 나옴.
#   python -X utf8 -m Simulator.main -m final     -> 최종 dataset 생성 -> 학습 -> 평가 (Track B/C)
#   python -X utf8 -m Simulator.main -m sen       -> SENSITIVITY_PARAMS 전체 + subgroup_ratio_key 스윕
#                                                     (feature마다 sensitivity_results/<param_name>.json 자동 저장)
#   python -X utf8 -m Simulator.main -m each_sen  -> feature 1개만 스윕(위 EACH_SENSITIVITY_PARAM/EACH_SENSITIVITY_K 값 사용).
#                                                     확인할 feature나 K를 바꾸려면 CLI가 아니라 코드 상단의
#                                                     EACH_SENSITIVITY_PARAM/EACH_SENSITIVITY_K 줄을 직접 수정할 것.
#   python -X utf8 -m Simulator.main -m ratio     -> 지인:기관 하위집단 비중(RELATION_TYPE_RATIO, A~E)만 스윕
#   python -X utf8 -m Simulator.main -m stress    -> 가정-파괴(stress) 모드: sophistication은 "mid" 고정,
#                                                     STRESS_SCENARIO_KEY 시나리오 1개의 확률/θ 절대값만 스윕
#                                                     (위 STRESS_SCENARIO_KEY/STRESS_K 값 사용). 시나리오나 K를
#                                                     바꾸려면 CLI가 아니라 코드 상단의 두 줄을 직접 수정할 것.
#   python -X utf8 -m Simulator.main -m ablation  -> 파생 feature 선정 파이프라인: baseline(원본만) vs
#                                                     full(원본+파생 13개) 비교 -> train set 기준
#                                                     다중공선성 점검(+데이터 누수 플래그) -> SHAP 참고 +
#                                                     Permutation Importance 기반 가지치기 -> 최종 재학습/평가.
#                                                     feature_selection_results/feature_selection_pipeline.json에 저장.
#                                                     임계값을 바꾸려면 코드 상단의
#                                                     ABLATION_MULTICOLLINEARITY_THRESHOLD/
#                                                     ABLATION_LEAKAGE_THRESHOLD를 직접 수정할 것.
#   python -X utf8 -m Simulator.main -m tune      -> Optuna로 XGBoost/LightGBM 둘 다 하이퍼파라미터
#                                                     탐색 후 val PR-AUC 더 높은 쪽(winner)을 선택.
#                                                     train_set 안에서 train_sub/val_sub로 나눠
#                                                     val_sub 기준으로 탐색(test_set은 안 건드림).
#                                                     결과는 optuna_results/best_params.json에 저장되고,
#                                                     이후 -m final을 돌리면 winner 모델+하이퍼파라미터가 자동 반영됨.
#                                                     trial 수를 바꾸려면 코드 상단의 TUNE_N_TRIALS 수정.
#   -m은 --mode의 짧은 별칭.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Voice Phishing Detection 시뮬레이터 파이프라인 실행")
    parser.add_argument(
        "-m", "--mode", default="final",
        choices=["generate", "final", "sen", "each_sen", "ratio", "stress", "ablation", "tune"],
        help="실행 모드 (기본값: final)",
    )
    args = parser.parse_args()

    main(mode=args.mode)
