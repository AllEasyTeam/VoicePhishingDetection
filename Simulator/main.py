# Simulator 코드 작성을 위한 main.py
# 역할: 전체 파이프라인의 진입점. mode에 따라 Generation과 Detection을 순서대로 호출.

from pathlib import Path
from Simulator.Generation import config
from Simulator.Generation.dataset_builder import build_dataset
from Simulator.Detection.train_eval import split_data, train_model, evaluate, prepare_categorical
from Simulator.Detection.derived_features import add_derived_features
from Simulator.Detection.sensitivity_analysis import run_sensitivity, run_stress_sensitivity
from Simulator.schema import Track
from Simulator.schema_utils import get_feature_columns

# 최종 dataset 저장 폴더: 실행 위치(cwd)와 무관하게 항상 프로젝트 루트 기준으로 고정.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATASET_DIR = _PROJECT_ROOT / "DataSet"


# 최종 모드 실행 파라미터. "이번 최종 실행을 어떻게 돌릴지"에 대한 파이프라인 설정값이라 main.py에 지역 상수로 둠.
N = 100000                  # 생성할 사건 수
SUBGROUP_RATIO_KEY = "D"   # 정상 이벤트 하위집단(지인/기관) 비중 키 (민감도 분석 결과 B->D로 확정)
GEN_SEED = 42              # build_dataset()용 random_state

# 최종 dataset 생성 시 각 feature에 적용할 sophistication. 민감도 분석(필요 시 K=10/stress 모드
# 재검증까지 거쳐) 결과로 확정한 값. build_dataset()의 sophistication 인자로 그대로 넘어감
# (get_soph()가 dict에서 SOPH_* key를 직접 조회 -> 12개 전부 채워져 있어 "__base__" 폴백은 안 씀).
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


def save_final_dataset(df, out_dir: Path = _DATASET_DIR) -> None:
    """생성 직후(카테고리 변환/분할 전)의 원본 dataset을 DataSet/ 폴더에 CSV+Parquet 둘 다로 저장.
    같은 이름으로 덮어씀(재실행 시 항상 최신 dataset만 유지, 버전별 파일 누적 안 함)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "final_dataset.csv", index=False, encoding="utf-8-sig")  # utf-8-sig: Excel에서 한글 깨짐 방지
    df.to_parquet(out_dir / "final_dataset.parquet", index=False)  # dtype(카테고리 등) 그대로 보존


def run_final():
    """최종 모드: 확정한 config 값으로 데이터셋 1개 생성 -> DataSet/ 폴더에 저장
    -> split_data() 2분할(train/test) -> Track 시나리오(B/C)별로 feature만 다르게 골라 train_model()/evaluate() 반복."""
    df = build_dataset(
        n=N,
        phishing_rate=config.CLASS_IMBALANCE,
        subgroup_ratio_key=SUBGROUP_RATIO_KEY,
        random_state=GEN_SEED,
        config=config,
        sophistication=FINAL_SOPHISTICATION,
    )
    df = add_derived_features(df)  # 5개 파생 feature 추가 (schema_columns.py에 등록된 컬럼을 채움)
    save_final_dataset(df)  # 파생 feature 포함, category 변환/분할 전 상태를 최종본으로 저장

    df = prepare_categorical(df)  # train/test로 나뉘기 전에 category dtype 한 번만 확정

    train_set, test_set = split_data(df) # train/test 2분할

    results = {}  # {시나리오명("B"/"C"): evaluate() 결과}
    for scenario_name, tracks in TRACK_SCENARIOS.items():
        # TRACK_SCENARIOS 개수만큼 반복. (Track별로 다른 모델 생성 및 평가 진행)
        cols = get_feature_columns(track_filter=tracks)
        model = train_model(train_set, feature_cols=cols)
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


def main(mode: str, param_name=None, k=None, scenario_key=None):
    # param_name/k: mode="each_sen"일 때만 사용. scenario_key/k: mode="stress"일 때만 사용.
    # 둘 다 직접 Python으로 호출할 때만 넘기는 선택 인자이고, CLI(-m each_sen / -m stress)로는 안 받음
    # -> 코드 상단의 EACH_SENSITIVITY_PARAM/EACH_SENSITIVITY_K, STRESS_SCENARIO_KEY/STRESS_K를
    # 바꿔서 확인할 feature(또는 시나리오)/K를 정함(안 넘기면 그 상수값을 그대로 씀).
    if mode == "final":
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
    else:
        raise ValueError(f"알 수 없는 mode: {mode}")

    print(result)
    return result


# 실행 방법 (프로젝트 루트에서):
#   python -X utf8 -m Simulator.main              -> 기본값(mode=final)
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
#   -m은 --mode의 짧은 별칭.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Voice Phishing Detection 시뮬레이터 파이프라인 실행")
    parser.add_argument(
        "-m", "--mode", default="final", choices=["final", "sen", "each_sen", "ratio", "stress"],
        help="실행 모드 (기본값: final)",
    )
    args = parser.parse_args()

    main(mode=args.mode)
