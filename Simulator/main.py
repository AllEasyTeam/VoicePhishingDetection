# Simulator 코드 작성을 위한 main.py
# 역할: 전체 파이프라인의 진입점. mode에 따라 Generation과 Detection을 순서대로 호출.

from Simulator.Generation import config
from Simulator.Generation.dataset_builder import build_dataset
from Simulator.Detection.train_eval import split_data, train_model, evaluate, prepare_categorical
from Simulator.Detection.sensitivity_analysis import run_sensitivity
from Simulator.schema import Track
from Simulator.schema_utils import get_feature_columns


# 최종 모드 실행 파라미터. "이번 최종 실행을 어떻게 돌릴지"에 대한 파이프라인 설정값이라 main.py에 지역 상수로 둠.
N = 10000                  # 생성할 사건 수
SUBGROUP_RATIO_KEY = "B"   # 정상 이벤트 하위집단(지인/기관) 비중 키 (민감도 분석 진행 후 확정)
GEN_SEED = 42              # build_dataset()용 random_state

# Track별 접근 가능 feature 시나리오 (누적 구조: 통신사가 가장 넓은 범위에 접근 가능).
# 최종 dataset 1개를 그대로 두고, 시나리오별로 사용하는 컬럼만 달라짐.
TRACK_SCENARIOS = {
   # "A": [Track.CARRIER, Track.DEVICE, Track.DEVICE_STRUCTURAL],  # 통신사: A+B+C 전부 접근 가능 (현재로써는 접근 가능한 데이터 X)
    "B": [Track.DEVICE],                                          # 단말: B만 접근 가능
    "C": [Track.DEVICE, Track.DEVICE_STRUCTURAL],                 # 단말+구조적신호: B+C 접근 가능
}


# 민감도분석 대상 feature 목록. run_sensitivity()의 param_name으로 그대로 넘어감.
# config.py의 SOPH_* 상수와 일치해야 함(= normal_generator.py/phishing_generator.py의
# _get_soph() 조회 key). 하드코딩 문자열 대신 상수를 써서, 오타 시 여기서 바로 NameError로 드러나게 함.
SENSITIVITY_PARAMS = [
    config.SOPH_NUMBER_TYPE_BAND,
    config.SOPH_FIRST_CONTACT,
    config.SOPH_REPEAT_GAP,          # 정상 전용 (피싱 쪽엔 sophistication 분기 없음)
    config.SOPH_NUM_IN_MSG,
    config.SOPH_INNER_NUM_DIFFERS,
    config.SOPH_MSG_OFFICIAL_MATCH,
    config.SOPH_SMS_TO_CALL_GAP,     # 정상 전용
    config.SOPH_URL_RATE,            # 정상 전용
    config.SOPH_APP_INSTALL,
    config.SOPH_SEQUENTIAL_CALLERS,
    config.SOPH_SMS_TO_CALL,         # 피싱 전용
]


def run_final():
    """최종 모드: 확정한 config 값으로 데이터셋 1개 생성 -> split_data() 3분할
    -> Track 시나리오(A/B/C)별로 feature만 다르게 골라 train_model()/evaluate() 반복."""
    df = build_dataset(
        n=N,
        phishing_rate=config.CLASS_IMBALANCE,
        subgroup_ratio_key=SUBGROUP_RATIO_KEY,
        random_state=GEN_SEED,
        config=config,
    )
    df = prepare_categorical(df)  # train/val/test로 나뉘기 전에 category dtype 한 번만 확정

    train_set, val_set, test_set = split_data(df) # train/validation/test 3분할

    results = {}  # {시나리오명("A"/"B"/"C"): evaluate() 결과}
    for scenario_name, tracks in TRACK_SCENARIOS.items():
        # 시나리오(A,B,C) 개수만큼 총 3번 반복. (Track별로 다른 모델 생성 및 평가 진행)
        cols = get_feature_columns(track_filter=tracks)
        model = train_model(train_set, val_set, feature_cols=cols)
        results[scenario_name] = evaluate(model, test_set, feature_cols=cols)  # test set으로 최종 평가
    return results


def run_sensitivity_mode():
    """민감도분석 모드: config.py 스윕 파라미터를 하나씩 바꿔가며 run_sensitivity() 반복 호출 -> 비교표 출력."""
    result = {}  # {param_name: run_sensitivity()의 반환값(summary 리스트)} 형태로 feature별 결과를 보관

    for param_name in SENSITIVITY_PARAMS:
        result[param_name] = run_sensitivity(
            param_name=param_name,
            candidate_values=["low", "mid", "high"],
            fixed_config=config,
            phishing_rate=config.CLASS_IMBALANCE,
            n=N,
            subgroup_ratio_key=SUBGROUP_RATIO_KEY,
        )

    # RELATION_TYPE_RATIO(정상 이벤트 하위집단 지인:기관 비중, A~E) 스윕.
    # sophistication 기반이 아니라 build_dataset()의 subgroup_ratio_key 자체를 바꾸는 축이라 별도 호출.
    result["subgroup_ratio_key"] = run_sensitivity(
        param_name="subgroup_ratio_key",
        candidate_values=list(config.RELATION_TYPE_RATIO.keys()),  # ["A", "B", "C", "D", "E"]
        fixed_config=config,
        sweep_target="subgroup_ratio_key",
        phishing_rate=config.CLASS_IMBALANCE,
        n=N,
    )

    return result


def main(mode: str):
    if mode == "final":
        result = run_final()
    elif mode == "sensitivity":
        result = run_sensitivity_mode()
    else:
        raise ValueError(f"알 수 없는 mode: {mode}")

    print(result)
    return result


if __name__ == "__main__":
    main(mode="final")
