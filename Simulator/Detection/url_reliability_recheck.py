# is_reliable_url stress 결과("다른 feature로 credit이 옮겨가서 부분 복구된다")를
# gain이 아니라 permutation importance로 재확인하는 스크립트.
# Track C(지금은 C 원본 17개+파생 4개=21개 feature) 기준.
#
# 배경: is_reliable_url stress 테스트는 gain importance만으로
#       "msg_number_official_match/has_appinstall_link/cold_contact/structural_phishing_score/
#       is_sequential_callers로 credit이 옮겨간다"고 해석했음. 그런데 gain은 학습 중 분할에
#       얼마나 자주 쓰였는지를 보는 지표라, 상관된 feature가 있으면 credit이 한쪽으로
#       쏠리는 착시가 생길 수 있음(sms_to_call/sms_to_call_gap 사례로 이미 확인됨).
#       그래서 "진짜 대체"인지 held-out 기준인 permutation importance로 다시 확인.
#
# 실행 (프로젝트 루트):
#   python -X utf8 -m Simulator.Detection.url_reliability_recheck

import json
from pathlib import Path

from sklearn.inspection import permutation_importance

from Simulator.Generation import config
from Simulator.Generation.dataset_builder import build_dataset
from Simulator.Detection.derived_features import add_candidate_features
from Simulator.Detection.train_eval import prepare_categorical, split_data, train_model, evaluate
from Simulator.Detection.sensitivity_analysis import _temporary_config_overrides
from Simulator.schema import Track
from Simulator.schema_utils import get_feature_columns
from Simulator.main import load_tuned_hyperparams, N, SUBGROUP_RATIO_KEY, GEN_SEED

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUT_DIR = _PROJECT_ROOT / "url_reliability_recheck_results"

# is_reliable_url stress 테스트와 동일한 5단계.
# NORMAL_IS_RELIABLE_URL(정상 URL이 신뢰 도메인일 확률)은 낮추고,
# PHISHING_IS_RELIABLE_URL(피싱 URL이 신뢰 도메인처럼 보일 확률)은 올려서 "혼선" 축 하나로 스윕.
LEVELS = {
    "baseline": {"NORMAL_IS_RELIABLE_URL": 1.0, "PHISHING_IS_RELIABLE_URL": 0.0},
    "mild":     {"NORMAL_IS_RELIABLE_URL": 0.95, "PHISHING_IS_RELIABLE_URL": 0.05},
    "moderate": {"NORMAL_IS_RELIABLE_URL": 0.85, "PHISHING_IS_RELIABLE_URL": 0.15},
    "strong":   {"NORMAL_IS_RELIABLE_URL": 0.70, "PHISHING_IS_RELIABLE_URL": 0.30},
    "extreme":  {"NORMAL_IS_RELIABLE_URL": 0.50, "PHISHING_IS_RELIABLE_URL": 0.50},
}

# gain에서 credit이 옮겨간 것처럼 보였던 is_reliable_url + 5개 "대체 후보".
FOCUS_FEATURES = [
    "is_reliable_url",
    "msg_number_official_match",
    "has_appinstall_link",
    "cold_contact",
    "structural_phishing_score",
    "is_sequential_callers",
]

PERM_N_REPEATS = 10


def main():
    model_type, hyperparams = load_tuned_hyperparams()
    model_type = model_type or "XGBoost"
    # 원래 질문("Track C 우위가 이 가정에 얼마나 취약한가")에 맞춰 Track C(단말+구조적신호) 21개로 스코프 고정.
    cols = get_feature_columns(track_filter=[Track.DEVICE, Track.DEVICE_STRUCTURAL])
    print(f"model_type={model_type}, Track C feature 수={len(cols)}")

    results = {}
    for level_name, overrides in LEVELS.items():
        with _temporary_config_overrides(config, overrides):
            df = build_dataset(
                N, phishing_rate=config.CLASS_IMBALANCE, subgroup_ratio_key=SUBGROUP_RATIO_KEY,
                random_state=GEN_SEED, config=config, sophistication="mid",
            )
        # override는 build_dataset 동안에만 필요. 학습/평가는 생성된 df만 사용.
        df = add_candidate_features(df)
        df = prepare_categorical(df)
        train_set, test_set = split_data(df)

        model = train_model(train_set, feature_cols=cols, hyperparams=hyperparams, model_type=model_type)
        m = evaluate(model, test_set, feature_cols=cols, threshold_df=train_set, pr_auc_method="average_precision")

        X_test, y_test = test_set[cols], test_set["is_phishing"]
        perm = permutation_importance(
            model, X_test, y_test, scoring="average_precision",
            n_repeats=PERM_N_REPEATS, random_state=42,
        )
        perm_mean = {c: float(v) for c, v in zip(cols, perm.importances_mean)}
        gain = {c: float(v) for c, v in zip(cols, model.feature_importances_)}

        results[level_name] = {
            "f1": m["f1"], "PR-AUC": m["PR-AUC"],
            "permutation_importance": {f: round(perm_mean.get(f, 0.0), 6) for f in FOCUS_FEATURES},
            "gain_importance": {f: round(gain.get(f, 0.0), 6) for f in FOCUS_FEATURES},
        }

        print(f"\n[{level_name}] F1={m['f1']:.4f} PR-AUC={m['PR-AUC']:.4f}")
        for f in FOCUS_FEATURES:
            print(f"  {f:28s} perm={perm_mean.get(f, 0.0): .5f}  gain={gain.get(f, 0.0):.4f}")

    print("\n" + "=" * 70)
    print("permutation importance 요약 (baseline -> extreme)")
    print(f"{'feature':28s}" + "".join(f"{lvl:>12s}" for lvl in LEVELS))
    for f in FOCUS_FEATURES:
        row = "".join(f"{results[lvl]['permutation_importance'][f]:12.5f}" for lvl in LEVELS)
        print(f"{f:28s}{row}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "url_reliability_recheck.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model_type": model_type, "track_c_feature_count": len(cols), "levels": LEVELS,
                    "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")
    return results


if __name__ == "__main__":
    main()
