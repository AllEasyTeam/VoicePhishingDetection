# is_reliable_url stress를 "Track C(21개 feature)로 스코프를 좁혀서" 다시 실행하는 스크립트.
# 원래 진행했던 테스트는 전체 25개 feature 기준이었고, 이 스크립트는 그것과 별개인
# "Track C 단독 버전" 결과 하나뿐임(전체 25개 버전을 다시 만들지 않음 — 그 결과는 Structure.md에
# 텍스트로만 남아있음). 아래 2개는 "왜 굳이 Track C로 좁혀서 다시 도는가"에 대한 이유 2가지임
# (서로 다른 결과물을 각각 만든다는 뜻이 아니라, 이 하나의 실행을 정당화하는 근거 2가지).
#
# 이유 1(permutation 재확인 필요): is_reliable_url stress 테스트는 gain importance만으로
#       "msg_number_official_match/has_appinstall_link/cold_contact/structural_phishing_score/
#       is_sequential_callers로 credit이 옮겨간다"고 해석했음. 그런데 gain은 학습 중 분할에
#       얼마나 자주 쓰였는지를 보는 지표라, 상관된 feature가 있으면 credit이 한쪽으로
#       쏠리는 착시가 생길 수 있음(sms_to_call/sms_to_call_gap 사례로 이미 확인됨).
#       그래서 "진짜 대체"인지 held-out 기준인 permutation importance로 다시 확인해야 함.
# 이유 2(Track C로 스코프를 좁혀야 하는 이유): 원래 stress 테스트는 "전체 모델"이 이 가정에
#       얼마나 취약한지를 봤지만, 원래 질문은 "Track C가 Track B를 앞서는 우위가 이 가정에
#       얼마나 취약한가"였음. 그래서 이유 1의 permutation 재확인도, K-Fold 성능 재산출도
#       전부 Track C(21개, `run_stress_sensitivity()`에 feature_cols로 전달)로만 진행함.
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
from Simulator.Detection.sensitivity_analysis import (
    _temporary_config_overrides, run_stress_sensitivity, summarize_fold_results,
)
from Simulator.schema import Track
from Simulator.schema_utils import get_feature_columns
from Simulator.main import N, SUBGROUP_RATIO_KEY, GEN_SEED

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
    # 1단계(run_stress_sensitivity)가 기본 하이퍼파라미터를 쓰는 것과 동일하게 2단계도 기본값 사용.
    # 튜닝된 하이퍼파라미터는 baseline 가정에 맞춰진 값이라, 그걸 그대로 쓰면 "feature 자체의
    # 구조적 취약성"과 "baseline에 과적합된 하이퍼파라미터라 가정이 바뀌면 더 흔들리는 효과"가
    # 섞여버림 -> 이 stress 테스트 계열(-m stress 포함) 전체가 기본 하이퍼파라미터로 통일.
    # 코드의 구조적으로 1단계와 2단계가 연결되지는 않지만, 동일한 질문에 대한 해석을 진행하기 위해서는 하이퍼파라미터 통일이 필요했음.
    model_type = "XGBoost"
    # 원래 질문("Track C 우위가 이 가정에 얼마나 취약한가")에 맞춰 Track C(단말+구조적신호) 21개로 스코프 고정.
    cols = get_feature_columns(track_filter=[Track.DEVICE, Track.DEVICE_STRUCTURAL])
    print(f"model_type={model_type}, Track C feature 수={len(cols)}")

    # 1단계) Track C만으로 K-Fold 성능(F1/PR-AUC ± SEM) 재산출 — 원래 stress 테스트(전체 25개 기준)와
    #    비교하기 위한 표. sensitivity_results/stress_is_reliable_url_trackC.json에도 저장됨.
    # 즉, "is_reliable_url 가정이 무너지면 Track C가 얼마나 흔들리는가"에 대한 "얼마나(성능이 얼마나 떨어지는가)"를 확인하는 단계.
    print("\n" + "=" * 70)
    print("[1] Track C K-Fold 성능 재산출 (run_stress_sensitivity, feature_cols=Track C)")
    print("=" * 70)
    stress_summary = run_stress_sensitivity(
        scenario_key="stress_is_reliable_url_trackC",
        level_overrides=LEVELS,
        fixed_config=config,
        description="is_reliable_url stress를 Track C(21개) feature로만 재검증",
        n=N,
        phishing_rate=config.CLASS_IMBALANCE,
        subgroup_ratio_key=SUBGROUP_RATIO_KEY,
        gen_seed=GEN_SEED,
        feature_cols=cols,
    )
    stress_perf = {}
    for entry in stress_summary:
        s = summarize_fold_results(entry["fold_results"])
        # confusion_matrix는 summarize_fold_results()가 스칼라가 아니라서 버리는 값이라,
        # 원본 fold_results(entry["fold_results"])에서 직접 꺼내 fold별 TN/FP/FN/TP 평균을 냄.
        # sklearn.metrics.confusion_matrix(y_true, y_pred) 관례: [0][0]=TN [0][1]=FP [1][0]=FN [1][1]=TP.
        cms = [fr["confusion_matrix"] for fr in entry["fold_results"]]
        tn = [int(cm[0][0]) for cm in cms]
        fp = [int(cm[0][1]) for cm in cms]
        fn = [int(cm[1][0]) for cm in cms]
        tp = [int(cm[1][1]) for cm in cms]
        cm_mean = {
            "TN": round(sum(tn) / len(tn), 2), "FP": round(sum(fp) / len(fp), 2),
            "FN": round(sum(fn) / len(fn), 2), "TP": round(sum(tp) / len(tp), 2),
        }
        stress_perf[entry["value"]] = {**s, "confusion_matrix_mean": cm_mean}
        print(f"  [{entry['value']}] F1={s['f1']['mean']:.4f}±{s['f1']['sem']:.4f}  "
              f"PR-AUC={s['PR-AUC']['mean']:.4f}±{s['PR-AUC']['sem']:.4f}  "
              f"recall={s['recall']['mean']:.4f}  FN(평균)={cm_mean['FN']}")

    # 2단계) 레벨별 단일 split 학습 + permutation importance
    # is_reliable_url과 대체 후보 feature들의 permutation importance가 레벨에 따라 어떻게 변화하는지 보는 단계.
    # 1단계에서는 fold별 성능 요약만 반환하고 개별 feature의 permutation importance를 계산/보관하지 않기 때문에,
    # 2단계에서 별도로 레벨마다 train_model() -> evaluate() 호출한 뒤 그 안에서 permutation_importance() 추가로 구함.
    # "is_reliable_url 가정이 무너지면 Track C가 얼마나 흔들리는가"에 대해 "왜(어떤 feature가 그 빈자리를 메꾸는가)"를 확인하는 단계.
    print("\n" + "=" * 70)
    print("[2] 단일 split 학습 + permutation importance (gain 재해석용)")
    print("=" * 70)
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

        model = train_model(train_set, feature_cols=cols)
        m = evaluate(model, test_set, feature_cols=cols, threshold_df=train_set)

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
        json.dump({
            "model_type": model_type,
            "track_c_feature_count": len(cols),
            "levels": LEVELS,
            "track_c_kfold_performance": stress_perf,  # sensitivity_results/stress_is_reliable_url_trackC.json에도 저장됨
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")
    print(f"저장(K-Fold 원본): sensitivity_results/stress_is_reliable_url_trackC.json")
    return results


if __name__ == "__main__":
    main()
