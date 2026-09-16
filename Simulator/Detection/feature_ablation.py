# 파생 feature 13개의 선정 파이프라인 (baseline vs full 비교 -> 다중공선성 점검(+데이터 누수
# 플래그) -> Embedded method(SHAP 참고용) + Permutation Importance 기반 가지치기 -> 최종 재학습).
#
# 4단계 구성:
#   1) baseline(원본 feature만) vs full(원본+파생 13개) 성능 비교. 둘 다 train_set으로 학습,
#      고정된 test_set으로만 평가.
#   2) train_set 기준으로만 다중공선성(상관계수) 점검 -> 위배되는 쌍은 타깃과 상관 더 높은
#      쪽만 남김. 데이터 누수(타깃과의 상관계수)는 "플래그만 하고 자동 제거하지 않음"
#      (누수처럼 보여도 실제로는 핵심 유효 feature일 수 있어서, 최종 판단은 3단계의
#      객관적 지표(permutation importance)에 맡김).
#   3) 2번을 통과한 feature로 학습(Embedded method) -> SHAP는 참고용으로 계산해서 리포트에
#      남기되, 실제 가지치기 기준은 Permutation Importance <= 0(섞었을 때 성능이 안
#      떨어지거나 오히려 좋아짐 = 기여가 없다는 뜻)로 객관화. permutation importance는
#      "학습에 안 쓰인 데이터"에서 계산해야 의미가 있어서, train_set을 다시 한번
#      train_sub(학습용)/val_sub(permutation 검증용)로 나눔 -> test_set은 이 단계에서
#      전혀 안 건드림.
#   4) 3번까지 살아남은 최종 feature set으로 train_set 전체를 다시 학습 -> test_set에서
#      최종 평가.
#
# run_sensitivity()/run_stress_sensitivity()가 K-Fold로 "생성 파라미터"를 스윕하는 것과 달리,
# 이건 dataset 1개를 고정 분할해서 "feature 선정" 자체를 진단하는 용도라 K-Fold를 쓰지 않음.
import json
from pathlib import Path
from typing import Optional

import numpy as np
import shap
import shap.explainers._tree as _shap_tree_mod
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from Simulator.Generation.dataset_builder import build_dataset
from Simulator.schema_utils import get_feature_columns
from Simulator.Detection.train_eval import train_model, evaluate, prepare_categorical, split_data
from Simulator.Detection.derived_features import add_candidate_features

# 결과 저장 폴더: 실행 위치(cwd)와 무관하게 항상 프로젝트 루트 기준으로 고정.
# sensitivity_results/(생성 파라미터 민감도)와는 성격이 달라서(feature 선정 검증) 별도 폴더로 분리.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS_DIR = _PROJECT_ROOT / "feature_selection_results"

# add_candidate_features()가 만드는 13개 컬럼명. get_feature_columns()에는 아직 없으므로
# 여기서 직접 나열해서 feature_cols 구성에 씀.
CANDIDATE_FEATURES = [
    "cross_channel_urgency_score",
    "repeat_pressure_intensity",
    "is_malicious_bait_sms",
    "cold_contact",
    "urgency_path",
    "unreg_sender_with_url",
    "is_sms_initiated_unreg",
    "is_rapid_s2c_contact",
    "is_zero_gap_repeat",
    "has_any_msg_bait",
    "structural_phishing_score",
    "is_high_risk_number_type",
    "suspicious_unreg_number_combo",
]

DEFAULT_MULTICOLLINEARITY_THRESHOLD = 0.8  # |상관계수|가 이 값을 넘는 파생 feature 쌍 중 하나 제거
DEFAULT_LEAKAGE_THRESHOLD = 0.95            # 타깃(is_phishing)과의 |상관계수|가 이 값을 넘으면 누수 "의심"(플래그만, 자동 제거 안 함)
DEFAULT_VAL_SIZE = 0.2                      # train_set 중 permutation importance 검증용으로 떼어둘 비율
DEFAULT_PERM_N_REPEATS = 10                 # permutation importance 반복 횟수(셔플 때마다 우연 변동 있어서 여러 번 평균)


def _select_by_correlation(train_df, candidate_features, target_col="is_phishing", threshold=0.8):
    """train set 기준 다중공선성 점검. 후보 feature끼리의 상관계수가 threshold를 넘는 쌍이 있으면,
    타깃과의 상관계수가 더 높은(=더 예측력 있는) 쪽만 남기고 나머지를 제거(greedy)."""
    corr_with_target = train_df[candidate_features + [target_col]].corr()[target_col].drop(target_col)
    ordered = corr_with_target.abs().sort_values(ascending=False).index.tolist()

    feature_corr = train_df[candidate_features].corr()

    kept, dropped = [], []
    for feat in ordered:
        conflict = None
        for kept_feat in kept:
            r = feature_corr.loc[feat, kept_feat]
            if abs(r) > threshold:
                conflict = (kept_feat, r)
                break
        if conflict is not None:
            dropped.append({
                "feature": feat,
                "correlated_with": conflict[0],
                "correlation": round(float(conflict[1]), 4),
            })
        else:
            kept.append(feat)
    return kept, dropped


def _check_leakage(train_df, candidate_features, target_col="is_phishing", threshold=0.95):
    """train set 기준 데이터 누수 "의심" 점검. 파생 feature가 타깃과 지나치게(threshold 이상)
    상관되면 플래그만 함(자동 제거 안 함) -> 누수처럼 보여도 실제로는 핵심 유효 feature일 수
    있으므로, 최종 판단은 3단계의 permutation importance에 맡김. 결과 리포트에 남겨서
    사람이 직접 확인할 수 있게만 함."""
    corr = train_df[candidate_features + [target_col]].corr()[target_col].drop(target_col)
    return [
        {"feature": f, "correlation_with_target": round(float(corr[f]), 4)}
        for f in candidate_features if abs(corr[f]) > threshold
    ]


_SHAP_PATCHED = False


def _patch_shap_xgboost_base_score_bug():
    """shap(0.49.1)이 xgboost(3.x)가 base_score를 "[4.9E-1]" 같은 대괄호 벡터 문자열로
    직렬화하는 걸 못 읽어서 TreeExplainer 생성 시 ValueError가 나는 알려진 호환성 버그를
    우회. decode_ubjson_buffer()가 반환한 값에서 base_score의 대괄호만 벗겨서 되돌려줌
    (다른 값/로직은 전혀 건드리지 않음). 모듈 전체에 한 번만 적용."""
    global _SHAP_PATCHED
    if _SHAP_PATCHED:
        return
    _orig_decode = _shap_tree_mod.decode_ubjson_buffer

    def _patched_decode(fd):
        jmodel = _orig_decode(fd)
        bs = jmodel["learner"]["learner_model_param"]["base_score"]
        if isinstance(bs, str) and bs.startswith("["):
            jmodel["learner"]["learner_model_param"]["base_score"] = bs.strip("[]")
        return jmodel

    _shap_tree_mod.decode_ubjson_buffer = _patched_decode
    _SHAP_PATCHED = True


def _compute_shap_importance(model, X):
    """train_sub에서 |SHAP value| 평균(feature별 기여도)을 계산. 참고/리포트용이며,
    실제 가지치기 기준(permutation importance)과는 별개."""
    _patch_shap_xgboost_base_score_bug()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        # 이진분류에서 클래스별 리스트로 나오는 shap 버전 대응 -> 양성(1) 클래스 값 사용.
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    mean_abs = np.abs(shap_values).mean(axis=0)
    return {col: float(v) for col, v in zip(X.columns, mean_abs)}


def _compute_permutation_importance(model, X_val, y_val, n_repeats=DEFAULT_PERM_N_REPEATS, random_state=42):
    """val_sub(학습에 안 쓰인 데이터)에서 Permutation Importance 계산. scoring="average_precision"
    (~PR-AUC)을 씀 -> threshold 선택에 의존하지 않는 지표라, evaluate()의 F1-threshold 탐색
    로직을 별도로 끌어올 필요 없이 objective하게 비교 가능."""
    result = permutation_importance(
        model, X_val, y_val, scoring="average_precision",
        n_repeats=n_repeats, random_state=random_state,
    )
    return {col: float(v) for col, v in zip(X_val.columns, result.importances_mean)}


def run_feature_selection_pipeline(
    fixed_config,                       # config.py 모듈. build_dataset()에 그대로 전달.
    n: int = 100000,
    phishing_rate: Optional[float] = None,
    subgroup_ratio_key: str = "D",
    sophistication="mid",               # feature 선정이 목적이라 sophistication은 고정(기본 mid).
    gen_seed: int = 42,
    multicollinearity_threshold: float = DEFAULT_MULTICOLLINEARITY_THRESHOLD,
    leakage_threshold: float = DEFAULT_LEAKAGE_THRESHOLD,
    val_size: float = DEFAULT_VAL_SIZE,
    out_dir: Path = _RESULTS_DIR,
) -> dict:
    """4단계 feature 선정 파이프라인 실행."""
    df = build_dataset(
        n, phishing_rate=phishing_rate, subgroup_ratio_key=subgroup_ratio_key,
        random_state=gen_seed, config=fixed_config, sophistication=sophistication,
    )
    df = add_candidate_features(df)
    df = prepare_categorical(df)
    train_set, test_set = split_data(df)  # stratify=is_phishing (train_eval.split_data() 참고)

    # permutation importance(3단계)용으로 train_set을 한 번 더 나눔. train_sub은 3단계
    # 진단용 모델 학습에만 쓰고, val_sub은 그 모델이 한 번도 보지 못한 데이터로 permutation
    # importance를 계산하는 데만 씀. test_set은 여기서 전혀 관여하지 않음.
    train_sub, val_sub = train_test_split(
        train_set, test_size=val_size, random_state=42, stratify=train_set["is_phishing"]
    )

    baseline_cols = get_feature_columns()
    full_cols = baseline_cols + CANDIDATE_FEATURES

    # --- 1단계: baseline vs full (train_set 전체로 학습, test_set으로만 평가) ---
    baseline_model = train_model(train_set, feature_cols=baseline_cols)
    baseline_metrics = evaluate(baseline_model, test_set, feature_cols=baseline_cols, threshold_df=train_set)

    full_model = train_model(train_set, feature_cols=full_cols)
    full_metrics = evaluate(full_model, test_set, feature_cols=full_cols, threshold_df=train_set)

    # --- 2단계: 다중공선성 점검(제거) + 데이터 누수 점검(플래그만, train_set 전체 기준) ---
    kept_after_corr, dropped_corr = _select_by_correlation(
        train_set, CANDIDATE_FEATURES, threshold=multicollinearity_threshold
    )
    leakage_flags = _check_leakage(train_set, CANDIDATE_FEATURES, threshold=leakage_threshold)
    step2_survivors = kept_after_corr  # 누수는 플래그만 하고 여기서 빼지 않음(3단계 객관적 지표로 최종 판단)

    # --- 3단계: Embedded method(train_sub) + SHAP(참고용) + Permutation Importance(val_sub, 실제 기준) ---
    step3_cols = baseline_cols + step2_survivors
    step3_model = train_model(train_sub, feature_cols=step3_cols)

    X_train_sub_step3 = train_sub[step3_cols]
    shap_importance = _compute_shap_importance(step3_model, X_train_sub_step3)

    X_val_step3 = val_sub[step3_cols]
    y_val_step3 = val_sub["is_phishing"]
    perm_importance = _compute_permutation_importance(step3_model, X_val_step3, y_val_step3)

    final_candidates = [f for f in step2_survivors if perm_importance.get(f, 0.0) > 0]
    pruned_by_permutation = [f for f in step2_survivors if perm_importance.get(f, 0.0) <= 0]

    # --- 4단계: 최종 feature set으로 train_set 전체를 다시 학습 -> test_set 최종 평가 ---
    final_cols = baseline_cols + final_candidates
    final_model = train_model(train_set, feature_cols=final_cols)
    final_metrics = evaluate(final_model, test_set, feature_cols=final_cols, threshold_df=train_set)

    def _metrics_summary(m):
        return {k: m[k] for k in ("threshold", "accuracy", "precision", "recall", "f1", "PR-AUC",
                                   "Lift@Top5%", "Lift@Top10%", "Lift@Top20%")}

    payload = {
        "kind": "feature_selection_pipeline",
        "n": n,
        "step1_baseline_vs_full": {
            "baseline_feature_count": len(baseline_cols),
            "full_feature_count": len(full_cols),
            "baseline": _metrics_summary(baseline_metrics),
            "full": _metrics_summary(full_metrics),
        },
        "step2_multicollinearity": {
            "threshold": multicollinearity_threshold,
            "dropped_pairs": dropped_corr,
            "survivors": kept_after_corr,
        },
        "step2_leakage_check": {
            "threshold": leakage_threshold,
            "flagged_features(참고용, 자동 제거 안 함)": leakage_flags,
        },
        "step3_embedded_importance": {
            "candidates_evaluated": step2_survivors,
            "shap_importance(참고용)": {k: round(v, 6) for k, v in shap_importance.items()},
            "permutation_importance(실제 가지치기 기준, val_sub 기준)": {
                k: round(v, 6) for k, v in perm_importance.items()
            },
            "pruned_by_permutation(<=0)": pruned_by_permutation,
            "final_candidate_features": final_candidates,
        },
        "step4_final_retrain": {
            "final_feature_set": final_cols,
            "test_metrics": _metrics_summary(final_metrics),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "feature_selection_pipeline.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload
