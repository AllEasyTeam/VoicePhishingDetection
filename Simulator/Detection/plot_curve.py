import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc

from Simulator.Detection.train_eval import split_data, prepare_categorical
from Simulator.main import get_or_generate_final_dataset, load_tuned_hyperparams
from Simulator.schema_utils import get_feature_columns

from pathlib import Path
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from Simulator.Detection.train_eval import split_data, prepare_categorical, train_model
from Simulator.main import get_or_generate_final_dataset, load_tuned_hyperparams
from Simulator.schema_utils import get_feature_columns

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUT_DIR = _PROJECT_ROOT / "shap_results"

def plot_xgboost_learning_curve():
    # 1. 데이터 로드 및 전처리
    df = get_or_generate_final_dataset()
    df = prepare_categorical(df)
    train_set, test_set = split_data(df)

    # 2. 최적 하이퍼파라미터 로드 및 피처 선택 (Track C: 전체 21개 피처 기준)
    _, tuned_params = load_tuned_hyperparams()
    feature_cols = get_feature_columns()  # Track C 피처

    X_train = train_set[feature_cols]
    y_train = train_set["is_phishing"]
    X_test = test_set[feature_cols]
    y_test = test_set["is_phishing"]

    # 3. 과적합 관찰을 위해 n_estimators를 넉넉히 주고(예: 150) 학습 히스토리 기록
    params = {
        **tuned_params,
        "n_estimators": 150,  # 최적값인 101 라운드 전후의 과적합 추이를 확인하기 위함
        "eval_metric": ["logloss", "aucpr"],  # 손실함수 및 1% 불균형 핵심 지표(PR-AUC)
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": 42,
        "n_jobs": -1,
    }

    model = xgb.XGBClassifier(**params)

    # eval_set에 train과 test를 모두 전달하여 라운드별 점수 기록
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    results = model.evals_result()
    epochs = len(results["validation_0"]["logloss"])
    x_axis = range(1, epochs + 1)

    # 4. 시각화 (Logloss & PR-AUC 2개 서브플롯)
    plt.figure(figsize=(14, 5))

    # (1) Log Loss 곡선 (Loss 기반 과적합 확인)
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, results["validation_0"]["logloss"], label="Train Logloss", color="blue")
    plt.plot(x_axis, results["validation_1"]["logloss"], label="Test Logloss", color="red", linestyle="--")
    plt.axvline(x=tuned_params.get("n_estimators", 101), color="green", linestyle=":", label=f"Best Iteration ({tuned_params.get('n_estimators', 101)})")
    plt.title("XGBoost Log Loss Curve (Train vs Test)")
    plt.xlabel("Boosting Rounds (n_estimators)")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # (2) PR-AUC 곡선 (불균형 탐지 성능 기반 과적합 확인)
    plt.subplot(1, 2, 2)
    plt.plot(x_axis, results["validation_0"]["aucpr"], label="Train PR-AUC", color="blue")
    plt.plot(x_axis, results["validation_1"]["aucpr"], label="Test PR-AUC", color="red", linestyle="--")
    plt.axvline(x=tuned_params.get("n_estimators", 101), color="green", linestyle=":", label=f"Best Iteration ({tuned_params.get('n_estimators', 101)})")
    plt.title("XGBoost PR-AUC Curve (Train vs Test)")
    plt.xlabel("Boosting Rounds (n_estimators)")
    plt.ylabel("PR-AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("xgboost_learning_curve.png", dpi=300)
    plt.show()
    print("학습 곡선 그래프 생성 완료: xgboost_learning_curve.png")


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUT_DIR = _PROJECT_ROOT / "shap_results"


def run_shap_analysis():
    # 1. 데이터 로드 및 전처리 (동일 Split 기준)
    df = get_or_generate_final_dataset()
    df = prepare_categorical(df)
    train_set, test_set = split_data(df)

    # 2. 최적 모델 및 하이퍼파라미터 로드
    feature_cols = get_feature_columns()  # Track C (21개 피처)
    model_type, hyperparams = load_tuned_hyperparams()
    model_type = model_type or "XGBoost"

    # 모델 학습
    model = train_model(train_set, feature_cols=feature_cols, hyperparams=hyperparams, model_type=model_type)

    X_test = test_set[feature_cols]
    y_test = test_set["is_phishing"]

    # 3. SHAP TreeExplainer 계산
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 그래프 1: 전역 요약 플롯 (Beeswarm Summary Plot)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    plt.title(f"[{model_type}] Track C Global SHAP Summary Plot", fontsize=14, pad=15)
    summary_path = _OUT_DIR / "shap_summary_plot.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"전역 SHAP 플롯 저장 완료: {summary_path}")

    # -------------------------------------------------------------
    # 그래프 2: 실제 피싱 사건 1건에 대한 알림 근거 플롯 (Waterfall Plot)
    # -------------------------------------------------------------
    # Test 셋에서 실제 피싱인 고위험 샘플 인덱스 1개 선택
    phishing_indices = test_set.index[test_set["is_phishing"] == 1].tolist()
    sample_loc = test_set.index.get_loc(phishing_indices[0])

    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values[sample_loc], max_display=8, show=False)
    plt.title(f"User Alert Explanation (Sample #{sample_loc})", fontsize=13, pad=12)
    waterfall_path = _OUT_DIR / "shap_local_waterfall_plot.png"
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=300)
    plt.close()
    print(f"개별 알림용 Waterfall 플롯 저장 완료: {waterfall_path}")


if __name__ == "__main__":
    run_shap_analysis()
if __name__ == "__main__":
    plot_xgboost_learning_curve()