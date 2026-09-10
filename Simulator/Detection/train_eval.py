# 분할->학습->평가 모두 한 번에 처리하는 파일.
def prepare_categorical(df):
    # XGBoost 학습/평가용 dtype 준비: 문자열(범주형) 컬럼을 category dtype으로 변환.
    # train/val/test, K-Fold 등으로 나뉘기 "전" df 전체에 반드시 한 번만 호출해야 함.
    # 분할 후 조각마다 따로 astype("category")하면 조각끼리 카테고리 코드북이 달라질 수 있어서
    # (예: 특정 fold에 특정 범주가 우연히 하나도 없으면) XGBoost predict()에서
    # "category not in the training set" 에러가 남. 슬라이싱(.iloc[])은 카테고리 목록을
    # 새로 계산하지 않고 그대로 물려받으므로, 분할 전에 한 번만 하면 모든 조각이 동일한
    # 카테고리 코드북을 공유하게 됨.
    df = df.copy()
    categorical_cols = df.select_dtypes(include="object").columns
    df[categorical_cols] = df[categorical_cols].astype("category")
    return df


def split_data(df):
    # dataframe을 train, test set으로 분할하는 함수.
    # train: 70%  test: 30% 으로 분할.
    # dataset 자체를 분할하기 때문에, get_feature_columns() 등의 schema_utils 함수는 호출하지 않음.
    from sklearn.model_selection import train_test_split

    # train과 test set으로 분할.
    train, test = train_test_split(
        df,
        test_size = 0.3, # 30%를 test set으로 분할.
        random_state = 42, # 재현성 확보를 위해 random_state 고정.(값이 중요한 게 아님. 동일 값을 사용하는 게 중요.)
        stratify = df["is_phishing"]
    )

    return train, test

def pr_auc(y, proba):
    # Precision-Recall AUC.
    from sklearn.metrics import precision_recall_curve, auc

    p, r, _ = precision_recall_curve(y, proba)
    return float(auc(r, p))


def lift_at_top_k(y, proba, k=0.05):
    """
    상위 k% 표본 내 피싱 농축 배수(Lift) 계산
    - y: 실제 피싱 여부 라벨 (0 또는 1)
    - proba: 모델이 예측한 피싱 확률
    - k: 상위 표본 비율 (0.05=상위 5%, 0.1=상위 10%, 0.2=상위 20%)
    """
    import numpy as np

    y_arr = np.array(y)
    cutoff = max(int(np.ceil(len(y_arr) * k)), 1)

    # 예측 확률 기준 내림차순 정렬 후 상위 k% 인덱스 추출
    top_indices = np.argsort(proba)[::-1][:cutoff]

    # 상위 k% 내 피싱 적중률 / 전체 자연 피싱 발생률
    top_rate = np.mean(y_arr[top_indices])
    base_rate = np.mean(y_arr)

    lift = top_rate / base_rate if base_rate > 0 else 0.0
    return round(float(lift), 2)


def train_model(df, val=None, feature_cols=None):
    # 학습을 위한 함수. (model : xgboost)
    # val: 선택적 검증셋(조기종료용). 현재는 main.py/sensitivity_analysis.py 둘 다
    # val 없이 호출함(최종 모드도 train/test 2분할만 씀) -> 아래 val 분기는 지금 호출
    # 경로에서는 안 타지만, 나중에 조기종료가 다시 필요해지면 val을 넘기기만 하면 되도록 남겨둠.
    # feature_cols: 학습에 사용할 feature 컬럼 목록. Track 시나리오별로 다르게 넘어옴. 민감도 분석에서는 전체 feature 사용.
    import xgboost as xgb
    from Simulator.schema_utils import get_feature_columns

    # feature_cols 존재하는 경우: Track 시나리오가 넘어온 경우(Track A,B,C) -> 해당 feature만 사용
    # feature_cols 존재하지 않는 경우: 민감도 분석 진행 -> 전체 feature 모두 사용
    feature_cols = feature_cols or get_feature_columns()
    # 방어적 필터: feature_cols에 is_feature=False인 컬럼(phone_number, call_time 등)이
    # 실수로 섞여 들어와도 여기서 한 번 더 걸러냄.
    valid_cols = set(get_feature_columns())
    feature_cols = [c for c in feature_cols if c in valid_cols]
    X = df[feature_cols].copy()
    y = df["is_phishing"]

    # 실제 category dtype 변환은 prepare_categorical()이 분할 전에 이미 끝내둠(fold 간
    # 카테고리 코드북을 통일하기 위함). 여기 있는 건 그걸 거치지 않고 바로 호출된 경우를 위한
    # 방어용 fallback일 뿐 -> object 컬럼이 남아있을 때만 동작(보통은 이미 없어서 no-op).
    # 주의: 이 fallback은 dtype이 아예 안 맞는 경우만 막아주고, prepare_categorical() 없이
    # train_fold/val_fold를 따로 변환하면 fold 간 카테고리 코드북이 어긋나는 문제(원래 버그)는 못 막음.
    categorical_cols = X.select_dtypes(include="object").columns
    X[categorical_cols] = X[categorical_cols].astype("category")

    # 클래스 불균형(피싱 1% vs 정상 99%) 보정.
    # scale_pos_weight = 음성 개수 / 양성 개수 -> 양성(피싱) 오분류에 더 큰 패널티를 줌.
    # config.CLASS_IMBALANCE(고정값)가 아니라 실제 df에서 계산 -> Track/fold마다 표본이 달라져도 항상 정확한 비율 반영.
    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1

    model = xgb.XGBClassifier(
        n_estimators=300,       # 트리 개수. 아래 학습률을 낮춘 만큼 넉넉하게 잡고 조기종료/고정 반복으로 제어.
        max_depth=4,            # 얕은 트리. 양성 표본이 적어(전체 1%) 트리가 깊으면 소수 양성 사례에 과적합하기 쉬움.
        learning_rate=0.05,     # 낮은 학습률 -> 한 트리가 과도하게 영향력을 갖지 않도록 완만하게 학습.
        scale_pos_weight=scale_pos_weight,  # 클래스 불균형 보정(위에서 계산).
        eval_metric="aucpr",    # PR-AUC. 극단적 불균형에서 accuracy/plain AUC보다 양성 클래스 성능을 잘 반영.
        tree_method="hist",     # 히스토그램 기반 분할 탐색 + 카테고리 dtype 지원에 필요.
        enable_categorical=True,  # category dtype 컬럼(number_type 등)을 원-핫 없이 직접 학습.
        random_state=42,        # 재현성 고정(다른 seed들과 마찬가지로 값 자체보다 고정 여부가 중요).
    )

    if val is not None:
        # val이 넘어온 경우(현재 호출 경로에서는 안 씀): 검증셋으로 조기종료 ->
        # n_estimators=300까지 다 안 돌고 검증 성능이 20라운드 연속 개선 안 되면 멈춤
        # (과적합 방지, 학습 시간 단축).
        # train의 categorical_cols만 쓰지 않고 X_val 자체 object 컬럼을 변환 
        # (val에만 남은 object가 빠지지 않도록)
        X_val = val[feature_cols].copy()
        val_cat_cols = X_val.select_dtypes(include="object").columns
        X_val[val_cat_cols] = X_val[val_cat_cols].astype("category")
        y_val = val["is_phishing"]

        model.set_params(early_stopping_rounds=20)
        model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    else:
        # 현재 기본 경로(최종 모드/K-Fold 모두): val 없이 n_estimators 고정 학습.
        # max_depth=4/learning_rate=0.05처럼 이미 보수적인 하이퍼파라미터로
        # 과적합을 억제하고 있어서, 조기종료 없이도 감당 가능하다고 판단.
        model.fit(X, y)

    return model


def evaluate(model, df, feature_cols=None, threshold_df=None):
    # 평가를 위한 함수.
    # accuracy / precision / recall / F1 / confusion_matrix / feature importance 반환.
    # feature_cols: train_model()과 동일한 컬럼 목록을 넘겨야 함(Track 시나리오 일치 필요).
    # threshold_df: threshold 탐색용 데이터(보통 train). None이면 df에서 탐색(낙관 편향 가능).
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix,
        precision_recall_curve,
    )
    from Simulator.schema_utils import get_feature_columns

    feature_cols = feature_cols or get_feature_columns()
    # 방어적 필터: feature_cols에 is_feature=False인 컬럼이 섞여 들어와도 한 번 더 걸러냄.
    valid_cols = set(get_feature_columns())
    feature_cols = [c for c in feature_cols if c in valid_cols]

    # train_model()과 동일하게, 실제 category dtype 변환은 prepare_categorical()이 분할 전에
    # 이미 끝내둠. 여기 있는 건 그걸 거치지 않고 바로 호출된 경우를 위한 방어용 fallback일 뿐
    # (보통은 이미 category라 no-op) -> fold 간 카테고리 코드북 불일치 문제는 못 막으니
    # 반드시 prepare_categorical()을 먼저 거쳐야 함.
    #   -> fallback은 _prepare_xy()로 옮겨 eval/threshold_df 양쪽에 공통 적용.
    # 목적: model.predict()의 기본 threshold(0.5)는 극단적 클래스 불균형(피싱 1% vs 정상 99%)
    # 데이터에서는 최적이 아닐 수 있음(양성 확률이 0.5를 잘 못 넘어서 recall이 과도하게 낮게 나올 수 있음). 
    # 그래서 확률(proba)만 뽑아서, precision-recall curve 상에서 F1이 최대가 되는 threshold를 직접 탐색해 적용함.
    #   -> 탐색은 threshold_df(없으면 df), 적용은 평가셋 df. thresholds가 비면 0.5로 fallback.
    # 주의(한계): 지금은 별도의 검증셋이 없어서(run_final()도 train/test 2분할만 씀),
    # threshold를 "평가 대상 df 자기 자신"에서 찾는다
    #   -> threshold_df(보통 train)에서 탐색 후 평가셋에 고정 적용. main/sensitivity는 분리 호출.
    #   threshold_df=None이면 예전처럼 df에서 탐색해 낙관 편향 가능.
    # 진짜 편향 없는 평가를 원하면 threshold 탐색은 별도 검증셋에서, 최종 성능 측정은
    # test set에서 하도록 분리해야 함.
    #   -> train 탐색 / test·val 평가는 반영됨. 별도 val 3분할은 아직 없음.
    def _prepare_xy(frame):
        X_part = frame[feature_cols].copy()
        # prepare_categorical() 이후라면 보통 no-op. 직접 호출된 경우만 object→category.
        cat_cols = X_part.select_dtypes(include="object").columns
        X_part[cat_cols] = X_part[cat_cols].astype("category")
        return X_part, frame["is_phishing"]

    X, y = _prepare_xy(df)

    thr_frame = threshold_df if threshold_df is not None else df
    X_thr, y_thr = _prepare_xy(thr_frame)
    proba_thr = model.predict_proba(X_thr)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_thr, proba_thr)
    # precision_recall_curve는 precision/recall을 thresholds보다 1개 더 많이 반환함
    # (마지막 지점은 threshold 없이 recall=0 지점) -> 그 마지막 지점은 탐색에서 제외.
    if len(thresholds) == 0:
        # 양성 표본이 없거나 curve가 비면 기본값 0.5 (argmax 크래시 방지)
        # 민감도 K-Fold에서 fold마다 양성이 매우 적을 때 위험
        best_threshold = 0.5
    else:
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        best_threshold = float(thresholds[f1_scores.argmax()])

    proba = model.predict_proba(X)[:, 1]  # 평가셋 피싱(1) 확률
    pred = (proba >= best_threshold).astype(int)

    lift_5p = lift_at_top_k(y, proba, k=0.05)
    lift_10p = lift_at_top_k(y, proba, k=0.1)
    lift_20p = lift_at_top_k(y, proba, k=0.2)

    return {
        "threshold": best_threshold,  # 이번 평가에 실제로 적용된 threshold(참고/재현용)
        # --- fold 집계용 스칼라 지표 ---
        "accuracy": accuracy_score(y, pred),
        # zero_division=0: 한 클래스만 예측될 때 경고/에러 대신 0으로 처리
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred),
        "classification_report": classification_report(y, pred, zero_division=0),
        # feature명 → 중요도. XGBoost 기본(gain 기반) importance
        "feature_importance": dict(zip(feature_cols, model.feature_importances_)),
        # PR-AUC: precision-recall auc로 베이스라인 대비 압도적으로 높다는 것을 보임()
        "PR-AUC": pr_auc(y, proba),
        # Top-K : 상위층에서 흔들림 없이 잘 잡아주는지.
        "Lift@Top5%" : lift_5p,
        "Lift@Top10%" : lift_10p,
        "Lift@Top20%" : lift_20p,
    }
