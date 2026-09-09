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
        X_val = val[feature_cols].copy()
        X_val[categorical_cols] = X_val[categorical_cols].astype("category")
        y_val = val["is_phishing"]

        model.set_params(early_stopping_rounds=20)
        model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    else:
        # 현재 기본 경로(최종 모드/K-Fold 모두): val 없이 n_estimators 고정 학습.
        # max_depth=4/learning_rate=0.05처럼 이미 보수적인 하이퍼파라미터로
        # 과적합을 억제하고 있어서, 조기종료 없이도 감당 가능하다고 판단.
        model.fit(X, y)

    return model


def evaluate(model, df, feature_cols=None):
    # 평가를 위한 함수.
    # accuracy / precision / recall / F1 / confusion_matrix / feature importance 반환.
    # feature_cols: train_model()과 동일한 컬럼 목록을 넘겨야 함(Track 시나리오 일치 필요).
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )
    from Simulator.schema_utils import get_feature_columns

    feature_cols = feature_cols or get_feature_columns()
    # 방어적 필터: feature_cols에 is_feature=False인 컬럼이 섞여 들어와도 한 번 더 걸러냄.
    valid_cols = set(get_feature_columns())
    feature_cols = [c for c in feature_cols if c in valid_cols]
    X = df[feature_cols].copy()
    y = df["is_phishing"]

    # train_model()과 동일하게, 실제 category dtype 변환은 prepare_categorical()이 분할 전에
    # 이미 끝내둠. 여기 있는 건 그걸 거치지 않고 바로 호출된 경우를 위한 방어용 fallback일 뿐
    # (보통은 이미 category라 no-op) -> fold 간 카테고리 코드북 불일치 문제는 못 막으니
    # 반드시 prepare_categorical()을 먼저 거쳐야 함.
    categorical_cols = X.select_dtypes(include="object").columns
    X[categorical_cols] = X[categorical_cols].astype("category")

    pred = model.predict(X)

    return {
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
    }