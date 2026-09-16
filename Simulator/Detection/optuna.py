import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import average_precision_score
import warnings

# 불필요한 경고문구 숨김
warnings.filterwarnings('ignore')

def create_objective(X_train, y_train, X_val, y_val, model_type='LightGBM'):
    """
    Optuna 목적함수를 생성하는 클로저(Closure) 함수
    미리 분할된 Train / Validation 셋을 입력받아 데이터 누수를 방지합니다.
    """
    # 양성 클래스(피싱) 가중치 기준점 계산
    base_scale = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        if model_type == 'LightGBM':
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',  # 트리 분기 자체는 logloss로 수행
                'n_estimators': 1000,        # 충분히 크게 주고 Early Stopping으로 제어
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', base_scale * 0.5, base_scale * 1.5),
                
                # 트리 복잡도 제어
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                
                # 샘플링 (일반화)
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                
                # 정규화
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            
            # LightGBM 학습 및 Early Stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='custom', # 내부 평가는 무시하고 콜백에서 제어
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            # 검증셋 예측 확률 추출
            preds = model.predict_proba(X_val)[:, 1]

        elif model_type == 'XGBoost':
            params = {
                'eval_metric': 'aucpr', # XGBoost는 PR-AUC를 기본 메트릭으로 지원
                'early_stopping_rounds': 50,
                'n_estimators': 1000,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', base_scale * 0.5, base_scale * 1.5),
                
                # 트리 복잡도 제어
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_child_weight': trial.suggest_float('min_child_weight', 5, 50),
                
                # 샘플링
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                
                # 정규화
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                
                'tree_method': 'hist',
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = xgb.XGBClassifier(**params)
            
            # XGBoost 학습 및 Early Stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            preds = model.predict_proba(X_val)[:, 1]

        # PR-AUC 계산 (sklearn의 average_precision_score 사용)
        pr_auc = average_precision_score(y_val, preds)
        
        return pr_auc

    return objective