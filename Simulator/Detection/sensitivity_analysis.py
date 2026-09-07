# 민감도 분석 진행을 위한 파일
from typing import List, Optional
from sklearn.model_selection import StratifiedKFold


def run_sensitivity(
    param_name: str,                    # 스윕 대상 feature 이름(config.py 섹션명). ex) "URL_존재확률"
    candidate_values: List[str],        # param_name에 하나씩 대입해볼 sophistication 값 목록. ex) [LOW, MID, HIGH]
    fixed_config,                       # config.py 모듈. param_name 외 나머지 값은 전부 여기서 그대로(고정) 가져다 씀
    base_sophistication: Optional[str] = None,  # param_name을 제외한 다른 feature들에 공통 적용할 sophistication. 스윕 대상 외 변수를 통제하기 위함
    n: int = 10000,                     # 후보값마다 생성할 사건(row) 개수. 후보값끼리 동일해야 공정 비교 가능
    phishing_rate: Optional[float] = None,  # build_dataset()에 넘길 클래스 불균형 비율(보통 config.CLASS_IMBALANCE)
    subgroup_ratio_key: str = "B",      # 정상 이벤트 하위집단(지인/기관) 비중 키. 스윕과 무관한 값이라 고정
    gen_seed: int = 42,                 # build_dataset()용 random_state. 후보값 사이에 동일해야 "파라미터만 바뀌었다"고 말할 수 있음
    fold_seed: int = 7,                 # StratifiedKFold용 random_state. 마찬가지로 후보값 사이에 고정 필요
    k: int = 5,                         # K-Fold 분할 개수
):
    # 한 feature에 대한 민감도 분석 진행을 위한 함수.
    # K-Fold 학습/평가 -> 결과 비교표 반환.
    pass
