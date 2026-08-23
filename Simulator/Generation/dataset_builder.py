# 완성된 dataset 만드는 파일.
import random
import pandas as pd
from typing import Union, Dict

from Simulator.Generation.normal_generator import generate_normal_event
from Simulator.Generation.phishing_generator import generate_phishing_event

def build_dataset(
    n: int, 
    phishing_rate: float, 
    config, 
    sophistication: Union[str, Dict[str, str]] = "중간", 
    subgroup_ratio_key: str = "B"
) -> pd.DataFrame:
    """
    Algorithm 1 전체 의사 코드를 1:1로 구현한 데이터셋 생성 함수
    
    Require:
    - n: 생성할 사건 수 (int)
    - phishing_rate: 클래스 불균형 비율 (float, 0~1)
    - config: Generation/config.py 설정 모듈
    - sophistication: 값-옵션 선택 ('짧게'/'중간'/'길게' 또는 feature별 dict)
    - subgroup_ratio_key: 하위집단 지인:기관 비율 키 ('A'~'E')
    """
    # 1: dataset <- 빈 리스트
    dataset = []

    # 피싱 유형 비율 정규화 (config.TYPE_RATIO 기준)
    valid_types = {k: v for k, v in config.TYPE_RATIO.items() if v is not None}
    type_names = list(valid_types.keys())
    type_weights = list(valid_types.values())
    total_weight = sum(type_weights)
    norm_weights = [w / total_weight for w in type_weights]

    # 지인:기관 비중 기준값 (config.지인기관_비중_후보)
    p_acquaintance = config.지인기관_비중_후보[subgroup_ratio_key]

    # 2: for i in range(n):
    for i in range(n):
        # 3: is_phishing <- random() < phishing_rate
        is_phishing = random.random() < phishing_rate

        # 4: if is_phishing:
        if is_phishing:
            # 5: type <- 유형 4종 중 확률적 선택 (config.TYPE_RATIO 기준)
            p_type = random.choices(type_names, weights=norm_weights)[0]
            # 6: sophistication <- 값-옵션 선택 (외부 인자 반영)
            # 7: event <- generate_phishing_event(type, sophistication, config)
            event = generate_phishing_event(
                p_type=p_type,
                sophistication=sophistication, 
                config=config
            )
        # 8: else:
        else:
            # 9: subgroup <- 지인/기관 확률적 선택 (config.지인기관_비중 기준)
            subgroup = "지인" if random.random() < p_acquaintance else "기관"
            # 10: event <- generate_normal_event(subgroup, config)
            event = generate_normal_event(
                subgroup=subgroup, 
                config=config, 
                sophistication=sophistication
            )
        # 11: end if

        # 12: event.피싱여부 <- is_phishing ? 1 : 0
        event["is_phishing"] = 1 if is_phishing else 0

        # 13: if is_phishing:
        # 14:     event.유형라벨 <- type
        # 15: end if
        if is_phishing:
            event["incident_type"] = p_type
        else:
            event["incident_type"] = "정상"

        # 16: dataset.append(event)
        dataset.append(event)
    # 17: end for

    # 18: return dataset (DataFrame 변환 및 셔플링)
    df = pd.DataFrame(dataset)
    return df.sample(frac=1.0).reset_index(drop=True)
