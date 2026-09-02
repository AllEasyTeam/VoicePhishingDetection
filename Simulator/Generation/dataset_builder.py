# 완성된 dataset 만드는 파일.
import random
import numpy as np
import pandas as pd
from typing import Union, Dict, Optional

from Simulator.Generation.normal_generator import generate_normal_event
from Simulator.Generation.phishing_generator import generate_phishing_event

def build_dataset(
    n: int, 
    phishing_rate: float, 
    config, 
    sophistication: Union[str, Dict[str, str]] = "중간", 
    subgroup_ratio_key: str = "B",
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Algorithm 1 전체 의사 코드를 1:1로 구현한 데이터셋 생성 함수
    
    Require:
    - n: 생성할 사건 수 (int)
    - phishing_rate: 클래스 불균형 비율 (float, 0~1)
    - config: Generation/config.py 설정 모듈
    - sophistication: 값-옵션 선택 ('짧게'/'중간'/'길게' 또는 feature별 dict)
    - subgroup_ratio_key: 하위집단 지인:기관 비율 키 ('A'~'E')
    - random_state: 재현성을 위한 난수 시드값
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    # 1: dataset <- 빈 리스트
    dataset = []

    # 보이스피싱 유형(대출사기형, 기관사칭형, 지인사칭형, 기타) 비율 정규화 (config.TYPE_RATIO 기준)
    # total_weight이 1이 아닐 때, norm_weights 계산하여 비율 유지하면서 정규화하기 위한 과정.
    valid_types = {k: v for k, v in config.TYPE_RATIO.items() if v is not None}
    type_names = list(valid_types.keys())
    type_weights = list(valid_types.values())
    total_weight = sum(type_weights)
    norm_weights = [w / total_weight for w in type_weights]

    # 하위집단(지인 / 기관_개인 / 기관_기업) 비중 설정 (config.관계유형_비중_후보 기준)
    subgroup_dict = config.관계유형_비중_후보[subgroup_ratio_key]
    subgroup_names = list(subgroup_dict.keys())
    subgroup_weights = list(subgroup_dict.values())

    # 2: for i in range(n):
    for i in range(n):
        # 3: 이번 사건이 피싱/정상인지 확률적 결정 (클래스 불균형 비율보다 작아야 함.)
        # is_phishing이 phishing_rate보다 작으면 True(이번 사건은 피싱), 아니면 False(이번 사건은 정상) -> 매 사건마다 독립적으로 계산.
        is_phishing = random.random() < phishing_rate

        # 4: 이번 사건이 피싱 사건인 경우
        if is_phishing:
            # 5: type <- 피싱 유형 4종 중 확률적 선택 (config.TYPE_RATIO 기준)
            p_type = random.choices(type_names, weights=norm_weights)[0]
            # 6: sophistication <- 값-옵션 선택 (외부 인자 반영)
            # 7: generate_phishing_event(피싱 유형, sophistication, config) 호출 -> 피싱 event 생성
            event = generate_phishing_event(
                p_type=p_type,
                sophistication=sophistication, 
                config=config
            )
        # 8: 이번 사건이 정상 사건인 경우
        else:
            # 9: subgroup <- 하위집단 확률적 선택 (지인, 기관_개인, 기관_기업)
            subgroup = random.choices(subgroup_names, weights=subgroup_weights)[0]
            
            # 10: generate_normal_event(하위집단, config, sophistication) 호출 -> 정상 event 생성
            event = generate_normal_event(
                subgroup=subgroup, 
                config=config, 
                sophistication=sophistication
            )

        # 11: event는 dict. is_phishing이 True(=피싱 사건), False(=정상 사건) 여부에 따라 "is_phishing" key, value 새로 추가.
        event["is_phishing"] = 1 if is_phishing else 0

        # 12: event는 dict. is_phishing 값에 따라서, "incident_type" key, value 추가. (피싱 사건이면 피싱 유형) 
        if is_phishing:
            event["incident_type"] = p_type
        else:
            event["incident_type"] = "정상"

        # 13: dataset에 event 추가.
        dataset.append(event)

    # 14: return dataset (DataFrame 변환 및 셔플링-dataset 생성에서 순서 의존성 차단 목적)
    df = pd.DataFrame(dataset)
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

#소량 생성 검증용 테스트 코드. 추후 삭제
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 단독 실행 시 상위 패키지 경로 탐색 보정
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
    from Simulator.Generation import config
    
    # 1. 20건 생성 (검증을 위해 피싱 비율을 임의로 상향 설정)
    sample_size = 20
    df_sample = build_dataset(
        n=sample_size, 
        phishing_rate=0.3, 
        config=config, 
        sophistication="중간", 
        subgroup_ratio_key="B",
        random_state=42
    )

    # ------------------------------------------------------------
    # 2. 기본 생성 현황 확인
    # ------------------------------------------------------------
    print("=" * 80)
    print(f"📊 [합성 데이터 20건 생성 결과] (총 {len(df_sample)}건)")
    print("=" * 80)
    print(f"- 피싱 라벨 분포:\n{df_sample['is_phishing'].value_counts().to_dict()}")
    print(f"- 사건 유형 분포:\n{df_sample['incident_type'].value_counts().to_dict()}")
    print("\n" + "-" * 80)
    print("📋 [주요 Feature 샘플 미리보기 (상위 10건)]")
    print("-" * 80)
    
    preview_cols = [
        "phone_number", "number_type", "has_prior_history", "repeat_gap", 
        "sms_to_call", "sms_to_call_gap_min", "is_url_in_msg", "has_appinstall_link", 
        "is_sequential_callers", "is_phishing", "incident_type"
    ]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df_sample[preview_cols].head(10).to_string())

    # ------------------------------------------------------------
    # 3. 스키마 결측치 및 비즈니스 규칙 정합성 검증
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🔍 [스키마 결측 규칙 무결성 검증]")
    print("=" * 80)

    # 규칙 1: 구조적 선행 게이트 검증
    rule1_prior = (df_sample[df_sample["has_prior_history"] == 0]["repeat_gap"].isna()).all()
    rule1_sms = (df_sample[df_sample["sms_to_call"] == 0]["sms_to_call_gap_min"].isna()).all()
    rule1_num = (df_sample[df_sample["is_num_in_msg"] == 0]["inner_num_differs"].isna()).all()
    rule1_url = (df_sample[df_sample["is_url_in_msg"] == 0][["is_reliable_url", "has_appinstall_link"]].isna()).all().all()

    # 규칙 3: 항상 관측되어야 하는 컬럼 검증
    always_observed_cols = [
        "phone_number", "number_type", "call_time", "call_type", 
        "hour_bucket", "first_contact_type", "in_contacts", "is_global"
    ]
    rule3_check = (df_sample[always_observed_cols].isna().sum().sum() == 0)

    # 규칙 4: Track A 전용 컬럼 (전부 NaN이어야 함)
    rule4_check = (df_sample[["is_carrier_altered", "using_duration", "unique_callees"]].isna().all()).all()

    # 특수 규칙: 순차복수사칭은 NaN 없이 0 또는 1이어야 함
    special_rule_check = (df_sample["is_sequential_callers"].isna().sum() == 0) and (df_sample["is_sequential_callers"].isin([0, 1]).all())

    print(f"1. [규칙 1] 선행 조건 미충족 시 NaN 처리:")
    print(f"   - 과거이력 0건 -> repeat_gap NaN: {'✅ 통과' if rule1_prior else '❌ 실패'}")
    print(f"   - 연계 0 -> sms_to_call_gap_min NaN: {'✅ 통과' if rule1_sms else '❌ 실패'}")
    print(f"   - 문자 내 번호 0 -> inner_num_differs NaN: {'✅ 통과' if rule1_num else '❌ 실패'}")
    print(f"   - URL 0 -> 도메인/앱설치 링크 NaN: {'✅ 통과' if rule1_url else '❌ 실패'}")
    
    print(f"2. [규칙 3] 항상 관측 컬럼 결측치 0건 유지: {'✅ 통과' if rule3_check else '❌ 실패'}")
    print(f"3. [규칙 4] Track A 데이터 접근 불가(전부 NaN): {'✅ 통과' if rule4_check else '❌ 실패'}")
    print(f"4. [특수규칙] 순차복수사칭 확정적 0/1 채움 (NaN 없음): {'✅ 통과' if special_rule_check else '❌ 실패'}")
    print("=" * 80)