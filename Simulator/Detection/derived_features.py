# 파생 feature 생성 파일.
# 역할: Generation이 만든 원본 컬럼들로부터 Detection 쪽에서만 쓰는 파생 feature를 계산.
#      config.py는 import하지 않음(Generation-Detection Separation 원칙 유지 -> 생성 시
#      쓴 파라미터가 아니라 dataset의 "출력 컬럼"만으로 계산).
# 호출 시점: build_dataset() 직후, prepare_categorical() 이전(schema_columns.py의 SCHEMA에
#          등록된 5개 컬럼을 실제로 채워 넣는 단계라 get_feature_columns() 전에 반드시 필요).
import numpy as np


def add_derived_features(df):
    """확정된 5개 파생 feature를 df에 추가해서 반환. 원본 df는 변경하지 않음(copy 사용).
    urgency_path가 is_malicious_bait_sms 값을 그대로 쓰므로, 계산 순서(3번 -> 5번)가 중요함."""
    df = df.copy()

    # 1. 다채널 전이 긴급도: 문자->통화로 연계됐고, 그 간격이 짧을수록 큰 값(연계 없으면 0).
    df["cross_channel_urgency_score"] = np.where(
        df["sms_to_call"] == 1,
        1.0 / (df["sms_to_call_gap"].fillna(9999.0) + 1.0),
        0.0,
    )

    # 2. 재연락 압박 강도: 사건 내 재연락이 있었고, 그 간격이 짧을수록 큰 값(재연락 없으면 0).
    df["repeat_pressure_intensity"] = np.where(
        df["has_repeat_contact"] == 1,
        1.0 / (df["repeat_gap"].fillna(9999.0) + 1.0),
        0.0,
    )

    # 3. 악성 미끼 문자 식별: URL이 있고, 그 URL이 비신뢰 도메인이거나 앱 설치를 유도.
    #    is_reliable_url/has_appinstall_link는 is_url_in_msg==0이면 NaN이라, NaN==0 / NaN==1이
    #    둘 다 False로 평가되어 별도 처리 없이 자동으로 0(정상) 처리됨.
    df["is_malicious_bait_sms"] = (
        (df["is_url_in_msg"] == 1)
        & ((df["is_reliable_url"] == 0) | (df["has_appinstall_link"] == 1))
    ).astype(int)

    # 4. 완전 낯선 접촉: 주소록 미저장(in_contacts=0) + 과거 접촉 이력 없음(has_prior_history=0).
    df["cold_contact"] = (
        (df["in_contacts"] == 0) & (df["has_prior_history"] == 0)
    ).astype(int)

    # 5. 유인 경로: 악성 미끼 문자(3번) 이후 통화로 연계. 반드시 3번 계산 이후에 실행.
    df["urgency_path"] = (
        (df["sms_to_call"] == 1) & (df["is_malicious_bait_sms"] == 1)
    ).astype(int)

    return df
