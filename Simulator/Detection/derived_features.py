# 파생 feature 생성 파일.
# 역할: Generation이 만든 원본 컬럼들로부터 Detection 쪽에서만 쓰는 파생 feature 13개를 전부 계산.
#
# 13개 중 4개(structural_phishing_score/is_sms_initiated_unreg/cold_contact/repeat_pressure_intensity)는
# ablation 검증 통과 후 schema_columns.py의 SCHEMA에 확정 등록됨(2026-09-19, Structure.md 참고).
# main.py::generate_final_dataset()이 이 함수 호출 후 SCHEMA 등록 컬럼만 필터링해서 최종 dataset에 반영.
# 나머지 9개는 검증 탈락(다중공선성 제거/permutation importance<=0)했거나 대기 중이라 여전히 미등록.
# 이 함수 자체는 향후 재검증(feature_ablation.py)을 위해 항상 13개 전부를 계산해서 반환함.
import numpy as np


def add_candidate_features(df):
    """13개 파생 feature(확정 4개 + 미확정 9개)를 df에 추가해서 반환. 원본 df는 변경하지 않음(copy 사용).
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

    # 6. 미등록 발신자의 URL 발송: 주소록 미저장 + URL 포함.
    df["unreg_sender_with_url"] = (
        (df["in_contacts"] == 0) & (df["is_url_in_msg"] == 1)
    ).astype(int)

    # 7. 미등록 발신자의 문자 개시: 첫 접촉이 문자(0) + 주소록 미저장.
    df["is_sms_initiated_unreg"] = (
        (df["first_contact_type"] == 0) & (df["in_contacts"] == 0)
    ).astype(int)

    # 8. 초단기 문자->통화 전환: sms_to_call_gap은 sms_to_call==0이면 NaN이라
    #    "NaN <= 5"가 자동으로 False가 되어 별도 처리 없이 게이트됨.
    df["is_rapid_s2c_contact"] = (
        (df["sms_to_call"] == 1) & (df["sms_to_call_gap"] <= 5)
    ).astype(int)

    # 9. 즉각적 재촉 전화: repeat_gap도 has_repeat_contact==0이면 NaN -> 동일하게 자동 게이트.
    df["is_zero_gap_repeat"] = (
        (df["has_repeat_contact"] == 1) & (df["repeat_gap"] <= 1)
    ).astype(int)

    # 10. 문자 미끼 보유 여부(정보 손실형 OR, 참고용).
    df["has_any_msg_bait"] = (
        (df["is_url_in_msg"] == 1) | (df["is_num_in_msg"] == 1)
    ).astype(int)

    # 11. 문자 구조적 위협 누적 지수: 5개 하위 징후 카운트 합산.
    #     inner_num_differs/has_appinstall_link는 게이트 조건(is_num_in_msg/is_url_in_msg==0)일 때
    #     NaN이라 "그 하위 조건 자체가 없었다"는 의미 -> fillna(0)으로 "위협 아님" 처리.
    url_unreliable = np.where(
        df["is_url_in_msg"] == 1, (df["is_reliable_url"] == 0).astype(int), 0
    )
    df["structural_phishing_score"] = (
        df["is_num_in_msg"].fillna(0)
        + df["inner_num_differs"].fillna(0)
        + df["is_url_in_msg"].fillna(0)
        + url_unreliable
        + df["has_appinstall_link"].fillna(0)
    )

    # 12. 고위험 번호 대역: 국제번호(is_global) 또는 070(인터넷전화) 대역.
    #     is_global은 number_type=="00X(국제)"로 이미 결정되는 값이라 사실상 "070 또는 is_global".
    df["is_high_risk_number_type"] = (
        (df["is_global"] == 1) | (df["number_type"].astype(str).str.contains("070", na=False))
    ).astype(int)

    # 13. 미등록 + 고위험 번호 대역 결합 (12번 재사용).
    df["suspicious_unreg_number_combo"] = (
        (df["in_contacts"] == 0) & (df["is_high_risk_number_type"] == 1)
    ).astype(int)

    return df
