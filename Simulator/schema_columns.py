"""
schema_columns.py — 실제 컬럼 데이터.

역할: schema.py의 ColumnSchema 틀을 이용해, 실제 컬럼(원본 23개 + 확정 파생 feature 4개)을 채워넣은 목록.
      컬럼 추가/수정 시 이 파일만 건드리면 됨.

Row 단위: 1 row = 1 사건(경험 전체 요약). 1개 번호가 아님.
  복수 번호·복수 문자/통화가 있어도 그 전체를 한 row로 요약.
  단일값이 필요한 컬럼은 "이 사건 첫 이벤트" 기준 (phone_number / first_contact_type 등).

phone_number이 is_feature = False인 이유: 학습 시 전화번호를 통째로 외울 수 있기 때문. 
call_time이 is_feature = False인 이유: 학습 시 call_time으로 얻은 hour_bucket을 사용하기 때문. (중복 방지)
"""

from Simulator.schema import ColumnSchema, ValueType, Track

SCHEMA: list[ColumnSchema] = [
    # 사건 대표번호(=최초 접촉 발신번호). 번호 단위 PK가 아님.
    ColumnSchema("phone_number", "사건 대표번호(최초 접촉 발신번호)", ValueType.TEXT, Track.ID, is_feature=False), # is_feature = False -> 학습 시 제외.

    ColumnSchema("number_type", "발신번호 종류/대역(010/070 ...)", ValueType.CATEGORICAL, Track.DEVICE),  # 대표번호 기준

    ColumnSchema("call_time", "발신/수신 시간", ValueType.DATETIME, Track.DEVICE, is_feature=False), # is_feature = False -> 학습 시 제외.  # 사건 첫 이벤트 시각

    # call_type=1(수신) 고정 가정이 실측 근거 없이 판별력만 갖는 것으로 판단되어 보류.
    # ColumnSchema("call_type", "발신/수신", ValueType.BINARY, Track.DEVICE),  # 사건 첫 이벤트 기준 (피해자 단말 관점)

    ColumnSchema("hour_bucket", "활동 시간대", ValueType.CATEGORICAL, Track.DEVICE, nullable=True,
                 depends_on="피싱 기타(ETC) 유형은 표본 부족(n=2)으로 NaN"),  # 사건 첫 이벤트 기준

    ColumnSchema("first_contact_type", "개시 채널(0:문자/1:통화)", ValueType.BINARY, Track.DEVICE),  # 사건 첫 이벤트 기준 (번호별 X)

    ColumnSchema("has_prior_history", "과거 통화 이력(사건 간: 이전 별도 접촉 여부)", ValueType.BINARY, Track.DEVICE),  # 사건 간(row 간) 개념. 독립 feature -> 하위 컬럼을 게이트하지 않음

    ColumnSchema("has_repeat_contact", "사건 내 반복 접촉 여부", ValueType.BINARY, Track.DEVICE),  # 사건 내(한 row 안) 개념. repeat_gap의 게이트 조건

    ColumnSchema("repeat_gap", "재연락 간격(통화 간의 간격, 분 단위)",
                 ValueType.CONTINUOUS_TIME, Track.DEVICE,
                 nullable=True,
                 depends_on="has_repeat_contact = 0이면 간격 성립 안 함(NaN)"),  # 이 사건 안 연락 간격 (같은 번호 한정 X). 사건 내 반복이 여러 번이어도 평균이 아니라 첫 번째 반복까지의 간격

    ColumnSchema("in_contacts", "번호 저장 여부", ValueType.BINARY, Track.DEVICE),  # 대표번호 기준

    ColumnSchema("is_num_in_msg", "문자 내 번호 포함 여부", ValueType.BINARY, Track.DEVICE_STRUCTURAL),  # 사건 내 그런 문자가 있었는지(존재 여부)

    ColumnSchema("inner_num_differs", "발신번호-문자내번호 불일치",
                 ValueType.BINARY, Track.DEVICE_STRUCTURAL,
                 nullable=True,
                 depends_on="is_num_in_msg=0이면 비교대상 없음(NaN)"),  # 사건 내 그런 불일치가 있었는지(존재 여부)

    ColumnSchema("msg_number_official_match", "문자 내 번호-대표번호 일치 여부",
                 ValueType.BINARY, Track.DEVICE_STRUCTURAL,
                 nullable=True,
                 depends_on="is_num_in_msg=0이면 비교대상 없음(NaN)"),  # 사건 내 그런 일치가 있었는지(존재 여부)

    ColumnSchema("sms_to_call", "문자→통화 연계 여부", ValueType.BINARY, Track.DEVICE),  # 이 사건 전체 요약

    ColumnSchema("sms_to_call_gap", "문자→통화 전환 간격(분 단위)",
                 ValueType.CONTINUOUS_TIME, Track.DEVICE,
                 nullable=True,
                 depends_on="sms_to_call=0이면 전환 자체가 없음(NaN)"),  # 사건 첫 이벤트(문자) -> 그 다음 통화 1건까지의 간격. 평균 아님

    ColumnSchema("is_url_in_msg", "(문자) URL 포함 여부", ValueType.BINARY, Track.DEVICE_STRUCTURAL),  # 사건 내 그런 문자가 있었는지(존재 여부)

    ColumnSchema("is_reliable_url", "(문자) URL 도메인 종류(신뢰/비신뢰)",
                 ValueType.BINARY, Track.DEVICE_STRUCTURAL,
                 nullable=True,
                 depends_on="is_url_in_msg=0이면 도메인 자체가 없음(NaN)"),  # 사건 내 그런 URL이 있었는지(존재 여부)

    ColumnSchema("has_appinstall_link", "(문자) 앱설치 유도(.apk 확장자)",
                 ValueType.BINARY, Track.DEVICE_STRUCTURAL,
                 nullable=True,
                 depends_on="is_url_in_msg=0이면 .apk 여부 판정 불가(NaN)"),  # 사건 내 그런 링크가 있었는지(존재 여부)

    ColumnSchema("is_carrier_altered", "발신번호 변작 여부", ValueType.BINARY,
                 Track.CARRIER, nullable=True,
                 depends_on="통신사 실측자료 있는 사건만 값 존재, 대부분 NaN"),

    ColumnSchema("number_cluster", "발신번호 유사성(클러스터링)", ValueType.BINARY,
                 Track.CARRIER, nullable=True,
                 depends_on="통신사 실측자료(다수 사건 비교) 있는 사건만 값 존재, 시뮬레이터에서는 항상 NaN"),

    ColumnSchema("using_duration", "번호 사용 기간(일 단위)",
                 ValueType.CONTINUOUS_TIME, Track.CARRIER, nullable=True,
                 depends_on="통신사 실측자료 있는 사건만 값 존재, 대부분 NaN"),

    ColumnSchema("unique_callees", "발신 대상 규모(명수)",
                 ValueType.CONTINUOUS_COUNT, Track.CARRIER, nullable=True,
                 depends_on="통신사 실측자료 있는 사건만 값 존재, 대부분 NaN"),

    ColumnSchema("is_global", "국제번호 여부", ValueType.BINARY, Track.DEVICE),  # 대표번호 기준

    ColumnSchema("is_sequential_callers", "순차복수사칭 여부", ValueType.BINARY, Track.DEVICE),  # 이 사건 안 복수번호 릴레이 여부

    # ── 파생 feature 4개 (ablation 검증 통과, 2026-09-19 확정. Structure.md 참고) ──
    # Detection/derived_features.py::add_candidate_features()가 원본 컬럼만으로 계산.
    # main.py::generate_final_dataset()이 add_candidate_features() 호출 후 이 SCHEMA에 등록된
    # 컬럼만 필터링해서 최종 dataset에 반영(나머지 9개 후보는 미등록 상태로 계속 제외됨).
    ColumnSchema("structural_phishing_score", "문자 구조적 위협 누적 지수(0~5점)",
                 ValueType.CONTINUOUS_COUNT, Track.DEVICE_STRUCTURAL),  # 하위 조건 미발생분은 0점 처리 -> 결측 없음
    ColumnSchema("is_sms_initiated_unreg", "미등록 발신자의 문자 개시 여부", ValueType.BINARY, Track.DEVICE),
    ColumnSchema("cold_contact", "완전 낯선 접촉 여부(미저장+이력 없음)", ValueType.BINARY, Track.DEVICE),
    ColumnSchema("repeat_pressure_intensity", "재연락 압박 강도(재연락 간격 기반 연속 점수)",
                 ValueType.CONTINUOUS_SCORE, Track.DEVICE),  # 재연락 없으면 0.0 처리 -> 결측 없음
]
# 나머지 파생 feature 9개(cross_channel_urgency_score/is_malicious_bait_sms/urgency_path/
# unreg_sender_with_url/is_rapid_s2c_contact/is_zero_gap_repeat/has_any_msg_bait/
# is_high_risk_number_type/suspicious_unreg_number_combo)는 ablation 검증에서 탈락(다중공선성
# 제거 또는 permutation importance<=0)했거나 아직 검증 대기 중이라 SCHEMA에 등록하지 않음
# (Detection/derived_features.py의 add_candidate_features() 참고).