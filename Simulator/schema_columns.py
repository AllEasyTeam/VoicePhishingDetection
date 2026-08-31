"""
schema_columns.py — 실제 21개 컬럼 데이터.

역할: schema.py의 ColumnSchema 틀을 이용해, 실제 컬럼 21개를 채워넣은 목록.    
      컬럼 추가/수정 시 이 파일만 건드리면 됨.

phone_number이 is_feature = False인 이유: 학습 시 전화번호를 통째로 외울 수 있기 때문. 
call_time이 is_feature = False인 이유: 학습 시 call_time으로 얻은 hour_bucket을 사용하기 때문. (중복 방지)
"""

from schema import ColumnSchema, ValueType, Track

SCHEMA: list[ColumnSchema] = [
    ColumnSchema("phone_number", "전화번호", ValueType.TEXT, Track.ID, is_feature=False), # is_feature = False -> 학습 시 제외.

    ColumnSchema("number_type", "발신번호 종류/대역(010/070 ...)", ValueType.CATEGORICAL, Track.DEVICE),

    ColumnSchema("call_time", "발신/수신 시간", ValueType.DATETIME, Track.DEVICE, is_feature=False), # is_feature = False -> 학습 시 제외.

    ColumnSchema("call_type", "발신/수신", ValueType.BINARY, Track.DEVICE),

    ColumnSchema("hour_bucket", "활동 시간대", ValueType.CATEGORICAL, Track.DEVICE),

    ColumnSchema("first_contact_type", "개시 채널(문자/통화)", ValueType.BINARY, Track.DEVICE),

    ColumnSchema("has_prior_history", "과거 통화 이력", ValueType.BINARY, Track.DEVICE),

    ColumnSchema("repeat_gap", "재연락 간격(통화 간의 간격, 분 단위)",
                 ValueType.CONTINUOUS_TIME, Track.DEVICE,
                 nullable=True,
                 depends_on="has_prior_history = 0이면 간격 성립 안 함(NaN)"),

    ColumnSchema("in_contacts", "번호 저장 여부", ValueType.BINARY, Track.DEVICE),

    ColumnSchema("is_num_in_msg", "문자 내 번호 포함 여부", ValueType.BINARY, Track.DEVICE_STRUCTURAL),

    ColumnSchema("inner_num_differs", "발신번호-문자내번호 불일치",
                 ValueType.BINARY, Track.DEVICE_STRUCTURAL,
                 nullable=True,
                 depends_on="is_num_in_msg=0이면 비교대상 없음(NaN)"),

    ColumnSchema("sms_to_call", "문자→통화 연계 여부", ValueType.BINARY, Track.DEVICE),

    ColumnSchema("sms_to_call_gap_min", "문자→통화 전환 간격(분 단위)",
                 ValueType.CONTINUOUS_TIME, Track.DEVICE,
                 nullable=True,
                 depends_on="sms_to_call=0이면 전환 자체가 없음(NaN)"),

    ColumnSchema("is_url_in_msg", "(문자) URL 포함 여부", ValueType.BINARY, Track.DEVICE_STRUCTURAL),

    ColumnSchema("is_reliable_url", "(문자) URL 도메인 종류(신뢰/비신뢰)",
                 ValueType.BINARY, Track.DEVICE_STRUCTURAL,
                 nullable=True,
                 depends_on="is_url_in_msg=0이면 도메인 자체가 없음(NaN)"),

    ColumnSchema("has_appinstall_link", "(문자) 앱설치 유도(.apk 확장자)",
                 ValueType.BINARY, Track.DEVICE_STRUCTURAL,
                 nullable=True,
                 depends_on="is_url_in_msg=0이면 .apk 여부 판정 불가(NaN)"),

    ColumnSchema("is_carrier_altered", "발신번호 변작 여부", ValueType.BINARY,
                 Track.CARRIER, nullable=True,
                 depends_on="통신사 실측자료 있는 사건만 값 존재, 대부분 NaN"),

    ColumnSchema("using_duration", "번호 사용 기간(일 단위)",
                 ValueType.CONTINUOUS_TIME, Track.CARRIER, nullable=True,
                 depends_on="통신사 실측자료 있는 사건만 값 존재, 대부분 NaN"),

    ColumnSchema("unique_callees", "발신 대상 규모(명수)",
                 ValueType.CONTINUOUS_COUNT, Track.CARRIER, nullable=True,
                 depends_on="통신사 실측자료 있는 사건만 값 존재, 대부분 NaN"),

    ColumnSchema("is_global", "국제번호 여부", ValueType.BINARY, Track.DEVICE),

    ColumnSchema("is_sequential_callers", "순차복수사칭 여부", ValueType.BINARY, Track.DEVICE),
]