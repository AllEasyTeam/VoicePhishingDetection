# 피싱 이벤트 1건 생성을 위한 파일

import random
import numpy as np
from datetime import datetime, timedelta
from typing import Union, Dict

def _get_soph(soph: Union[str, Dict[str, str]], key: str) -> str:
    return soph.get(key, "중간") if isinstance(soph, dict) else soph

def generate_phishing_event(p_type: str, sophistication: Union[str, Dict[str, str]], config) -> dict:
    """
    피싱 유형(대출/기관/지인사칭/협박)에 따른 차등 θ값을 적용하여 이벤트 1건 생성
    """
    # 1. 식별자 및 발신 번호 생성
    if p_type == "지인사칭형":
        # 지인 사칭은 주로 010 개인 번호 형태
        number_type = "010"
        phone_number = f"010{random.randint(10000000, 99999999)}"
        is_global = 0
    elif p_type == "기관사칭형":
        # 기관 사칭은 02(서울) 또는 070, 국제발신
        number_type = random.choices(["02", "070", "010"], weights=[0.4, 0.4, 0.2])[0]
        phone_number = f"02{random.randint(1000000, 9999999)}" if number_type == "02" else f"070{random.randint(10000000, 99999999)}"
        is_global = 1 if random.random() < 0.25 else 0
    else:  # 대출사기형, 협박형
        number_type = random.choices(["070", "010", "1588_변작"], weights=[0.6, 0.3, 0.1])[0]
        phone_number = f"070{random.randint(10000000, 99999999)}"
        is_global = 1 if random.random() < 0.15 else 0

    # 2. 통화/문자 시간 및 기본 속성 (규칙 3: 항상 관측)
    base_time = datetime(2026, 1, 1) + timedelta(minutes=random.randint(0, 525600))
    call_time = base_time.strftime("%Y-%m-%d %H:%M:%S")
    hour_bucket = base_time.hour
    call_type = 1  # 수신 (피해자 단말 관점)
    
    first_contact_type = "sms" if p_type in ["대출사기형", "지인사칭형"] else random.choice(["call", "sms"])

    # 3. 저장 및 과거 이력
    in_contacts = 0
    has_prior_history = 0
    repeat_gap = np.nan  # 규칙 1: 과거 이력 없으므로 NaN

    # 4. 문자 내 번호 및 불일치 (Track C)
    if first_contact_type == "sms":
        is_num_in_msg = 1 if random.random() < 0.85 else 0
    else:
        is_num_in_msg = 1 if random.random() < 0.20 else 0

    if is_num_in_msg == 1:
        inner_num_differs = 1 if random.random() < 0.85 else 0
    else:
        inner_num_differs = np.nan

    # 5. 문자 -> 통화 연계 간격 (Track B)
    sms_to_call = 1 if first_contact_type == "sms" else 0
    if sms_to_call == 1:
        # 피싱은 압박/유도로 전환 간격(theta)이 매우 짧음 (평균 2~5분)
        theta_gap = 2.0 if p_type in ["기관사칭형", "협박형"] else 5.0
        sms_to_call_gap_min = round(np.random.exponential(scale=theta_gap), 2)
    else:
        sms_to_call_gap_min = np.nan

    # 6. URL 및 앱 설치 유도 (.apk)
    if first_contact_type == "sms":
        is_url_in_msg = 1 if random.random() < 0.80 else 0
    else:
        is_url_in_msg = 1 if random.random() < 0.15 else 0

    if is_url_in_msg == 1:
        is_reliable_url = 0  # 피싱 도메인
        # 대출/기관 사칭은 악성 앱 설치 유도 확률이 매우 높음
        apk_prob = 0.85 if p_type in ["대출사기형", "기관사칭형"] else 0.40
        has_appinstall_link = 1 if random.random() < apk_prob else 0
    else:
        is_reliable_url = np.nan
        has_appinstall_link = np.nan

    # 7. 순차복수사칭 (특수 규칙: 확정적 0 또는 1)
    if p_type == "기관사칭형":
        is_sequential_callers = 1 if random.random() < 0.45 else 0
    else:
        is_sequential_callers = 0

    # 8. Track A 전용 feature (규칙 4: 통신사 데이터 부재로 NaN)
    is_carrier_altered = np.nan
    using_duration = np.nan
    unique_callees = np.nan

    return {
        "phone_number": phone_number,
        "number_type": number_type,
        "call_time": call_time,
        "call_type": call_type,
        "hour_bucket": hour_bucket,
        "first_contact_type": first_contact_type,
        "has_prior_history": has_prior_history,
        "repeat_gap": repeat_gap,
        "in_contacts": in_contacts,
        "is_num_in_msg": is_num_in_msg,
        "inner_num_differs": inner_num_differs,
        "sms_to_call": sms_to_call,
        "sms_to_call_gap_min": sms_to_call_gap_min,
        "is_url_in_msg": is_url_in_msg,
        "is_reliable_url": is_reliable_url,
        "has_appinstall_link": has_appinstall_link,
        "is_carrier_altered": is_carrier_altered,
        "using_duration": using_duration,
        "unique_callees": unique_callees,
        "is_global": is_global,
        "is_sequential_callers": is_sequential_callers
    }