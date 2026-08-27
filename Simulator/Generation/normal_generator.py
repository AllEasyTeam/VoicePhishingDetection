# 정상 이벤트 1건 생성을 위한 파일

import random
import numpy as np
from datetime import datetime, timedelta
from typing import Union, Dict

def _get_soph(soph: Union[str, Dict[str, str]], key: str) -> str:
    return soph.get(key, "중간") if isinstance(soph, dict) else soph

def _calculate_repeat_gap(subgroup: str, config, sophistication: Union[str, Dict[str, str]]) -> float:
    """config.REPEAT_CONTACT 세부 혼합분포를 적용하여 재연락 간격(분) 계산(임시)"""
    gap_soph = _get_soph(sophistication, "재연락_간격")
    
    if subgroup == "지인":
        cfg = config.REPEAT_CONTACT["지인"]
        theta = random.choices(cfg["theta_list"], weights=cfg["p_list"])[0]
        multiplier = cfg["배율"][gap_soph]
        return round(np.random.exponential(scale=theta * multiplier), 2)
    else:  # 기관 (기관_개인, 기관_기업)
        cfg = config.REPEAT_CONTACT["기관"]
        # 1단계: 재연락 발생 확률 검사
        prob_key = "중간" if gap_soph not in cfg["발생확률"] else gap_soph
        if random.random() < cfg["발생확률"][prob_key]:
            theta = random.choices(cfg["theta_list"], weights=cfg["p_list"])[0]
            multiplier = cfg["배율"][gap_soph]
            return round(np.random.exponential(scale=theta * multiplier), 2)
        else:
            return np.nan

def generate_normal_event(subgroup: str, config, sophistication: Union[str, Dict[str, str]] = "중간") -> dict:
    """
    정상 하위집단(지인/기관) 규칙에 따라 이벤트 1건 생성
    """
    # 1. 식별자 및 발신 대역 (Track B)
    if subgroup == "지인":
        phone_number = f"010{random.randint(10000000, 99999999)}"
    else:
        # 기관_개인 / 기관_기업: 화이트리스트 대표번호 또는 일반 유선번호
        if random.random() < 0.85:
            phone_number = random.choice(list(config.WHITELIST_SET))
        else:
            phone_number = f"02{random.randint(1000000, 9999999)}"
            
    # 번호 대역 표준화 분류
    number_type = config.classify_number_type(phone_number)
    is_global = 1 if number_type == "00X(국제)" else 0

    # 2. 통화 시간 및 기본 속성 (규칙 3: 항상 관측)
    base_time = datetime(2026, 1, 1) + timedelta(minutes=random.randint(0, 525600))
    call_time = base_time.strftime("%Y-%m-%d %H:%M:%S")
    hour_bucket = base_time.hour
    call_type = random.choice([0, 1])  # 0: 발신, 1: 수신
    first_contact_type = random.choice(["call", "sms"])

    # 3. 연락처 저장 및 과거 통화 이력 (config.F_GROUP)
    group_key = "지인" if subgroup == "지인" else "기관"
    in_contacts = 1 if random.random() < config.F_GROUP["연락처_저장확률"][group_key] else 0
    has_prior_history = 1 if random.random() < config.F_GROUP["과거통화이력_있음확률"][group_key] else 0
    
    # 4. 재연락 간격 (규칙 1: 과거 통화 이력 없으면 NaN)
    if has_prior_history == 1:
        repeat_gap = _calculate_repeat_gap(group_key, config, sophistication)
    else:
        repeat_gap = np.nan

    # 5. 문자 내 번호 포함 및 불일치 (Track C)
    num_in_msg_soph = _get_soph(sophistication, "문자내_번호_포함확률")
    is_num_in_msg = 1 if random.random() < config.B_GROUP["문자내_번호_포함확률"][num_in_msg_soph] else 0

    if is_num_in_msg == 1:
        match_soph = _get_soph(sophistication, "발신번호_문자내번호_일치율")
        match_rate = config.A_GROUP["발신번호_문자내번호_일치율"][match_soph]
        inner_num_differs = 0 if random.random() < match_rate else 1
    else:
        inner_num_differs = np.nan  # 규칙 1

    # 6. 문자 -> 통화 연계 및 간격 (Track B)
    sms_to_call = 1 if (first_contact_type == "sms" and random.random() < config.SMS_TO_CALL_확률) else 0
    if sms_to_call == 1:
        gap_soph = _get_soph(sophistication, "문자_통화_간격")
        theta = random.choices(config.SMS_TO_CALL_GAP["theta_list"], weights=config.SMS_TO_CALL_GAP["p_list"])[0]
        multiplier = config.SMS_TO_CALL_GAP["배율"][gap_soph]
        sms_to_call_gap_min = round(np.random.exponential(scale=theta * multiplier), 2)
    else:
        sms_to_call_gap_min = np.nan  # 규칙 1

    # 7. URL 및 앱 설치 유도 (Track C)
    url_soph = _get_soph(sophistication, "URL_존재확률")
    is_url_in_msg = 1 if random.random() < config.B_GROUP["URL_존재확률"][url_soph] else 0

    if is_url_in_msg == 1:
        is_reliable_url = 1  # 정상 메시지의 URL은 공식 도메인 (1.0)
        app_soph = _get_soph(sophistication, "앱설치_유도")
        has_appinstall_link = 1 if random.random() < config.A_GROUP["앱설치_유도"][app_soph] else 0
    else:
        is_reliable_url = np.nan    # 규칙 1
        has_appinstall_link = np.nan # 규칙 1

    # 8. 순차복수사칭 및 Track A 전용 결측치 (규칙 특수: 0 채움 / 규칙 4: NaN)
    seq_soph = _get_soph(sophistication, "순차복수사칭")
    is_sequential_callers = 1 if random.random() < config.A_GROUP["순차복수사칭"][seq_soph] else 0
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