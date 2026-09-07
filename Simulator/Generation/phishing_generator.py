# 피싱 이벤트 1건 생성을 위한 파일

import random
import numpy as np
from datetime import datetime, timedelta
from typing import Union, Dict

_AREA_CODES = [
    "031", "032", "033", "041", "042", "043", "044",
    "051", "052", "053", "054", "055", "061", "062", "063", "064",
]

_SOPH_ALIAS = {
    "짧게": "low", "중간": "mid", "길게": "high",
    "low": "low", "mid": "mid", "high": "high",
}


def _get_soph(soph: Union[str, Dict[str, str]], key: str) -> str:
    raw = soph.get(key, "mid") if isinstance(soph, dict) else soph
    return _SOPH_ALIAS.get(raw, "mid")


def _digits(n: int) -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(n))


def _phone_from_category(category: str) -> str:
    """classify_number_type 카테고리 + 이미지 허용 형식."""
    if category == "010":
        return "010" + _digits(8)
    if category == "특번":
        return random.choice(["15", "16", "18"]) + _digits(6)
    if category == "02(유선)":
        return "02" + str(random.randint(1, 9)) + _digits(random.choice([6, 7]))
    if category == "070(인터넷전화)":
        return "070" + _digits(random.randint(5, 8))
    if category == "00X(국제)":
        return "00" + _digits(random.randint(8, 12))
    # 기타
    if random.random() < 0.5:
        return random.choice(_AREA_CODES) + str(random.randint(1, 9)) + _digits(random.choice([6, 7]))
    prefix = random.choice(["060", "080", "030", "050"])
    max_total = 12 if prefix in ("030", "050") else 11
    return prefix + _digits(random.randint(5, max_total - 3))


def _pick_from_call_band(config, soph: str) -> str:
    band = config.PHISHING_CALL_NUMBER_TYPE
    cats = list(band.keys())
    return random.choices(cats, weights=[band[c][soph] for c in cats])[0]


def _generate_phishing_phone(
    p_type: str,
    first_contact_type: str,
    sophistication: Union[str, Dict[str, str]],
    config,
) -> str:
    """PHISHING_SMS_NUMBER_TYPE / PHISHING_CALL_NUMBER_TYPE 비중으로 발신번호 생성."""
    soph = _get_soph(sophistication, "발신번호_종류_대역")
    sms_band = config.PHISHING_SMS_NUMBER_TYPE.get(p_type)

    if first_contact_type == "sms" and isinstance(sms_band, dict):
        # acquaintance: {"010": 0.99, "기타": 0.01} 처럼 flat
        # 그 외: PHISHING_CALL_NUMBER_TYPE 차용 (카테고리 → {low/mid/high})
        sample = next(iter(sms_band.values()), None)
        if isinstance(sample, (int, float)):
            kind = random.choices(list(sms_band.keys()), weights=list(sms_band.values()))[0]
        else:
            kind = _pick_from_call_band(config, soph)
    else:
        kind = _pick_from_call_band(config, soph)
    return _phone_from_category(kind)


def generate_phishing_event(p_type: str, sophistication: Union[str, Dict[str, str]], config) -> dict:
    """
    피싱 유형(loan/institution/acquaintance/etc)에 따른 이벤트 1건 생성.
    """
    # 1. 개시 채널 → 발신번호
    contact_soph = _get_soph(sophistication, "문자선행개시")
    sms_prob = config.PHISHING_FIRST_CONTACT_TYPE_SMS[p_type][contact_soph]
    first_contact_type = "sms" if random.random() < sms_prob else "call"

    phone_number = _generate_phishing_phone(p_type, first_contact_type, sophistication, config)
    number_type = config.classify_number_type(phone_number)
    is_global = 1 if number_type == "00X(국제)" else 0

    # 2. 통화/문자 시간 및 기본 속성 (규칙 3: 항상 관측)
    base_time = datetime(2026, 1, 1) + timedelta(minutes=random.randint(0, 525600))
    call_time = base_time.strftime("%Y-%m-%d %H:%M:%S")
    hour_bucket = base_time.hour
    call_type = 1  # 수신 (피해자 단말 관점)

    # 3. 저장 및 과거 이력
    in_contacts = 1 if random.random() < config.PHISHING_IN_CONTACTS else 0
    has_prior_history = 1 if random.random() < config.PHISHING_HAS_PRIOR_HISTORY else 0
    if has_prior_history == 1:
        gap_cfg = config.PHISHING_REPEAT_GAP[p_type]
        theta = random.choices(gap_cfg["theta_list"], weights=gap_cfg["p_list"])[0]
        repeat_gap = round(np.random.exponential(scale=theta), 2)
    else:
        repeat_gap = np.nan  # 규칙 1

    # 4. 문자 내 번호 및 불일치 (Track C)
    num_soph = _get_soph(sophistication, "문자내_번호_포함확률")
    is_num_in_msg = 1 if random.random() < config.PHISHING_IS_NUM_IN_MSG[p_type][num_soph] else 0

    if is_num_in_msg == 1:
        differ_soph = _get_soph(sophistication, "발신번호_문자내번호_불일치율")
        inner_num_differs = 1 if random.random() < config.PHISHING_INNER_NUM_DIFFERS[differ_soph] else 0
    else:
        inner_num_differs = np.nan

    # 5. 문자 -> 통화 연계 간격 (Track B)
    sms_soph = _get_soph(sophistication, "문자_통화_연계")
    sms_to_call = 1 if (
        first_contact_type == "sms"
        and random.random() < config.PHISHING_SMS_TO_CALL[p_type][sms_soph]
    ) else 0
    if sms_to_call == 1:
        theta = config.PHISHING_SMS_TO_CALL_GAP[p_type]
        sms_to_call_gap_min = round(np.random.exponential(scale=theta), 2)
    else:
        sms_to_call_gap_min = np.nan

    # 6. URL 및 앱 설치 유도
    url_rate = config.PHISHING_IS_URL_IN_MSG[p_type]
    is_url_in_msg = 1 if random.random() < url_rate else 0

    if is_url_in_msg == 1:
        is_reliable_url = 1 if random.random() < config.PHISHING_IS_RELIABLE_URL else 0
        app_soph = _get_soph(sophistication, "앱설치_유도")
        has_appinstall_link = 1 if random.random() < config.PHISHING_HAS_APPINSTALL_LINK[p_type][app_soph] else 0
    else:
        is_reliable_url = np.nan
        has_appinstall_link = np.nan

    # 7. 순차복수사칭 (특수 규칙: 확정적 0 또는 1)
    seq_soph = _get_soph(sophistication, "순차복수사칭")
    is_sequential_callers = 1 if random.random() < config.PHISHING_IS_SEQUENTIAL_CALLERS[p_type][seq_soph] else 0

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
