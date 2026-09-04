# 피싱 이벤트 1건 생성을 위한 파일

import random
import numpy as np
from datetime import datetime, timedelta
from typing import Union, Dict

_AREA_CODES = [
    "031", "032", "033", "041", "042", "043", "044",
    "051", "052", "053", "054", "055", "061", "062", "063", "064",
]

# config.PHISHING_통화_발신번호_종류_대역 미러 (config 미수정 전제)
_CALL_BAND = {
    "010": {"짧게": 0.40, "중간": 0.55, "길게": 0.70},
    "특번(1566등)": {"짧게": 0.15, "중간": 0.25, "길게": 0.35},
    "070/기타": {"짧게": 0.05, "중간": 0.20, "길게": 0.35},
}

# config.PHISHING_문자_발신번호_종류_대역 미러
# 지인사칭형만 고유 비중, 나머지는 통화 대역 차용
_SMS_BAND_지인사칭 = {"010": 0.99, "기타": 0.01}


def _get_soph(soph: Union[str, Dict[str, str]], key: str) -> str:
    return soph.get(key, "중간") if isinstance(soph, dict) else soph


def _digits(n: int) -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(n))


def _generate_other_phone() -> str:
    """070/기타·기타 대역: 유선/특번/070/0N0/국제 (이미지 허용 형식)"""
    kind = random.choice(["특번", "02", "지역", "070", "0N0", "국제"])
    if kind == "특번":
        return random.choice(["15", "16", "18"]) + _digits(6)
    if kind == "02":
        return "02" + str(random.randint(1, 9)) + _digits(random.choice([6, 7]))
    if kind == "지역":
        return random.choice(_AREA_CODES) + str(random.randint(1, 9)) + _digits(random.choice([6, 7]))
    if kind == "070":
        return "070" + _digits(random.randint(5, 8))
    if kind == "0N0":
        prefix = random.choice(["060", "080", "030", "050"])
        max_total = 12 if prefix in ("030", "050") else 11
        return prefix + _digits(random.randint(5, max_total - 3))
    return "00" + _digits(random.randint(8, 12))


def _phone_from_kind(kind: str) -> str:
    if kind == "010":
        return "010" + _digits(8)
    if kind.startswith("특번"):
        return random.choice(["15", "16", "18"]) + _digits(6)
    return _generate_other_phone()  # 070/기타, 기타


def _pick_call_kind(soph: str) -> str:
    cats = list(_CALL_BAND.keys())
    return random.choices(cats, weights=[_CALL_BAND[c][soph] for c in cats])[0]


def _generate_phishing_phone(
    p_type: str,
    first_contact_type: str,
    sophistication: Union[str, Dict[str, str]],
) -> str:
    """문자/통화 대역에 따라 피싱 발신번호 생성."""
    soph = _get_soph(sophistication, "발신번호_종류_대역")
    if soph not in ("짧게", "중간", "길게"):
        soph = "중간"

    if first_contact_type == "sms" and p_type == "지인사칭형":
        kind = random.choices(
            list(_SMS_BAND_지인사칭.keys()),
            weights=list(_SMS_BAND_지인사칭.values()),
        )[0]
    else:
        # 통화 대역, 또는 문자인데 통화 대역 차용(기관/대출/기타)
        kind = _pick_call_kind(soph)
    return _phone_from_kind(kind)


def generate_phishing_event(p_type: str, sophistication: Union[str, Dict[str, str]], config) -> dict:
    """
    피싱 유형(대출/기관/지인사칭/기타)에 따른 차등 θ값을 적용하여 이벤트 1건 생성
    """
    # 1. 개시 채널 → 발신번호 (문자/통화 대역이 채널에 의존)
    first_contact_type = "sms" if p_type in ["대출사기형", "지인사칭형"] else random.choice(["call", "sms"])
    phone_number = _generate_phishing_phone(p_type, first_contact_type, sophistication)
    number_type = config.classify_number_type(phone_number)
    is_global = 1 if number_type == "00X(국제)" else 0

    # 2. 통화/문자 시간 및 기본 속성 (규칙 3: 항상 관측)
    base_time = datetime(2026, 1, 1) + timedelta(minutes=random.randint(0, 525600))
    call_time = base_time.strftime("%Y-%m-%d %H:%M:%S")
    hour_bucket = base_time.hour
    call_type = 1  # 수신 (피해자 단말 관점)

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
