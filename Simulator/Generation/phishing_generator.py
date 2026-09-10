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
    """sophistication 값을 "이 feature 기준으로" 최종 확정하는 함수."""
    # ex1) "high" 와 같은 형태의 입력 => 모든 feature에 동일하게 적용.
    # ex2) {"발신번호_종류_대역": "low"} 와 같은 형태의 입력 => key에 해당하는 feature만 적용.
    # soph이 dict이면 key값 우선, 없으면 __base__(민감도 base_sophistication), 그래도 없으면 mid.
    # soph이 dict이 아니라면 문자열 그대로 적용.
    if isinstance(soph, dict):
        raw = soph[key] if key in soph else soph.get("__base__", "mid")
    else:
        raw = soph
    # _SOPH_ALIAS 통해 최종적으로 "low", "mid", "high" 중 하나로 정규화.
    return _SOPH_ALIAS.get(raw, "mid")


def _digits(n: int) -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(n))


def _phone_from_category(category: str) -> str:
    """classify_number_type 카테고리 + 이미지 허용 형식."""
    if category == "010":
        return "010" + _digits(8)
    if category == "특번":
        return random.choice(["15", "16", "18"]) + _digits(6)
    if category == "02(서울 유선)":
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
    """확률분포를 이용해 발신번호 카테고리(발신번호_종류_대역)를 뽑는 함수."""
    band = config.PHISHING_CALL_NUMBER_TYPE
    cats = list(band.keys())
    return random.choices(cats, weights=[band[c][soph] for c in cats])[0]


def _generate_phishing_phone(
    p_type: str,
    first_contact_type: int,  # 0: 문자(sms), 1: 통화(call)
    sophistication: Union[str, Dict[str, str]],
    config,
) -> str:
    """PHISHING_SMS_NUMBER_TYPE(발신번호_종류_대역 - 문자) / PHISHING_CALL_NUMBER_TYPE(발신번호_종류_대역 - 통화) 비중으로 발신번호 생성."""
    # p_type: 피싱 유형(loan/institution/acquaintance/etc)
    # first_contact_type: 개시 채널(0: 문자/sms, 1: 통화/call)

    soph = _get_soph(sophistication, config.SOPH_NUMBER_TYPE_BAND)

    '''이 피싱 유형이 "문자로" 발신번호를 보낼 때 쓰는 확률분포(PHISHING_SMS_NUMBER_TYPE) 가져오기. 
    -> acquaintance: {"010": 0.99, "기타": 0.01} 처럼 flat
    -> 그 외 유형: PHISHING_CALL_NUMBER_TYPE 차용해서 동일하게 사용. (카테고리 → {low/mid/high})'''    
    sms_band = config.PHISHING_SMS_NUMBER_TYPE.get(p_type)

    # 개시 채널이 문자이고, sms_band가 dict이면(= 해당 피싱 유형이 문자 발신번호 확률분포를 갖는 경우) sms_band에서 확률분포에 따라 발신번호 카테고리 뽑기.
    # 그 외에는, PHISHING_CALL_NUMBER_TYPE에서 확률분포에 따라 발신번호 카테고리 뽑기.
    if first_contact_type == 0 and isinstance(sms_band, dict):  # 0: 문자(sms)
        sample = next(iter(sms_band.values()), None)
        if isinstance(sample, (int, float)):
            # acquaintance 같이 flat(민감도 분석 X)인 경우, 바로 확률로 뽑기.
            kind = random.choices(list(sms_band.keys()), weights=list(sms_band.values()))[0]
        else:
            # 그 외에 "low", "mid", "high"와 같은 후보값이 있는 경우, PHISHING_CALL_NUMBER_TYPE 차용하므로 sophistication까지 모두 고려하도록 함수 호출.
            kind = _pick_from_call_band(config, soph)
    else:
        # 개시 채널이 통화이거나, sms_band가 dict이 아닌 경우
        kind = _pick_from_call_band(config, soph)
    return _phone_from_category(kind)


def generate_phishing_event(p_type: str, sophistication: Union[str, Dict[str, str]], config) -> dict:
    """
    피싱 유형(loan/institution/acquaintance/etc)에 따른 이벤트 1건 생성.
    """
    # 1. 개시 채널 → 발신번호
    # column 중 "first_contact_type", "phone_number", "number_type", "is_global" 값 확정.
    contact_soph = _get_soph(sophistication, config.SOPH_FIRST_CONTACT)
    sms_prob = config.PHISHING_FIRST_CONTACT_TYPE_SMS[p_type][contact_soph]
    first_contact_type = 0 if random.random() < sms_prob else 1  # 0: 문자(sms), 1: 통화(call)

    phone_number = _generate_phishing_phone(p_type, first_contact_type, sophistication, config)
    number_type = config.classify_number_type(phone_number)
    is_global = 1 if number_type == "00X(국제)" else 0


    # 2. 통화/문자 시간 및 기본 속성 (call_time/call_type은 규칙 3: 항상 관측 / hour_bucket은 ETC만 결측)
    # column 중 "call_time", "hour_bucket", "call_type" 값 확정
    skew_prob = config.PHISHING_HOUR_BUCKET[p_type] 
    if skew_prob is None:
        # 보이스피싱 유형이 ETC인 경우(skew_prob이 None임)
        hour = random.randint(0, 23)  # 근거 없음 -> 24시간 균등하게 선택.
    elif random.random() < skew_prob:
        # 9~18시 사이에 일어날 확률 -> hour 구하기
        hour = random.randint(9, 17)  # 09~18시 사이
    else:
        # 9~18시 이외에 일어날 확률 -> hour 구하기
        hour = random.choice([h for h in range(24) if not (9 <= h < 18)])  # 그 외 시간대

    base_time = datetime(2026, 1, 1) + timedelta(
        days=random.randint(0, 364), hours=hour, minutes=random.randint(0, 59)
    )
    call_time = base_time.strftime("%Y-%m-%d %H:%M:%S")  # call_time은 항상 실제 시각을 가짐
    hour_bucket = np.nan if skew_prob is None else hour   # ETC 유형만 hour_bucket 자체를 NaN 처리
    # call_type=1(수신) 고정 가정이 실측 근거 없이 판별력만 갖는 것으로 판단되어 보류.
    # call_type = 1  # 수신 (피해자 단말 관점)


    # 3. 저장 및 과거 이력
    # column 중 "in_contacts", "has_prior_history", "has_repeat_contact", "repeat_gap" 값 확정
    in_contacts = 1 if random.random() < config.PHISHING_IN_CONTACTS else 0
    has_prior_history = 1 if random.random() < config.PHISHING_HAS_PRIOR_HISTORY else 0  # 사건 간(독립적, 하위 컬럼 게이트 안 함)

    # PHISHING_HAS_REPEAT_CONTACT가 LOW/MID/HIGH 구조로 바뀜 -> sophistication 조회 필요.
    repeat_contact_soph = _get_soph(sophistication, config.SOPH_REPEAT_CONTACT)
    has_repeat_contact = 1 if random.random() < config.PHISHING_HAS_REPEAT_CONTACT[repeat_contact_soph] else 0  # 사건 내(repeat_gap 게이트)
    if has_repeat_contact == 1:
        # 사건 내 반복 접촉이 존재 => 재연락 간격이 존재할 수 있음. (게이트 조건)
        gap_cfg = config.PHISHING_REPEAT_GAP[p_type]
        theta = random.choices(gap_cfg["theta_list"], weights=gap_cfg["p_list"])[0]
        repeat_gap = round(np.random.exponential(scale=theta), 2)
    else:
        # 사건 내 반복 접촉 존재 X => 재연락 간격 존재 X (NaN 처리)
        repeat_gap = np.nan


    # 4. 문자 내 번호 및 불일치 (Track C)
    # column 중 "is_num_in_msg", "inner_num_differs", "msg_number_official_match" 값 확정,
    num_soph = _get_soph(sophistication, config.SOPH_NUM_IN_MSG)
    is_num_in_msg = 1 if random.random() < config.PHISHING_IS_NUM_IN_MSG[p_type][num_soph] else 0

    if is_num_in_msg == 1:
        # 문자 내 번호 포함 O => 발신번호-문자 내 번호 불일치 확인 가능.
        differ_soph = _get_soph(sophistication, config.SOPH_INNER_NUM_DIFFERS)
        inner_num_differs = 1 if random.random() < config.PHISHING_INNER_NUM_DIFFERS[differ_soph] else 0

        # 문자 내 번호 포함 O => 그 번호가 대표번호와 일치하는지도 확인 가능.
        official_soph = _get_soph(sophistication, config.SOPH_MSG_OFFICIAL_MATCH)
        msg_number_official_match = 1 if random.random() < config.PHISHING_MSG_NUMBER_OFFICIAL_MATCH[official_soph] else 0
    else:
        # 문자 내 번호 포함 X => 두 비교 모두 불가능 (NaN 처리)
        inner_num_differs = np.nan
        msg_number_official_match = np.nan


    # 5. 문자 -> 통화 연계 간격 (Track B)
    # column 중 "sms_to_call", "sms_to_call_gap" 값 확정
    sms_soph = _get_soph(sophistication, config.SOPH_SMS_TO_CALL)
    sms_to_call = 1 if (
        # 문자 -> 통화 연계 여부는 문자(0)로 시작해야 하므로 first_contact_type도 확인해야 함.
        first_contact_type == 0
        and random.random() < config.PHISHING_SMS_TO_CALL[p_type][sms_soph]
    ) else 0
    if sms_to_call == 1:
        # 문자->통화 연계 O => 문자->통화 전환 간격 구하기
        theta = config.PHISHING_SMS_TO_CALL_GAP[p_type]
        sms_to_call_gap = round(np.random.exponential(scale=theta), 2)
    else:
        # 문자->통화 연계 X => 문자->통화 전환 간격 존재 X (NaN 처리)
        sms_to_call_gap = np.nan


    # 6. URL 및 앱 설치 유도
    # column 중 "is_url_in_msg", "is_reliable_url", "has_appinstall_link" 값 확정.
    url_rate = config.PHISHING_IS_URL_IN_MSG[p_type]
    is_url_in_msg = 1 if random.random() < url_rate else 0

    if is_url_in_msg == 1:
        # 문자 내 url 존재 O => url 공식 도메인 확률 및 앱설치 유도 신호가 있는지 확인 가능.
        is_reliable_url = 1 if random.random() < config.PHISHING_IS_RELIABLE_URL else 0
        app_soph = _get_soph(sophistication, config.SOPH_APP_INSTALL)
        has_appinstall_link = 1 if random.random() < config.PHISHING_HAS_APPINSTALL_LINK[p_type][app_soph] else 0
    else:
        # 문자 내 url 존재 X => 모두 NaN 처리.
        is_reliable_url = np.nan
        has_appinstall_link = np.nan


    # 7. 순차복수사칭 (특수 규칙: 확정적 0 또는 1)
    # column 중 "is_sequential_callers" 값 확정
    seq_soph = _get_soph(sophistication, config.SOPH_SEQUENTIAL_CALLERS)
    is_sequential_callers = 1 if random.random() < config.PHISHING_IS_SEQUENTIAL_CALLERS[p_type][seq_soph] else 0


    # 8. Track A 전용 feature (규칙 4: 통신사 데이터 부재로 NaN)
    # column 중 "is_carrier_altered", "using_duration", "unique_callees", "number_cluster" 값 확정.
    is_carrier_altered = np.nan
    using_duration = np.nan
    unique_callees = np.nan
    number_cluster = np.nan  


    return {
        "phone_number": phone_number,
        "number_type": number_type,
        "call_time": call_time,
        # "call_type": call_type,
        "hour_bucket": hour_bucket,
        "first_contact_type": first_contact_type,
        "has_prior_history": has_prior_history,
        "has_repeat_contact": has_repeat_contact,
        "repeat_gap": repeat_gap,
        "in_contacts": in_contacts,
        "is_num_in_msg": is_num_in_msg,
        "inner_num_differs": inner_num_differs,
        "msg_number_official_match": msg_number_official_match,
        "sms_to_call": sms_to_call,
        "sms_to_call_gap": sms_to_call_gap,
        "is_url_in_msg": is_url_in_msg,
        "is_reliable_url": is_reliable_url,
        "has_appinstall_link": has_appinstall_link,
        "is_carrier_altered": is_carrier_altered,
        "using_duration": using_duration,
        "unique_callees": unique_callees,
        "number_cluster": number_cluster,
        "is_global": is_global,
        "is_sequential_callers": is_sequential_callers
    }
