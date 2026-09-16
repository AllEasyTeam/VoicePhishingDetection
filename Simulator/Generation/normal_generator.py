# 정상 이벤트 1건 생성을 위한 파일

import random
import numpy as np
from datetime import datetime, timedelta
from typing import Union, Dict

from Simulator.Generation.generator_utils import get_soph, digits, phone_from_category


def _group_key(subgroup: str, config) -> str:
    """기관_개인/기관_기업 → SUBGROUP_INSTITUTION 으로 묶음."""
    if subgroup == config.SUBGROUP_ACQUAINTANCE:
        return config.SUBGROUP_ACQUAINTANCE
    return config.SUBGROUP_INSTITUTION


def _generate_normal_phone(subgroup: str, config, soph: str) -> str:
    """
    - 지인: 010 고정
    - 기관: 화이트리스트 우선, 아니면 NORMAL_NUMBER_TYPE 비중으로 합성
    """
    if subgroup == config.SUBGROUP_ACQUAINTANCE:
        return "010" + digits(8)

    whitelist = config.WHITELIST_LIST
    if whitelist and random.random() < 0.85:
        return random.choice(whitelist)

    weights = config.NORMAL_NUMBER_TYPE["비중"][soph]
    # 기관 합성에서는 개인 이동번호(010) 제외
    org_weights = {k: v for k, v in weights.items() if k != "010"}
    categories = list(org_weights.keys())
    category = random.choices(categories, weights=[org_weights[c] for c in categories])[0]
    return phone_from_category(category)


def _calculate_repeat_gap(group_key: str, config, sophistication: Union[str, Dict[str, str]]) -> float:
    """NORMAL_REPEAT_GAP 혼합분포로 재연락 간격(분) 계산.
    호출 자체는 generate_normal_event()의 has_repeat_contact가 이미 게이트했으므로,
    여기서는(지인/기관 모두) 항상 값을 계산해서 반환함(별도의 내부 발생확률 체크 없음)."""
    gap_soph = get_soph(sophistication, config.SOPH_REPEAT_GAP)
    cfg = config.NORMAL_REPEAT_GAP[group_key]
    theta = random.choices(cfg["theta_list"], weights=cfg["p_list"])[0]
    return round(np.random.exponential(scale=theta * cfg["배율"][gap_soph]), 2)


def generate_normal_event(subgroup: str, config, sophistication: Union[str, Dict[str, str]] = "mid") -> dict:
    """
    정상 하위집단(지인/기관) 규칙에 따라 이벤트 1건 생성.
    subgroup: config.SUBGROUP_ACQUAINTANCE | SUBGROUP_INSTITUTION_PERSONAL | SUBGROUP_INSTITUTION_CORPORATE
    sophistication: "low"|"mid"|"high" 또는 feature별 dict
    """
    group_key = _group_key(subgroup, config)
    band_soph = get_soph(sophistication, config.SOPH_NUMBER_TYPE_BAND)

    # 1. 식별자 및 발신 대역 (Track B)
    phone_number = _generate_normal_phone(subgroup, config, band_soph)
    number_type = config.classify_number_type(phone_number)
    is_global = 1 if number_type == "00X(국제)" else 0

    # 2. 통화 시간 및 기본 속성 (규칙 3: 항상 관측)
    base_time = datetime(2026, 1, 1) + timedelta(minutes=random.randint(0, 525600))
    call_time = base_time.strftime("%Y-%m-%d %H:%M:%S")
    hour_bucket = base_time.hour
    # call_type=1(수신) 고정 가정이 실측 근거 없이 판별력만 갖는 것으로 판단되어 보류.
    # call_type = random.choice([0, 1])  # 0: 발신, 1: 수신

    contact_soph = get_soph(sophistication, config.SOPH_FIRST_CONTACT)
    sms_prob = config.NORMAL_FIRST_CONTACT_TYPE_SMS[group_key][contact_soph]
    first_contact_type = 0 if random.random() < sms_prob else 1  # 0: 문자(sms), 1: 통화(call)

    # 3. 연락처 저장 및 과거 통화 이력
    in_contacts = 1 if random.random() < config.NORMAL_IN_CONTACTS[group_key] else 0
    has_prior_history = 1 if random.random() < config.NORMAL_HAS_PRIOR_HISTORY[group_key] else 0  # 사건 간(독립적, 하위 컬럼 게이트 안 함)

    # 4. 재연락 간격 (규칙 1: 사건 내 반복 접촉 없으면 NaN)
    # NORMAL_HAS_REPEAT_CONTACT: 현재는 지인/기관 둘 다 sophistication(LOW/MID/HIGH) 구조.
    # (혹시 나중에 flat 값으로 바뀌어도 깨지지 않도록 dict 여부는 방어적으로 확인)
    repeat_contact_cfg = config.NORMAL_HAS_REPEAT_CONTACT[group_key]
    if isinstance(repeat_contact_cfg, dict):
        repeat_contact_soph = get_soph(sophistication, config.SOPH_REPEAT_CONTACT)
        repeat_contact_prob = repeat_contact_cfg[repeat_contact_soph]
    else:
        repeat_contact_prob = repeat_contact_cfg
    has_repeat_contact = 1 if random.random() < repeat_contact_prob else 0  # 사건 내(repeat_gap 게이트)
    if has_repeat_contact == 1:
        repeat_gap = _calculate_repeat_gap(group_key, config, sophistication)
    else:
        repeat_gap = np.nan

    # 5. 문자 내 번호 포함 및 불일치 (Track C)
    # NORMAL_INNER_NUM_DIFFERS 는 이미 불일치율
    num_in_msg_soph = get_soph(sophistication, config.SOPH_NUM_IN_MSG)
    is_num_in_msg = 1 if random.random() < config.NORMAL_IS_NUM_IN_MSG[num_in_msg_soph] else 0

    if is_num_in_msg == 1:
        differ_soph = get_soph(sophistication, config.SOPH_INNER_NUM_DIFFERS)
        inner_num_differs = 1 if random.random() < config.NORMAL_INNER_NUM_DIFFERS[differ_soph] else 0

        official_soph = get_soph(sophistication, config.SOPH_MSG_OFFICIAL_MATCH)
        msg_number_official_match = 1 if random.random() < config.NORMAL_MSG_NUMBER_OFFICIAL_MATCH[official_soph] else 0
    else:
        inner_num_differs = np.nan  # 규칙 1
        msg_number_official_match = np.nan  # 규칙 1

    # 6. 문자 -> 통화 연계 및 간격 (Track B)
    sms_to_call = 1 if (first_contact_type == 0 and random.random() < config.NORMAL_SMS_TO_CALL) else 0  # first_contact_type == 0: 문자(sms)로 시작한 경우
    if sms_to_call == 1:
        gap_soph = get_soph(sophistication, config.SOPH_SMS_TO_CALL_GAP)
        gap_cfg = config.NORMAL_SMS_TO_CALL_GAP
        theta = random.choices(gap_cfg["theta_list"], weights=gap_cfg["p_list"])[0]
        sms_to_call_gap = round(np.random.exponential(scale=theta * gap_cfg["배율"][gap_soph]), 2)
    else:
        sms_to_call_gap = np.nan  # 규칙 1

    # 7. URL 및 앱 설치 유도 (Track C)
    url_soph = get_soph(sophistication, config.SOPH_URL_RATE)
    is_url_in_msg = 1 if random.random() < config.NORMAL_IS_URL_IN_MSG[url_soph] else 0

    if is_url_in_msg == 1:
        is_reliable_url = 1 if random.random() < config.NORMAL_IS_RELIABLE_URL else 0
        app_soph = get_soph(sophistication, config.SOPH_APP_INSTALL)
        has_appinstall_link = 1 if random.random() < config.NORMAL_HAS_APPINSTALL_LINK[app_soph] else 0
    else:
        is_reliable_url = np.nan    # 규칙 1
        has_appinstall_link = np.nan # 규칙 1

    # 8. 순차복수사칭 및 Track A 전용 결측치 (규칙 특수: 0 채움 / 규칙 4: NaN)
    seq_soph = get_soph(sophistication, config.SOPH_SEQUENTIAL_CALLERS)
    is_sequential_callers = 1 if random.random() < config.NORMAL_IS_SEQUENTIAL_CALLERS[seq_soph] else 0
    is_carrier_altered = np.nan
    number_cluster = np.nan  # 다수 사건 비교(통신사 실측자료) 필요 -> 시뮬레이터에서는 항상 NaN
    using_duration = np.nan
    unique_callees = np.nan

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
        "number_cluster": number_cluster,
        "using_duration": using_duration,
        "unique_callees": unique_callees,
        "is_global": is_global,
        "is_sequential_callers": is_sequential_callers,
    }
