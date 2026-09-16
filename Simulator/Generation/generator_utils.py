# 정상/피싱 생성기가 공유하는 번호 합성·sophistication 조회 유틸.
import random
from typing import Union, Dict

AREA_CODES = [
    "031", "032", "033", "041", "042", "043", "044",
    "051", "052", "053", "054", "055", "061", "062", "063", "064",
]

# sophistication 별칭 (구 한글키 호환)
SOPH_ALIAS = {
    "짧게": "low", "중간": "mid", "길게": "high",
    "low": "low", "mid": "mid", "high": "high",
    "낮음": "low", "높음": "high",
}


def get_soph(soph: Union[str, Dict[str, str]], key: str) -> str:
    """sophistication을 이 feature 기준으로 low/mid/high로 확정.

    문자열이면 전체 feature에 동일 적용. dict이면 key 값 우선,
    없으면 __base__(민감도 base_sophistication), 그래도 없으면 mid.
    """
    if isinstance(soph, dict):
        raw = soph[key] if key in soph else soph.get("__base__", "mid")
    else:
        raw = soph
    return SOPH_ALIAS.get(raw, "mid")


def digits(n: int) -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(n))


def phone_from_category(category: str) -> str:
    """classify_number_type 카테고리 + 이미지 허용 형식."""
    if category == "010":
        return "010" + digits(8)
    if category == "특번":
        return random.choice(["15", "16", "18"]) + digits(6)
    if category == "02(서울 유선)":
        return "02" + str(random.randint(1, 9)) + digits(random.choice([6, 7]))
    if category == "070(인터넷전화)":
        return "070" + digits(random.randint(5, 8))
    if category == "00X(국제)":
        return "00" + digits(random.randint(8, 12))
    # 기타: 지방 유선 또는 0N0(060/080/030/050). 020/040/090 금지
    if random.random() < 0.5:
        return random.choice(AREA_CODES) + str(random.randint(1, 9)) + digits(random.choice([6, 7]))
    prefix = random.choice(["060", "080", "030", "050"])
    max_total = 12 if prefix in ("030", "050") else 11
    return prefix + digits(random.randint(5, max_total - 3))
