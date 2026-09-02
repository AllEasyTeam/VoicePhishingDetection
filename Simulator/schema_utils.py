"""
schema_utils.py — schema 조회·검증용 helper methods 정의 파일.

역할: schema_columns.py의 SCHEMA를 사용하는 함수들로 다른 file에서 import해서 사용.
"""

from typing import List, Optional
from Simulator.schema import Track
from Simulator.schema_columns import SCHEMA


def get_feature_columns(track_filter: Optional[List[Track]] = None) -> List[str]:
    """학습에 실제로 쓸 컬럼 이름 목록만 뽑아주는 함수. (is_feature=True인 column만, 특정 track만 filtering) """
    cols = [c for c in SCHEMA if c.is_feature]
    if track_filter:
        cols = [c for c in cols if c.track in track_filter]
    return [c.name for c in cols]


def get_nullable_columns() -> List:
    """결측 가능한 컬럼들만 뽑아주는 함수(nullable=True인 column만)"""
    return [c for c in SCHEMA if c.nullable]


def get_non_nullable_columns() -> List[str]:
    """항상 값이 있어야 하는 컬럼 이름 목록만 뽑아주는 함수(nullable=False인 column만)"""
    # get_nullable_columns()와 반대로 동작하는 함수.
    return [c.name for c in SCHEMA if not c.nullable]


def validate_schema() -> List[str]:
    """스키마 전체가 자체 모순 없는지 검사."""
    errors = []
    names = set()
    id_count = 0

    for col in SCHEMA:
        if col.name in names:
            errors.append(f"중복된 컬럼명: {col.name}")
            # column명이 중복되었을 때, error로 판단. 단순히 "이름" 문자열이 겹치는지만 확인.
        names.add(col.name)
        errors.extend(col.validate())
        # 각 column별로 validate() 호출 -> 검증.
        if col.track == Track.ID:
            id_count += 1
            # Track.ID로 지정된 column이 몇 개인지 count.
            # 이름이 다른 두 column이 각각 Track.ID로 지정돼도 names 활용한 중복 체크만으로는 확인 불가능. -> id_count != 1 별도 체크 필요.
            if col.is_feature:
                errors.append(f"{col.name}: ID 컬럼인데 is_feature=True로 설정됨")
                # ID column은 사건 식별용이라 feature로 사용 X(학습 시 사용 X) -> ID column이 is_feature=True로 설정되는 오류를 방지.

    # Track.ID 컬럼은 정확히 1개만 존재해야 함. id_count로 0개(식별자 없음)/2개 이상(중복 지정) 여부를 확인.
    if id_count != 1:
        errors.append(f"ID 컬럼이 {id_count}개 존재함 (정확히 1개여야 함)")

    return errors


if __name__ == "__main__":
    errs = validate_schema()
    if errs:
        print("스키마 오류:")
        for e in errs:
            print(" -", e)
    else:
        print(f"스키마 검증 통과. 총 {len(SCHEMA)}개 컬럼")
        print("단말/단말+구조적신호 학습 feature 개수:",
              len(get_feature_columns([Track.DEVICE, Track.DEVICE_STRUCTURAL])))