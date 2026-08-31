"""
schema.py - column schema 정의

역할: Track 종류(통신사, 단말...), column 값의 형태(이진, 텍스트...), ColumnSchema class 정의.

"""
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

class Track(Enum):
    # 어떤 Track에 속하는 column인가. 
    CARRIER = "통신사"  # 기존 A
    DEVICE = "단말"   # 기존 B
    DEVICE_STRUCTURAL = "단말+구조적 신호" # 기존 C
    ID = "id" # feature 아님. 사건 식별용

class ValueType(Enum):
    # column 값의 형태
    TEXT = "텍스트"
    CATEGORICAL = "범주형"
    DATETIME = "datetime"
    BINARY = "이진"
    CONTINUOUS_TIME = "연속형(시간)" # 분/일 단위
    CONTINUOUS_COUNT = "연속형(개수)" # 명수 등


@dataclass
class ColumnSchema:
    # column 한 개의 구조를 정의하는 class
    name: str # feature 변수명
    definition: str # 각 feature 변수명이 의미하는 한글 정의
    value_type: ValueType
    track: Track
    is_feature: bool = True # False라면 학습 시 제외.(phone_number)
    nullable: bool = False # 결측 가능 여부(True라면 결측 가능한 column)
    depends_on: Optional[str] = None # 결측 규칙 설명. (nullable=True일 때 설명 필요 및 결측 규칙 명시적 확인을 위한 용도)


    def validate(self) -> List[str]:
        # column 정의가 올바른지 검증하는 method.
        errors = []
        if self.nullable and self.depends_on is None:
            errors.append(f"{self.name}: nullable인데 depends_on 설명이 없음")
            # 결측 규칙이 명시되지 않은 column이므로 오류로 간주. 
        if not self.is_feature and self.nullable:
            errors.append(f"{self.name}: feature가 아닌 column은 결측 규칙 불필요")
        return errors