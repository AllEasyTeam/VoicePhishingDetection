# Simulator 코드 작성을 위한 main.py
# 역할: 전체 파이프라인의 진입점. mode에 따라 Generation과 Detection을 순서대로 호출.

from Simulator.Generation import config
from Simulator.Generation.dataset_builder import build_dataset
from Simulator.Detection.train_eval import split_data, train_model, evaluate
from Simulator.Detection.sensitivity_analysis import run_sensitivity


def run_final():
    """최종 모드: 확정한 config 값으로 데이터셋 1개 생성 -> split_data() 3분할 -> train_model()/evaluate()."""
    pass


def run_sensitivity_mode():
    """민감도분석 모드: config.py 스윕 파라미터를 하나씩 바꿔가며 run_sensitivity() 반복 호출 -> 비교표 출력."""
    


    pass


def main(mode: str):
    if mode == "final":
        run_final()
    elif mode == "sensitivity":
        run_sensitivity_mode()
    else:
        raise ValueError(f"알 수 없는 mode: {mode}")


if __name__ == "__main__":
    main(mode="final")
