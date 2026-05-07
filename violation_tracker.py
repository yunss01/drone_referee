"""
violation_tracker.py  —  Step 7
0.1초 연속 침범 판정 시 침범 확정 + 타이머 관리

사용법 (단독 테스트):
    python3 violation_tracker.py
"""

import time


# ──────────────────────────────────────────────────────────
#  설정값
# ──────────────────────────────────────────────────────────
VIOLATION_HOLD_SEC = 0.1   # 이 시간 이상 연속 침범 시 확정
# ──────────────────────────────────────────────────────────


class ViolationTracker:
    """
    바퀴 1개당 인스턴스 1개 생성해서 사용.

    사용 예시:
        tracker = ViolationTracker(wheel_id=0)

        for frame in camera_stream():
            violated = lane_checker.check(...)   # bool
            state    = tracker.update(violated)

            if state["confirmed"]:
                print(f"침범 확정! 지속 {state['duration']:.2f}초")
    """

    def __init__(self, wheel_id: int = 0):
        self.wheel_id       = wheel_id
        self._start_time    = None   # 침범 시작 시각
        self._confirmed     = False  # 한 번 확정되면 True 유지

    def update(self, violated: bool) -> dict:
        """
        매 프레임마다 호출.

        violated : 이번 프레임의 침범 여부 (lane_checker 결과)
        반환     : {
                     "violated"  : bool,   이번 프레임 원시 판정
                     "confirmed" : bool,   0.1초 이상 지속 여부
                     "duration"  : float,  현재 연속 침범 지속 시간(초)
                                           침범 아니면 0.0
                   }
        """
        now = time.time()

        if violated:
            if self._start_time is None:
                # 침범 시작
                self._start_time = now

            duration = now - self._start_time

            if duration >= VIOLATION_HOLD_SEC:
                self._confirmed = True
        else:
            # 침범 해제 → 타이머 초기화
            self._start_time = None
            self._confirmed  = False
            duration         = 0.0

        # violated=False 인 경우 duration=0.0 이 반환되도록
        if not violated:
            duration = 0.0
        else:
            duration = now - self._start_time

        return {
            "wheel_id"  : self.wheel_id,
            "violated"  : violated,
            "confirmed" : self._confirmed,
            "duration"  : duration,
        }

    def reset(self):
        """수동 초기화 (경진대회 라운드 시작 시 등)"""
        self._start_time = None
        self._confirmed  = False


# ── 단독 테스트 ──────────────────────────────────────────
if __name__ == "__main__":
    import itertools

    print("=== ViolationTracker 단독 테스트 ===\n")
    tracker = ViolationTracker(wheel_id=0)

    # 시나리오: False 3회 → True 계속 → False 2회
    scenario = (
        [False] * 3 +
        [True]  * 20 +   # 0.1초 넘도록 충분히
        [False] * 2
    )

    for frame_idx, violated in enumerate(scenario):
        time.sleep(0.016)   # 약 60fps 시뮬레이션
        state = tracker.update(violated)

        flag = ""
        if state["confirmed"]:
            flag = "  ★ 침범 확정"
        elif state["violated"]:
            flag = f"  (누적 {state['duration']*1000:.0f}ms)"

        print(f"  frame {frame_idx:02d} | violated={str(violated):<5} "
              f"| confirmed={state['confirmed']} "
              f"| duration={state['duration']*1000:5.1f}ms{flag}")

    print("\n✅ 테스트 완료")
