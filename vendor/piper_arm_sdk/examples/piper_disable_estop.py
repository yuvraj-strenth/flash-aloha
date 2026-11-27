import time
from piper_sdk import C_PiperInterface_V3

arm = C_PiperInterface_V3(can_name="can0")
arm.ConnectPort(piper_init=False)

try:
    for _ in range(30):
        feedback = arm.GetArmStatus()
        if feedback.time_stamp > 0:
            break
        time.sleep(0.1)
    else:
        raise TimeoutError("No arm status feedback on can0")

    status = feedback.arm_status
    if status.arm_status.value == 0x01:
        arm.EmergencyStop(0x02)  # Resume from emergency stop
        for _ in range(20):
            time.sleep(0.25)
            status = arm.GetArmStatus().arm_status
            if status.arm_status.value != 0x01:
                break
    if status.arm_status.value != 0x00:
        raise RuntimeError(f"Arm fault remains: {status.arm_status}")

    for _ in range(20):
        arm.MotionCtrl_2(
            ctrl_mode=0x01,          # CAN control
            move_mode=0x01,          # MOVE J
            move_spd_rate_ctrl=20,
            is_mit_mode=0x00,
        )
        time.sleep(0.25)
        status = arm.GetArmStatus().arm_status
        if status.ctrl_mode.value == 0x01 and status.mode_feed.value == 0x01:
            print(f"CAN / MOVE J confirmed; arm_status=0x{status.arm_status.value:02x}")
            motors = arm.GetArmLowSpdInfoMsgs()
            print("driver_codes:", [
                f"0x{getattr(motors, f'motor_{i}').foc_status_code:02x}"
                for i in range(1, 7)
            ])
            break
    else:
        raise TimeoutError(
            f"Mode did not change: ctrl={status.ctrl_mode}, "
            f"move={status.mode_feed}, arm_status={status.arm_status}"
        )
finally:
    arm.DisconnectPort()
