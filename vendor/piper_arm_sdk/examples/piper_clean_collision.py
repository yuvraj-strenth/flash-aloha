import time
from piper_sdk import C_PiperInterface_V3

arm = C_PiperInterface_V3(can_name="can0")
arm.ConnectPort(piper_init=False)

def codes():
    feedback = arm.GetArmLowSpdInfoMsgs()
    motors = [getattr(feedback, f"motor_{i}") for i in range(1, 7)]
    if any(m.can_id != 0x260 + i for i, m in enumerate(motors, 1)):
        return None
    return [m.foc_status_code for m in motors]

try:
    for _ in range(50):
        if codes() is not None:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("No feedback from all six motor drivers")

    for joint in (3, 4):
        arm.JointConfig(joint_num=joint, set_zero=0,
                        acc_param_is_effective=0, max_joint_acc=0x7FFF,
                        clear_err=0xAE)

    for _ in range(50):
        time.sleep(0.1)
        state = codes()
        if state is not None and all(code & 0x30 == 0 for code in state):
            break
    else:
        raise RuntimeError(f"Fault still present: {[hex(c) for c in codes()]}")

    arm.EnableArm(7)  # Enable all six joint motors
    time.sleep(0.5)
    state = codes()
    print("Driver codes:", [hex(c) for c in state])
    if not all(code & 0x40 for code in state):
        raise RuntimeError("One or more motors did not enable")
    print("Collision/error flags cleared; all joint motors enabled.")
finally:
    arm.DisconnectPort()