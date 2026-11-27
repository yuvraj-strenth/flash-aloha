from piper_sdk import C_PiperInterface_V3

piper = C_PiperInterface_V3("can0")
piper.ConnectPort()
piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=50, is_mit_mode=0xAD)
print("Mode changed")