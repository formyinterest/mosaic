; Mosaic 실행 단축키 (AutoHotkey v2)
; 단축키: Ctrl+Alt+R  (왼손만으로 누를 수 있도록 R = "Run". 변경하려면 아래 ^!r 부분 수정)
; ^  = Ctrl,  !  = Alt,  +  = Shift,  #  = Win

#SingleInstance Force

ExePath := A_ScriptDir . "\dist\mosaic\mosaic.exe"

^!r:: {
    if WinExist("ahk_exe mosaic.exe")
        return
    Run(ExePath)
}
