; Inno Setup script for the screen-mosaic tray app.
; Builds a single-file installer from the PyInstaller output in dist\mosaic.
;
; Per-user install (PrivilegesRequired=lowest) is intentional: mosaic.py writes
; mosaic_session_presets.json next to mosaic.exe when frozen (see
; session_presets_path() in mosaic.py), so the install directory must stay
; writable without elevation. Do not switch this to a Program Files install
; without also changing where that JSON is written.
;
; A Start Menu shortcut with a Ctrl+Alt+Shift+F12 hotkey is created via COM
; (WScript.Shell) in [Code], since Inno's [Icons] section has no hotkey
; property. Windows only arms shortcut hotkeys for .lnk files that live in an
; indexed location (Desktop / Start Menu), so the shortcut is placed in
; {group} rather than the install directory.
;
; Plain Ctrl+Alt+F12 was dropped because some other background app on a test
; machine had already claimed it via RegisterHotKey, silently blocking
; Explorer from ever owning it (confirmed by probing RegisterHotKey directly
; with explorer.exe killed). Ctrl+Alt+Shift+F12 was free.

#define MyAppName "화면 모자이크"
#define MyAppDirName "Mosaic"
#define MyAppExeName "mosaic.exe"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Mosaic"
#define MyAppHotkey "CTRL+ALT+SHIFT+F12"

[Setup]
AppId={{B6E2B6C1-2B7B-4E6C-9C1B-7A6B9E6B7B10}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\{#MyAppDirName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=installer_output
OutputBaseFilename=MosaicSetup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesInstallIn64BitMode=x64compatible
DisableWelcomePage=no

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Files]
Source: "dist\mosaic\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 바로가기 만들기"; GroupDescription: "추가 아이콘:"; Flags: unchecked

[Icons]
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[UninstallDelete]
Type: files; Name: "{group}\{#MyAppName}.lnk"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "설치를 마치고 {#MyAppName} 실행"; Flags: nowait postinstall skipifsilent

[Code]
procedure CreateHotkeyShortcut;
var
  WshShell: Variant;
  Shortcut: Variant;
  GroupDir: String;
begin
  GroupDir := ExpandConstant('{group}');
  if not DirExists(GroupDir) then
    ForceDirectories(GroupDir);
  try
    WshShell := CreateOleObject('WScript.Shell');
    Shortcut := WshShell.CreateShortcut(GroupDir + '\{#MyAppName}.lnk');
    Shortcut.TargetPath := ExpandConstant('{app}\{#MyAppExeName}');
    Shortcut.WorkingDirectory := ExpandConstant('{app}');
    Shortcut.IconLocation := ExpandConstant('{app}\{#MyAppExeName}');
    Shortcut.Hotkey := '{#MyAppHotkey}';
    Shortcut.Save;
  except
    MsgBox('바로가기 단축키(Ctrl+Alt+Shift+F12) 등록에 실패했습니다. 시작 메뉴에서 ' +
           '{#MyAppName} 바로가기를 직접 만든 뒤 속성에서 단축키를 지정해 주세요.',
           mbInformation, MB_OK);
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    CreateHotkeyShortcut;
end;
