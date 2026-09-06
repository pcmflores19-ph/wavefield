; Inno Setup script for the Wavefield Windows installer.
;
; Build the app first, then compile this:
;     pyinstaller packaging/autocut.spec --noconfirm
;     "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\autocut.iss
;
; Produces packaging/output/AutoCut-Setup-<version>.exe - one file to hand to
; someone, no Python and no ffmpeg needed on their machine.

#define AppName "Wavefield"
#define AppExeName "Wavefield.exe"
#define AppPublisher "Paul Flores"
#define AppURL "https://github.com/pcmflores19-ph/wavefield"

; Version comes from packaging/build.py as /DAppVersion=..., which reads it
; out of auto_cut/version.py. The fallback is only for running ISCC by hand.
; (An earlier attempt to read version.py with the Inno preprocessor was left
; half-written and never actually ran, so the version silently drifted.)
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{8F3C2A64-5B7E-4D91-A0C3-7E5D9B1A2F48}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE

; Per-user install by default, so no administrator password is needed. That
; matters for the audience here - people on managed university laptops who
; cannot elevate.
PrivilegesRequiredOverridesAllowed=dialog
PrivilegesRequired=lowest

; Only ever offer to close Wavefield itself.
;
; Inno asks Windows which programs are holding the files it is about to
; replace, and by default that covers every .exe and .dll being installed -
; including the Microsoft C++ runtime that PyInstaller ships beside the app.
; Other programs can end up bound to those copies, so a plain upgrade
; announced that it needed to close Google Chrome. Alarming, and not something
; a podcaster should have to reason about. Restricting the check to our own
; executable keeps the useful behaviour - closing a running Wavefield so it
; can be replaced - and drops the rest.
CloseApplicationsFilter=Wavefield.exe

OutputDir=output
OutputBaseFilename=Wavefield-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExeName}
; The installer's own icon, and the one shown in Add/Remove Programs.
SetupIconFile=autocut.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; \
    GroupDescription: "Additional shortcuts:"

[Files]
; The whole PyInstaller one-folder build, ffmpeg included. The three Microsoft
; C++ runtime DLLs are left out here and installed by the entry below.
Source: "..\dist\Wavefield\*"; DestDir: "{app}"; \
    Excludes: "\_internal\MSVCP140.dll,\_internal\VCRUNTIME140.dll,\_internal\VCRUNTIME140_1.dll"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

; The Microsoft C++ runtime, without ignoreversion. These three are identical
; from build to build, so left to its own judgement Setup compares versions,
; sees nothing newer and skips them - rather than trying to overwrite a DLL
; that another running program is holding open, which fails with
; "DeleteFile failed; code 5" and aborts the upgrade.
Source: "..\dist\Wavefield\_internal\MSVCP140.dll"; DestDir: "{app}\_internal"
Source: "..\dist\Wavefield\_internal\VCRUNTIME140.dll"; DestDir: "{app}\_internal"
Source: "..\dist\Wavefield\_internal\VCRUNTIME140_1.dll"; DestDir: "{app}\_internal"

; Also beside the .exe, not only in _internal, so the [Run] entry
; below and the Settings button can both name one obvious path.
Source: "setup_whisperx.ps1"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; \
    Tasks: desktopicon

[Run]
; Speech recognition is a separate install: WhisperX with PyTorch is several
; gigabytes, and which build is right depends on whether there is an NVIDIA
; card in the machine. The script works that out, installs into its own
; environment and writes the path into Wavefield's settings, so nobody has
; to meet pip. Ticked by default - without it the transcript never appears.
Filename: "powershell.exe"; \
    Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\setup_whisperx.ps1"""; \
    Description: "Set up speech recognition (downloads about 2-3 GB)"; \
    Flags: postinstall skipifsilent

Filename: "{app}\{#AppExeName}"; \
    Description: "Start {#AppName}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; The WhisperX environment the setup script builds. Several gigabytes,
; and useless without the app, so it goes when the app goes.
Type: filesandordirs; Name: "{localappdata}\Wavefield"
; Decoded audio and transcripts the app caches next to itself. Regenerated on
; demand, and can run to gigabytes, so leaving it behind would be rude.
;
; A frozen build writes these beside its own modules, which PyInstaller puts in
; _internal - not next to the .exe. Both spellings are listed because a build
; run from source, or a future PyInstaller that drops _internal, would use the
; other one, and an uninstaller that leaves gigabytes behind is worse than a
; rule that matches nothing.
Type: filesandordirs; Name: "{app}\_internal\.cache"
Type: files; Name: "{app}\_internal\autocut_crash.log"
Type: filesandordirs; Name: "{app}\.cache"
Type: files; Name: "{app}\autocut_crash.log"
