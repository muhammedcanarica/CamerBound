#define MyAppName "CamerBound"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Muhammed Can Arıca"
#define MyAppExeName "CamerBound.exe"

[Setup]
AppId={{C58C83F8-4D15-4D5F-9738-A3728EE6AA6F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\CamerBound
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SourceDir=..\..
OutputDir=dist-installer
OutputBaseFilename=CamerBound_Setup
SetupIconFile=packaging\assets\CamerBound.ico
UninstallDisplayIcon={app}\CamerBound.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
VersionInfoVersion=1.0.0.0
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} Windows Installer
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppVersion}

[Languages]
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"

[Tasks]
Name: "desktopicon"; Description: "Masaüstü kısayolu oluştur"; GroupDescription: "Ek kısayollar:"; Flags: unchecked

[Files]
Source: "dist\CamerBound\*"; DestDir: "{app}"; Excludes: "config\settings.json,data\*"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "dist\CamerBound\config\settings.json"; DestDir: "{app}\config"; Flags: ignoreversion onlyifdoesntexist uninsneveruninstall
Source: "packaging\assets\CamerBound.ico"; DestDir: "{app}"; Flags: ignoreversion

[Dirs]
Name: "{app}\data"; Flags: uninsneveruninstall

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\CamerBound.ico"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\CamerBound.ico"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "CamerBound'u Başlat"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent
