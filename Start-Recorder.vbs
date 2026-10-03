' Starts Sotto without a console window.
'
' Compared with the previous version:
'   * The working directory is set to the script folder rather than
'     WshShell.CurrentDirectory - that field does not reliably point here when
'     launched from a shortcut or from Explorer.
'   * If pythonw.exe is missing or the launch fails there is a message instead
'     of silence.

Option Explicit

Dim shell, fso, scriptDir, target, command, result

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
target = fso.BuildPath(scriptDir, "main.py")

If Not fso.FileExists(target) Then
    MsgBox "main.py was not found in:" & vbCrLf & scriptDir, _
           vbCritical, "Sotto"
    WScript.Quit 1
End If

' Set the working directory so relative paths resolve correctly
shell.CurrentDirectory = scriptDir

command = "pythonw.exe """ & target & """"

On Error Resume Next
result = shell.Run(command, 0, False)
If Err.Number <> 0 Then
    Err.Clear
    ' Second attempt through the Python launcher
    result = shell.Run("pyw.exe """ & target & """", 0, False)
    If Err.Number <> 0 Then
        MsgBox "Python could not be started." & vbCrLf & vbCrLf & _
               "Please install Python 3 and make sure pythonw.exe is on " & _
               "the PATH." & vbCrLf & vbCrLf & _
               "To troubleshoot, run this from a command prompt:" & vbCrLf & _
               "    python """ & target & """", _
               vbCritical, "Sotto"
        WScript.Quit 1
    End If
End If
On Error Goto 0
