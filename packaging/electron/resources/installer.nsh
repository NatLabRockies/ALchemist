; Backstop for updates and reinstalls: the app stops its server before "Restart to
; update", but if a server is still running (an orphan, or the app was killed),
; end it so its files can be replaced.
!macro customInit
  nsExec::Exec 'taskkill /im alchemist-server.exe /f'
!macroend
