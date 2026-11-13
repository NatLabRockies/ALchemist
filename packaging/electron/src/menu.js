// Application menus. The Edit menu matters on macOS: without it, copy and paste don't work in text fields.
import { Menu, dialog, shell } from 'electron'

export function buildMenu ({ platform, version, docsUrl, logsDir, isPackaged }) {
  const about = () => dialog.showMessageBox({
    type: 'info', title: 'About ALchemist', message: `ALchemist ${version}`,
    detail: 'Active learning for experiment design.\nNational Laboratory of the Rockies. BSD-3-Clause.',
  })
  const quit = { role: 'quit', label: 'Quit ALchemist' }
  const template = [
    ...(platform === 'darwin'
      ? [{ label: 'ALchemist', submenu: [{ label: 'About ALchemist', click: about }, { type: 'separator' }, { role: 'hide' }, { role: 'hideOthers' }, { role: 'unhide' }, { type: 'separator' }, quit] }]
      : [{ label: 'File', submenu: [quit] }]),
    { role: 'editMenu' },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        ...(isPackaged ? [] : [{ role: 'toggleDevTools' }]),
        { type: 'separator' },
        { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    { role: 'windowMenu' },
    {
      role: 'help',
      submenu: [
        { label: 'ALchemist Documentation', click: () => shell.openExternal(docsUrl) },
        { label: 'Open Logs Folder', click: () => shell.openPath(logsDir) },
        ...(platform === 'darwin' ? [] : [{ type: 'separator' }, { label: 'About ALchemist', click: about }]),
      ],
    },
  ]
  return Menu.buildFromTemplate(template)
}
