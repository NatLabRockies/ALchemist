/** Save shortcut labels in the platform's own notation. */
const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform);
export const SAVE_SHORTCUT = IS_MAC ? '⌘S' : 'Ctrl+S';
export const SAVE_AS_SHORTCUT = IS_MAC ? '⇧⌘S' : 'Ctrl+Shift+S';
