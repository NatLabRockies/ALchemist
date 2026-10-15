import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

/**
 * index.html carries a blocking inline script that sets the stored theme on
 * <html> before first paint (useTheme only runs after the bundle loads). The
 * script is run here against a stubbed document and localStorage.
 */
// vitest runs from alchemist-web/, where index.html lives
const html = readFileSync(resolve(process.cwd(), 'index.html'), 'utf8');
const head = html.slice(html.indexOf('<head>'), html.indexOf('</head>'));
const inline = [...head.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)].map((m) => m[1]);

function boot(stored: string | null | 'throws') {
  const classes = new Set<string>(['dark']); // index.html ships class="dark"
  const document = {
    documentElement: {
      classList: {
        add: (c: string) => classes.add(c),
        remove: (c: string) => classes.delete(c),
        toggle: (c: string, force?: boolean) => {
          if (force === undefined ? !classes.has(c) : force) classes.add(c);
          else classes.delete(c);
        },
        contains: (c: string) => classes.has(c),
      },
    },
  };
  const localStorage = {
    getItem: (key: string) => {
      if (stored === 'throws') throw new Error('storage blocked');
      return key === 'theme' ? stored : null;
    },
  };
  new Function('document', 'localStorage', inline.join('\n'))(document, localStorage);
  return classes.has('dark');
}

describe('index.html theme bootstrap', () => {
  it('has a blocking inline script in <head>', () => {
    expect(inline).toHaveLength(1);
    expect(head).not.toMatch(/<script[^>]*\btype="module"/);
    expect(head).not.toMatch(/<script[^>]*\b(async|defer)\b/);
  });

  it('removes the dark class when light is stored', () => {
    expect(boot('light')).toBe(false);
  });

  it('keeps dark when dark is stored', () => {
    expect(boot('dark')).toBe(true);
  });

  it('defaults to dark with nothing stored', () => {
    expect(boot(null)).toBe(true);
  });

  it('defaults to dark for an unrecognised value', () => {
    expect(boot('sepia')).toBe(true);
  });

  it('defaults to dark when storage throws', () => {
    expect(boot('throws')).toBe(true);
  });
});
