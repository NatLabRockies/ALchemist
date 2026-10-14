import { describe, it, expect } from 'vitest';
import { isUntitledSession, sessionFileStem } from './fileNames';

const ID = '77491db8-ce1a-4a91-ad70-34c120a6fea4';

describe('sessionFileStem', () => {
  it('uses the session title', () => {
    expect(sessionFileStem('C1 syngas campaign', ID)).toBe('C1 syngas campaign');
  });

  it('falls back to the session id when the session is untitled', () => {
    expect(sessionFileStem('Untitled Session', ID)).toBe('alchemist_session_77491db8');
    expect(sessionFileStem('', ID)).toBe('alchemist_session_77491db8');
    expect(sessionFileStem('   ', ID)).toBe('alchemist_session_77491db8');
    expect(sessionFileStem(undefined, ID)).toBe('alchemist_session_77491db8');
  });

  it('replaces characters that Windows or macOS forbid in filenames', () => {
    expect(sessionFileStem('Run 3/4: Pt <vs> Pd? "final"|*', ID)).toBe('Run 3_4_ Pt _vs_ Pd_ _final___');
    expect(sessionFileStem('a\\b\tc', ID)).toBe('a_b c');
  });

  it('collapses whitespace and strips trailing dots and spaces', () => {
    expect(sessionFileStem('  many   spaces  ', ID)).toBe('many spaces');
    expect(sessionFileStem('ends with dots...', ID)).toBe('ends with dots');
  });

  it('does not produce a Windows reserved device name', () => {
    expect(sessionFileStem('CON', ID)).toBe('CON_');
    expect(sessionFileStem('lpt1', ID)).toBe('lpt1_');
  });

  it('falls back when sanitising leaves nothing usable', () => {
    expect(sessionFileStem('...', ID)).toBe('alchemist_session_77491db8');
  });

  it('caps very long titles', () => {
    expect(sessionFileStem('x'.repeat(300), ID)).toHaveLength(120);
  });
});

describe('isUntitledSession', () => {
  it('treats blank and the default name as untitled', () => {
    expect(isUntitledSession('Untitled Session')).toBe(true);
    expect(isUntitledSession('  ')).toBe(true);
    expect(isUntitledSession(undefined)).toBe(true);
    expect(isUntitledSession('C1 campaign')).toBe(false);
  });
});
