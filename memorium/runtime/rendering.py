import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import subprocess
FONT_PATH = os.environ.get('VMC_CANVAS_FONT', 'DejaVuSans.ttf')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def memory_text(state):
    # ponytail: source/step cells, not a learned graph; test layouts before adding one.
    return '\n'.join(f'## Step {r["step"] + 1}: {r["title"]}\n{r["text"]}' for r in state)


@lru_cache(maxsize=256)
def font(size):
    from PIL import ImageFont
    return ImageFont.truetype(FONT_PATH, size)

@lru_cache(maxsize=16384)
def wrap(text, size, width):
    """Preserve grapheme clusters; None means this font candidate is too wide."""
    import regex
    face = font(size)
    lines = []
    for paragraph in text.split('\n'):
        atoms = regex.findall(r'\X', paragraph)
        start = 0
        while start < len(atoms):
            def fits(length):
                value = ''.join(atoms[start:start + length])
                box = face.getbbox(value, anchor='lt')
                return max(face.getlength(value), box[2] - box[0]) <= width
            lo, hi = 0, len(atoms) - start
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if fits(mid):
                    lo = mid
                else:
                    hi = mid - 1
            if lo == 0:
                return None
            spaces = [i + 1 for i, atom in enumerate(atoms[start:start + lo]) if atom.isspace()]
            cut = spaces[-1] if start + lo < len(atoms) and spaces else lo
            # A shorter prefix can have a different terminal glyph bearing.
            while cut and not fits(cut):
                cut -= 1
            if not cut:
                return None
            lines.append(''.join(atoms[start:start + cut]))
            start += cut
        if not paragraph:
            lines.append('')
    # Newline controls are not printed glyphs; every other character survives.
    assert ''.join(lines) == text.replace('\n', '')
    return tuple(lines)

def font_coverage(states):
    """Report unmapped characters; a drawn fallback box is not a faithful glyph."""
    result = subprocess.run(['fc-query', '--format', '%{charset}', font(22).path], check=True, capture_output=True, text=True)
    spans = [tuple(int(v, 16) for v in item.split('-')) for item in result.stdout.split()]
    text = '\n'.join(memory_text(r['state']) for r in states)
    missing = sorted({c for c in text if not c.isspace() and not any(s[0] <= ord(c) <= s[-1] for s in spans)})
    return {'font_sha256': sha(font(22).path), 'charset_sha256': digest(result.stdout),
            'unmapped_characters': [{'codepoint': f'U+{ord(c):04X}', 'character': c, 'occurrences': text.count(c)} for c in missing],
            'unmapped_occurrences': sum(text.count(c) for c in missing),
            'boundary': 'Same font for all layouts; no source deletion, transliteration or fallback-font substitution. Unicode glyph fidelity is incomplete where cmap entries are absent; source-string and pixel replay do not remove that limitation.'}
