"""Conservative recognition of Russian cardinal phrases for numeric matching.

Not a semantic validator. Unsupported forms still require digits from the editor.
Whole phrases are consumed so 20 cannot match the prefix of 21 or 20000.
"""
import re
from decimal import Decimal

_FORMS = (
    (0, 'ноль нуль нуля нулю нулём нуле'),
    (1, 'один одна одно одну одного одной одному одним одном'),
    (2, 'два две двух двум двумя'), (3, 'три трёх трех трём трем тремя'),
    (4, 'четыре четырёх четырех четырём четырем четырьмя'),
    (5, 'пять пяти пятью'), (6, 'шесть шести шестью'),
    (7, 'семь семи семью'), (8, 'восемь восьми восемью восьмью'),
    (9, 'девять девяти девятью'), (10, 'десять десяти десятью'),
    (11, 'одиннадцать одиннадцати одиннадцатью'),
    (12, 'двенадцать двенадцати двенадцатью'),
    (13, 'тринадцать тринадцати тринадцатью'),
    (14, 'четырнадцать четырнадцати четырнадцатью'),
    (15, 'пятнадцать пятнадцати пятнадцатью'),
    (16, 'шестнадцать шестнадцати шестнадцатью'),
    (17, 'семнадцать семнадцати семнадцатью'),
    (18, 'восемнадцать восемнадцати восемнадцатью'),
    (19, 'девятнадцать девятнадцати девятнадцатью'),
    (20, 'двадцать двадцати двадцатью'), (30, 'тридцать тридцати тридцатью'),
    (40, 'сорок сорока'), (50, 'пятьдесят пятидесяти пятьюдесятью'),
    (60, 'шестьдесят шестидесяти шестьюдесятью'),
    (70, 'семьдесят семидесяти семьюдесятью'),
    (80, 'восемьдесят восьмидесяти восемьюдесятью'),
    (90, 'девяносто девяноста'), (100, 'сто ста'), (200, 'двести двухсот'),
    (300, 'триста трёхсот трехсот'), (400, 'четыреста четырёхсот четырехсот'),
    (500, 'пятьсот пятисот'), (600, 'шестьсот шестисот'),
    (700, 'семьсот семисот'), (800, 'восемьсот восьмисот'),
    (900, 'девятьсот девятисот'),
    (1000, 'тысяча тысячи тысяч тысячу тысячей тысяче тысячами тысячах'),
    (1000000, 'миллион миллиона миллионов миллиону миллионом миллионе'),
    (1000000000, 'миллиард миллиарда миллиардов миллиарду миллиардом миллиарде'),
)
_VALUES = {word: value for value, forms in _FORMS for word in forms.split()}
_WORD = r'(?:' + '|'.join(sorted(_VALUES, key=len, reverse=True)) + r')(?!\w)'
_PHRASE = re.compile(r'(?<![\w-])(?:минус\s+)?' + _WORD + r'(?:\s+' + _WORD + r')*(?![\w-])', re.IGNORECASE)


_ROMAN = re.compile(r'(?<!\w)[IVXLCDM]+(?!\w)')
_ROMAN_CANONICAL = re.compile(r'M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})')
_ROMAN_VALUES = dict(I=1, V=5, X=10, L=50, C=100, D=500, M=1000)


def roman_numbers(text):
    values = []
    for match in _ROMAN.finditer(text):
        token = match.group()
        if not _ROMAN_CANONICAL.fullmatch(token):
            continue
        total = previous = 0
        for letter in reversed(token):
            value = _ROMAN_VALUES[letter]
            total += -value if value < previous else value
            previous = max(previous, value)
        values.append(Decimal(total))
    return values


def russian_cardinals(text):
    values = []
    for match in _PHRASE.finditer(text):
        # Do not take a suffix of a digit/word quantity or an integer part of
        # unsupported fractional/ordinal notation as an independent cardinal.
        if re.search(r'\d\s*$', text[:match.start()]):
            continue
        if re.match(r'\s*(?:\d|цел\w*|десят\w*|сотых|тысячных|перв\w*|втор\w*|треть\w*)', text[match.end():], re.IGNORECASE):
            continue
        words = match.group().lower().split()
        negative = words[0] == 'минус'
        if negative:
            words = words[1:]
        total = group = 0
        last_scale, ceiling = 10**12, 1000
        valid = True
        for word in words:
            value = _VALUES[word]
            if value >= 1000:
                if value >= last_scale or (group == 0 and ceiling == 0):
                    valid = False
                    break
                total += (group or 1) * value
                group, ceiling, last_scale = 0, 1000, value
            else:
                if value >= ceiling:
                    valid = False
                    break
                group += value
                ceiling = 100 if value >= 100 else (10 if value >= 20 else 0)
        if valid:
            values.append(Decimal(-(total + group) if negative else total + group))
    return values


_ORDINAL_END = r'(?:ый|ий|ой|ая|ое|ые|ого|его|ому|ему|ым|им|ом|ем|ую|юю|ых|их|ыми|ими)'
_ORDINAL_STEMS = (
    (1, r'перв' + _ORDINAL_END),
    (2, r'втор' + _ORDINAL_END),
    (3, r'трет(?:ий|ья|ье|ьи|ьего|ьему|ьим|ьем|ьей|ью|ьих|ьими)'),
    (4, r'четв[её]рт' + _ORDINAL_END),
    (5, r'пят' + _ORDINAL_END),
    (6, r'шест' + _ORDINAL_END),
    (7, r'седьм' + _ORDINAL_END),
    (8, r'восьм' + _ORDINAL_END),
    (9, r'девят' + _ORDINAL_END),
    (10, r'десят' + _ORDINAL_END),
    (11, r'одиннадцат' + _ORDINAL_END),
    (12, r'двенадцат' + _ORDINAL_END),
    (13, r'тринадцат' + _ORDINAL_END),
    (14, r'четырнадцат' + _ORDINAL_END),
    (15, r'пятнадцат' + _ORDINAL_END),
    (16, r'шестнадцат' + _ORDINAL_END),
    (17, r'семнадцат' + _ORDINAL_END),
    (18, r'восемнадцат' + _ORDINAL_END),
    (19, r'девятнадцат' + _ORDINAL_END),
    (20, r'двадцат' + _ORDINAL_END),
    (30, r'тридцат' + _ORDINAL_END),
    (40, r'сороков' + _ORDINAL_END),
    (50, r'пятидесят' + _ORDINAL_END),
    (60, r'шестидесят' + _ORDINAL_END),
    (70, r'семидесят' + _ORDINAL_END),
    (80, r'восьмидесят' + _ORDINAL_END),
    (90, r'девяност' + _ORDINAL_END),
    (100, r'сот' + _ORDINAL_END),
    (200, r'двухсот' + _ORDINAL_END),
    (300, r'тр[её]хсот' + _ORDINAL_END),
    (400, r'четыр[её]хсот' + _ORDINAL_END),
    (500, r'пятисот' + _ORDINAL_END),
    (600, r'шестисот' + _ORDINAL_END),
    (700, r'семисот' + _ORDINAL_END),
    (800, r'восьмисот' + _ORDINAL_END),
    (900, r'девятисот' + _ORDINAL_END),
    (1000, r'тысячн' + _ORDINAL_END),
    (2000, r'двухтысячн' + _ORDINAL_END),
)
_ORDINAL_COMPILED = [(val, re.compile(r'^(?:' + pat + r')$', re.IGNORECASE)) for val, pat in _ORDINAL_STEMS]
_ORDINAL_WORD_RE = re.compile(
    r'(?<![\w-])(?:' + '|'.join(pat for _, pat in _ORDINAL_STEMS) + r')(?![\w-])',
    re.IGNORECASE,
)
_ORDINAL_TENS = {
    'двадцать': 20, 'тридцать': 30, 'сорок': 40, 'пятьдесят': 50,
    'шестьдесят': 60, 'семьдесят': 70, 'восемьдесят': 80, 'девяносто': 90,
}
_ORDINAL_HUNDREDS = {
    'сто': 100, 'двести': 200, 'триста': 300, 'четыреста': 400,
    'пятьсот': 500, 'шестьсот': 600, 'семьсот': 700, 'восемьсот': 800, 'девятьсот': 900,
}
_HYPHEN_SMALL_VALUES = {
    word: value for value, forms in _FORMS[:20] for word in forms.split()
}
_HYPHEN_CARDINAL_RE = re.compile(
    r'(?:(?<=-)|(?<![\w-]))(' + '|'.join(sorted(_HYPHEN_SMALL_VALUES, key=len, reverse=True)) + r')(?:(?=-)|(?![\w-]))',
    re.IGNORECASE,
)


def russian_ordinals(text):
    """Recognize Russian ordinal forms (e.g. 'девятнадцатого', 'двадцать первого') for expected-value matching."""
    values = []
    for match in _ORDINAL_WORD_RE.finditer(text):
        if re.search(r'\d\s*$', text[:match.start()]):
            continue
        word = match.group().lower()
        val = None
        for v, pat in _ORDINAL_COMPILED:
            if pat.fullmatch(word):
                val = v
                break
        if val is None:
            continue
        # Combine preceding cardinal tens/hundreds (e.g., "двадцать первого" -> 21, "сто двадцатого" -> 120).
        prefix_words = re.findall(r'[а-яё]+', text[:match.start()].lower())[-2:]
        total = val
        if val < 10 and prefix_words and prefix_words[-1] in _ORDINAL_TENS:
            total += _ORDINAL_TENS[prefix_words[-1]]
            prefix_words = prefix_words[:-1]
        if total < 100 and prefix_words and prefix_words[-1] in _ORDINAL_HUNDREDS:
            total += _ORDINAL_HUNDREDS[prefix_words[-1]]
        values.append(Decimal(total))
    return values


def russian_hyphenated_cardinals(text):
    """Recognize hyphen-delimited phonetic Russian digits (e.g. 'три-би-один-би', 'пэ-один')."""
    values = []
    for match in _HYPHEN_CARDINAL_RE.finditer(text):
        # Only emit tokens actually adjacent to a hyphen; standalone words are already handled by russian_cardinals.
        left_h = match.start() > 0 and text[match.start() - 1] == '-'
        right_h = match.end() < len(text) and text[match.end()] == '-'
        if not (left_h or right_h):
            continue
        # Skip Russian compound ordinals/adverbs like "во-первых" or digit-attached forms like "19-го".
        if re.search(r'\d-$', text[:match.start()]):
            continue
        val = _HYPHEN_SMALL_VALUES.get(match.group(1).lower())
        if val is not None:
            values.append(Decimal(val))
    return values
