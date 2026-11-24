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
