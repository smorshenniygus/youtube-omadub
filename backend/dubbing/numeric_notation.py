"""Canonical numeric notation shared by source, output and repair constraints.

This is a surface-form check, not a proof of mathematical or semantic equivalence.
Subscripts, superscripts and fractions are quantities, not extra numeric facts.
"""
from decimal import Decimal, localcontext
import re
import unicodedata

_VULGAR = {'½':(1,2), '⅓':(1,3), '⅔':(2,3), '¼':(1,4), '¾':(3,4),
           '⅕':(1,5), '⅖':(2,5), '⅗':(3,5), '⅘':(4,5), '⅙':(1,6),
           '⅚':(5,6), '⅛':(1,8), '⅜':(3,8), '⅝':(5,8), '⅞':(7,8)}
_NUMERATORS = {'a':1, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5,
               'six':6, 'seven':7, 'eight':8, 'nine':9}
_DENOMINATORS = {'half':2, 'halves':2, 'third':3, 'thirds':3,
                 'quarter':4, 'quarters':4, 'fourth':4, 'fourths':4,
                 'fifth':5, 'fifths':5, 'sixth':6, 'sixths':6,
                 'seventh':7, 'sevenths':7, 'eighth':8, 'eighths':8,
                 'ninth':9, 'ninths':9, 'tenth':10, 'tenths':10}
_WORD_FRACTION = re.compile(r'\b(\d+|'+'|'.join(_NUMERATORS)+r')[ -]+('+
                            '|'.join(sorted(_DENOMINATORS, key=len, reverse=True))+r')\b', re.I)
_FRACTION = re.compile(r'(?<![\w/])([-−]?\d+)\s*/\s*(\d+)(?![\d/])')


def _ratio(numerator, denominator):
    with localcontext() as context:
        context.prec = 28
        return format(Decimal(numerator) / Decimal(denominator), 'f')


def written_fraction_values(text):
    """Written quantities are optional evidence, never compulsory digit facts."""
    values = []
    def replace(match):
        numerator = match.group(1).lower()
        denominator = match.group(2).lower()
        if numerator.isdigit() or text[match.end():].startswith('-'):
            return match.group()
        value = _NUMERATORS[numerator]
        if value != 1 and not denominator.endswith('s'):
            return match.group()
        # A fractional tail is not the whole amount: "a decade and a half"
        # must not authorize 0.5 decades. Leave such compound quantities in
        # words instead of publishing a misleading partial numeric constraint.
        if not re.search(r'\b(?:and|plus)\s*$', text[:match.start()], re.I):
            values.append(Decimal(_ratio(value, _DENOMINATORS[denominator])))
        return ' '
    return _WORD_FRACTION.sub(replace, text), values


def canonical_numeric_text(text):
    # Do this before NFKC: otherwise 1½ becomes the ambiguous string 11/2.
    def vulgar(match):
        whole = int(match.group(1) or 0)
        numerator, denominator = _VULGAR[match.group(2)]
        return _ratio(whole*denominator+numerator, denominator)
    text = re.sub(r'(\d+)?(['+''.join(_VULGAR)+r'])', vulgar, text)
    # Keep a signed superscript exponent separate from its base (2⁻³).
    text = re.sub(r'[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+', lambda m: '^'+unicodedata.normalize('NFKC', m.group()), text)
    text = unicodedata.normalize('NFKC', text).replace('⁄', '/')
    def written(match):
        numerator = match.group(1).lower()
        if not numerator.isdigit():
            return match.group()
        numerator = int(numerator)
        denominator = match.group(2).lower()
        # "two third-party tools" is not two thirds. Singular denominators
        # belong to a unit numerator; hyphenated suffixes may be ordinary words.
        if (numerator != 1 and denominator not in ('halves', 'thirds', 'quarters', 'fourths',
                'fifths', 'sixths', 'sevenths', 'eighths', 'ninths', 'tenths')) or text[match.end():].startswith('-'):
            return match.group()
        return _ratio(numerator, _DENOMINATORS[denominator])
    text = _WORD_FRACTION.sub(written, text)
    def fraction(match):
        numerator = int(match.group(1).replace('−', '-'))
        denominator = int(match.group(2))
        # Division by zero and idiomatic ratios like 50/50 or 20/20 stay as separate numbers.
        if not denominator or (abs(numerator) == denominator and denominator >= 10):
            return match.group()
        return _ratio(numerator, denominator)
    return _FRACTION.sub(fraction, text)
