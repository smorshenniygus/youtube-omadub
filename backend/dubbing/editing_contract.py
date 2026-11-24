"""Editor response contract, independent of CLI transport and numeric parsing."""
import copy

SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['items'],
          'properties': {'items': {'type': 'array', 'minItems': 1, 'maxItems': 6,
              'items': {'type': 'object', 'additionalProperties': False,
                  'required': ['id', 'text_ru'], 'properties': {
                      'id': {'type': 'string'},
                      'text_ru': {'type': 'string', 'minLength': 1, 'maxLength': 10000}}}}}}


class EditingError(RuntimeError):
    """Transport, process, or unsafe/unparseable output; not content-repairable."""


class ResponseEditingError(EditingError):
    """Parsed response failed validation; eligible for one bounded repair."""
    def __init__(self, message, feedback):
        super().__init__(message)
        self.feedback = feedback
        self.previous_output = None
        self.usage = {}


class NumericEditingError(ResponseEditingError):
    def __init__(self, feedback):
        super().__init__('Editing changed numeric facts', feedback)


def response_schema(rows):
    """Require the batch cardinality and allowed IDs; uniqueness is validated too."""
    if not isinstance(rows, list) or not 1 <= len(rows) <= 6:
        raise ValueError('Expected 1..6 response rows')
    schema = copy.deepcopy(SCHEMA)
    items = schema['properties']['items']
    items['minItems'] = items['maxItems'] = len(rows)
    variants = []
    for row in rows:
        variant = copy.deepcopy(SCHEMA['properties']['items']['items'])
        variant['properties']['id']['enum'] = [row['id']]
        # Character targets estimate duration; imposing them on structured
        # decoding can cut off words/facts. Only the resource bound is hard.
        variants.append(variant)
    items['items'] = {'anyOf': variants}
    return schema
