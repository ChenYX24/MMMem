"""Canvas layout and validation."""
import hashlib
import json
import math
import re

WIRE_EDGE_PATTERNS = {edge: (r"^0(\.[0-9]+)?$" if edge in ("left", "top")
                            else r"^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$")
                      for edge in ("left", "top", "right", "bottom")}


def writer_json_schema(image_ids=None, flow=False):
    """Constrain the wire format; transition still owns geometry, provenance and topology."""
    def closed(properties):
        return {'type': 'object', 'properties': properties, 'required': list(properties),
                'additionalProperties': False}

    key = {'type': 'string', 'pattern': '^[A-Za-z0-9_.:-]{1,120}$'}
    box = closed({edge: {'type': 'string', 'pattern': pattern}
                  for edge, pattern in WIRE_EDGE_PATTERNS.items()})
    text = closed({'id': key, 'kind': {'const': 'text'}, 'box': box,
                   'font_size': {'type': 'integer', 'minimum': 1, 'maximum': 96},
                   'text': {'type': 'string'}})
    image = closed({'id': key, 'kind': {'const': 'image'}, 'box': box,
                    'asset_id': key if image_ids is None else {'enum': list(image_ids)}})
    canvas = closed({'width': {'type': 'integer', 'minimum': 128, 'maximum': 4096},
                     'height': {'type': 'integer', 'minimum': 128, 'maximum': 4096},
                     'items': {'type': 'array', 'items': {'anyOf': [text] if image_ids == [] else [text, image]},
                               'minItems': 1, 'maxItems': 64}})
    if flow:
        for item in (text, image):
            del item['properties']['box']
            item['required'].remove('box')
        canvas = closed({'layout': {'const': 'rows'}, 'width': {'enum': [512, 1024, 2048, 4096]},
                         'rows': {'type': 'array', 'minItems': 1, 'maxItems': 64,
                            'items': {'type': 'array', 'minItems': 1, 'maxItems': 64,
                                'items': {'anyOf': [text] if image_ids == [] else [text, image]}}}})
    upsert = closed({'id': key, 'source_ids': {'type': 'array', 'items': key, 'minItems': 1},
                     'canvas': canvas})
    parent = {'anyOf': [{'type': 'null'}, closed({'id': key,
                        'related': {'type': ['string', 'null']}})]}
    return closed({name: {'type': 'array', 'items': item, 'maxItems': 16}
                   for name, item in [('upsert', upsert),
                       ('reparent', closed({'id': key, 'parent': parent})), ('delete', key)]})

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", value):
        raise ValueError("invalid identifier")
    return value

def canvas_box_pixels(box, width, height):
    """Named normalized edges, with strict containment before pixel quantization."""
    if (not isinstance(box, dict) or set(box) != {'left', 'top', 'right', 'bottom'}
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in box.values())):
        raise ValueError('box requires finite named left, top, right, bottom edges; no xywh arrays')
    left, top, right, bottom = (box[k] for k in ('left', 'top', 'right', 'bottom'))
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError(f'canvas item outside bounds or reversed: {box}; '
                         'require 0 <= left < right <= 1 and 0 <= top < bottom <= 1')
    x, y = math.ceil(left * width), math.ceil(top * height)
    w, h = math.floor(right * width) - x, math.floor(bottom * height) - y
    if w < 1 or h < 1:
        raise ValueError(f'canvas box has no complete pixel at {width}x{height}: {box}')
    return x, y, w, h

def validate_canvas(canvas, allowed_images):
    if not isinstance(canvas, dict) or set(canvas) != {'width', 'height', 'items'}:
        raise ValueError('invalid canvas schema')
    if any(type(canvas[k]) is not int or not 128 <= canvas[k] <= 4096 for k in ('width', 'height')):
        raise ValueError('invalid canvas size')
    if not isinstance(canvas['items'], list) or not 1 <= len(canvas['items']) <= 64:
        raise ValueError('invalid canvas item count')
    keys = []
    for item in canvas['items']:
        if not isinstance(item, dict):
            raise ValueError('invalid canvas item')
        base = {'id', 'kind', 'box'}
        required = base | ({'text', 'font_size'} if item.get('kind') == 'text' else {'asset_id'})
        if item.get('kind') not in ('text', 'image') or set(item) != required:
            raise ValueError('only declarative text/image canvas items are executable')
        keys.append(identifier(item['id']))
        try:
            canvas_box_pixels(item['box'], canvas['width'], canvas['height'])
        except ValueError as exc:
            raise ValueError(f'canvas item {item["id"]}: {exc}') from exc
        if item['kind'] == 'text':
            if not isinstance(item['text'], str) or not item['text'].strip() or len(item['text']) > 16000:
                raise ValueError('invalid canvas text')
            if type(item['font_size']) is not int or not 1 <= item['font_size'] <= 96:
                raise ValueError('invalid font size')
        elif identifier(item['asset_id']) not in allowed_images:
            raise ValueError('image must come from the cited, already observed sources')
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate canvas item id')
