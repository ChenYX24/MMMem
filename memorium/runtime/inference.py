"""Model inference and lossless page rendering."""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

from .rendering import font, wrap, font_coverage
from .model_io import json_object
from .canvas import canvas_box_pixels, digest, validate_canvas, writer_json_schema

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

class GenerationLimitError(RuntimeError):
    """A structured generation exhausted its budget before completing JSON."""

class CanvasSizeError(ValueError):
    """A valid flow needs multiple display pages; the stored node is unchanged."""

def check_structured_output_limit(output, output_tokens, max_new_tokens):
    if output_tokens >= max_new_tokens:
        try:
            json.loads(output)
        except json.JSONDecodeError as exc:
            raise GenerationLimitError(
                f'incomplete JSON at output limit {max_new_tokens}; no same-budget feedback retry') from exc


class CanvasRenderer:
    def __init__(self, directory, assets=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.assets = assets or {}
        for key, asset in self.assets.items():
            if set(asset) != {'path', 'sha256'} or file_sha(asset['path']) != asset['sha256']:
                raise ValueError(f'unverified image asset: {key}')

    def compile_flow(self, canvas):
        """Keep declared rows/font/content; widen within the existing canvas limits if needed."""
        try:
            return self._compile_flow_at_width(canvas)
        except CanvasSizeError as original:
            for width in (512, 1024, 2048, 4096):
                if width <= canvas['width']:
                    continue
                try:
                    return self._compile_flow_at_width({**canvas, 'width': width})
                except CanvasSizeError:
                    pass
            raise original

    def _compile_flow_at_width(self, canvas):
        """Compile declared rows to editable boxes at the requested font, or reject the whole canvas."""
        from PIL import Image, ImageOps
        if (not isinstance(canvas, dict) or set(canvas) != {'layout', 'width', 'rows'}
                or canvas['layout'] != 'rows' or type(canvas['width']) is not int
                or canvas['width'] not in (512, 1024, 2048, 4096)
                or not isinstance(canvas['rows'], list) or not 1 <= len(canvas['rows']) <= 64):
            raise ValueError('flow canvas requires layout:rows, width, and nonempty rows; no guessed boxes')
        width, top, placed = canvas['width'], 16, []
        for row in canvas['rows']:
            if not isinstance(row, list) or not row or len(row) > 64:
                raise ValueError('flow rows must contain 1..64 items')
            cell_width = (width - 32 - 16 * (len(row) - 1)) // len(row)
            if cell_width < 1:
                raise CanvasSizeError('flow row is too wide; split items into more rows')
            heights = []
            for item in row:
                if not isinstance(item, dict):
                    raise ValueError('invalid flow item')
                required = {'id', 'kind'} | ({'text', 'font_size'} if item.get('kind') == 'text' else {'asset_id'})
                if set(item) != required or item.get('kind') not in ('text', 'image'):
                    raise ValueError('flow items declare text/font_size or asset_id, never coordinates')
                if item['kind'] == 'text':
                    size, text = item['font_size'], item['text']
                    if (type(size) is not int or not 1 <= size <= 96 or not isinstance(text, str)
                            or not text.strip() or len(text) > 16000):
                        raise ValueError('invalid flow text/font')
                    lines = wrap(text, size, cell_width)
                    if lines is None:
                        raise CanvasSizeError('flow text too wide at requested font; change row or split node')
                    boxes = [font(size).getbbox(line, anchor='lt') for line in lines]
                    if any(b[2] - b[0] > cell_width for b in boxes):
                        raise CanvasSizeError('flow glyph exceeds cell width at requested font')
                    heights.append(max(1, max(i * math.ceil(size * 1.4) + b[3] for i, b in enumerate(boxes))))
                else:
                    asset = self.assets[item['asset_id']]
                    if file_sha(asset['path']) != asset['sha256']:
                        raise ValueError('native image changed after admission')
                    with Image.open(asset['path']) as original:
                        original = ImageOps.exif_transpose(original)
                        heights.append(max(1, math.ceil(cell_width * original.height / original.width)))
            height = max(heights)
            for col, item in enumerate(row):
                placed.append((copy.deepcopy(item), 16 + col * (cell_width + 16), top, cell_width, height))
            top += height + 16
        if top > 4096 or len(placed) > 64:
            raise CanvasSizeError(f'flow canvas needs {top}px height/{len(placed)} items; split into child nodes; no clipping')
        # Power-of-two dimensions make integer-pixel edge fractions exactly representable.
        height = 1 << (max(128, top) - 1).bit_length()
        items = []
        for item, x, y, w, h in placed:
            item['box'] = {'left': x / width, 'top': y / height,
                           'right': (x + w) / width, 'bottom': (y + h) / height}
            items.append(item)
        result = {'width': width, 'height': height, 'items': items}
        validate_canvas(result, set(self.assets))
        return result

    def render(self, node):
        from PIL import Image, ImageDraw, ImageOps
        spec = node['canvas']
        coverage = font_coverage([{'state': [{'step': 0, 'title': '',
            'text': '\n'.join(i['text'] for i in spec['items'] if i['kind'] == 'text')}]}])
        if coverage['unmapped_occurrences']:
            missing = ', '.join(i['codepoint'] for i in coverage['unmapped_characters'])
            raise ValueError(f'canvas font has unsupported glyphs: {missing}; no character substitution')
        cache = digest({'node': node, 'font_sha256': coverage['font_sha256'],
                        'assets': {i['asset_id']: self.assets[i['asset_id']]['sha256']
                                   for i in spec['items'] if i['kind'] == 'image'}})
        path = self.directory / f'{cache}.png'
        audit_path = path.with_suffix('.json')
        if path.exists() and audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if file_sha(path) != audit['image_sha256']:
                raise ValueError('cached canvas corrupted')
            return path, audit
        image = Image.new('RGB', (spec['width'], spec['height']), 'white')
        draw, audit = ImageDraw.Draw(image), []
        for item in spec['items']:
            x, y, w, h = canvas_box_pixels(item['box'], image.width, image.height)
            if item['kind'] == 'image':
                asset = self.assets[item['asset_id']]
                if file_sha(asset['path']) != asset['sha256']:
                    raise ValueError('native image changed after admission')
                with Image.open(asset['path']) as source:
                    tile = ImageOps.contain(ImageOps.exif_transpose(source).convert('RGB'), (w, h))
                image.paste(tile, (x + (w - tile.width) // 2, y + (h - tile.height) // 2))
                audit.append({'id': item['id'], 'asset_sha256': asset['sha256'], 'box_px': [x, y, w, h]})
                continue
            # Exact wrapping; never silently truncate an overfull node.
            fit = None
            for size in range(item['font_size'], 0, -1):
                lines = wrap(item['text'], size, w)
                step = math.ceil(size * 1.4)
                if lines is None:
                    continue
                bboxes = [font(size).getbbox(line, anchor='lt') for line in lines]
                if all(b[2] - b[0] <= w and i * step + b[3] <= h for i, b in enumerate(bboxes)):
                    fit = (size, lines, bboxes, step)
                    break
            if fit is None:
                raise ValueError('canvas text does not fit without clipping')
            size, lines, bboxes, step = fit
            for i, (line, box) in enumerate(zip(lines, bboxes)):
                draw.text((x - box[0], y + i * step), line, font=font(size), fill='black', anchor='lt')
            audit.append({'id': item['id'], 'text_sha256': digest(item['text']),
                          'font_size_requested': item['font_size'], 'font_size_rendered': size,
                          'box_px': [x, y, w, h], 'all_characters_rendered': True})
        # Overlap is visible in the audit, not falsely certified as readable or lossless.
        overlap = []
        for i, a in enumerate(audit):
            for b in audit[i + 1:]:
                x, y, w, h = a['box_px']; u, v, s, t = b['box_px']
                if max(x, u) < min(x + w, u + s) and max(y, v) < min(y + h, v + t):
                    overlap.append([a['id'], b['id']])
        image.save(path)
        report = {'items': audit, 'overlapping_items': overlap, 'glyph_coverage': coverage, 'image_sha256': file_sha(path),
                  'pixels': list(image.size), 'readability_verified': False}
        audit_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return path, report


def compile_json_grammar(tokenizer, vocab_size, stop_token_ids, schema):
    import xgrammar as xgr
    from importlib.metadata import version
    started = time.perf_counter()
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size,
                                             stop_token_ids=stop_token_ids)
    # Unrestricted JSON whitespace admitted the real conv43 generation loop.
    # Constrain formatting only; whitespace inside string values remains data.
    grammar = xgr.GrammarCompiler(info, max_threads=4).compile_json_schema(
        schema, any_whitespace=False, indent=None, separators=(',', ':'))
    return grammar, {'engine': 'xgrammar', 'version': version('xgrammar'),
        'schema_sha256': digest(schema), 'vocab_size': vocab_size,
        'json_format': 'compact_outside_strings',
        'stop_token_ids': info.stop_token_ids, 'compile_seconds': time.perf_counter() - started}

class ModelCalls:
    """One frozen backbone, explicit stage calls; optional task adapters are separate factors."""
    def __init__(self, model_path, backend, directory, max_input_tokens=32768, adapter_path=None):
        from memorium.runtime.model_io import Generator
        self.generator = Generator(model_path, backend, 512)
        self.processor, self.model = self.generator.processor, self.generator.model
        self.adapter_path = str(adapter_path) if adapter_path else None
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False).eval()
            self.generator.model = self.model
            self.adapter_hashes = {p.name: file_sha(p) for p in Path(adapter_path).glob('adapter*') if p.is_file()}
            if 'adapter_config.json' not in self.adapter_hashes or 'adapter_model.safetensors' not in self.adapter_hashes:
                raise ValueError('incomplete answerer adapter')
        self.factor = int(self.processor.image_processor.patch_size) * int(self.processor.image_processor.merge_size)
        processor = self.processor.image_processor
        processor.min_pixels, processor.max_pixels = self.factor ** 2, self.factor ** 2 * 16384
        if hasattr(processor, 'size'):
            processor.size = {'shortest_edge': processor.min_pixels, 'longest_edge': processor.max_pixels}
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_input_tokens = max_input_tokens
        self.call_number = len(list(self.directory.glob('*.json')))
        self.grammars = {}

    def inputs(self, prompt, images=()):
        content = [{'type': 'image', 'image': str(path)} for path in images]
        content.append({'type': 'text', 'text': prompt})
        return self.processor.apply_chat_template([{'role': 'user', 'content': content}],
                    add_generation_prompt=True, enable_thinking=False, tokenize=True,
                    return_dict=True, return_tensors='pt')

    def count(self, prompt, images=()):
        return int(self.inputs(prompt, images)['input_ids'].shape[-1])

    def context_limit(self):
        config = getattr(self.model.config, 'text_config', self.model.config)
        native = config.max_position_embeddings
        if type(native) is not int or native <= 0:
            raise ValueError('model must declare a positive native context capacity')
        configured = self.max_input_tokens
        if configured is not None and (type(configured) is not int or configured <= 0):
            raise ValueError('input cap must be a positive integer or None for native capacity')
        return native if configured is None else min(native, configured)

    def call(self, stage, prompt, images=(), max_new_tokens=512, json_schema=None):
        import torch
        if getattr(self, 'adapter_path', None) and stage != 'answerer':
            raise ValueError('answerer adapter cannot modify Writer or Selector calls')
        started = time.perf_counter()
        self.call_number += 1
        row = {'stage': stage, 'call': self.call_number, 'prompt': prompt,
               'images': [{'path': str(p), 'sha256': file_sha(p)} for p in images],
               'max_new_tokens': max_new_tokens,
               'weights': 'pretrained_frozen_no_task_adapter', 'status': 'STARTED'}
        if getattr(self, 'adapter_path', None):
            row.update(weights='answerer_lora', adapter_path=self.adapter_path, adapter_hashes=self.adapter_hashes)
        record = self.directory / f'{self.call_number:08d}.json'
        try:
            inputs = self.inputs(prompt, images)
            total = int(inputs['input_ids'].shape[-1])
            vision = int(inputs['input_ids'].eq(self.model.config.image_token_id).sum())
            row.update(input_tokens=total, input_vision_tokens=vision, input_other_tokens=total - vision,
                       image_grid_thw=inputs.get('image_grid_thw', torch.tensor([])).tolist())
            context_limit = self.context_limit()
            row.update(input_limit_policy='native_context_capacity_only' if self.max_input_tokens is None else 'explicit_cap',
                       effective_context_tokens=context_limit)
            if max_new_tokens is None:
                if stage != 'memorier':
                    raise ValueError('uncapped output is only enabled for the Writer diagnostic')
                max_new_tokens = self.model.config.text_config.max_position_embeddings - total
                row.update(max_new_tokens=max_new_tokens, output_limit_policy='native_context_capacity_only',
                           native_context_tokens=self.model.config.text_config.max_position_embeddings)
                if max_new_tokens <= 0:
                    raise ValueError('source already fills native model context; no truncation')
            if type(max_new_tokens) is not int or max_new_tokens <= 0:
                raise ValueError('output cap must be a positive integer')
            if row.get('output_limit_policy') != 'native_context_capacity_only' and total + max_new_tokens > context_limit:
                raise ValueError('model context cap exceeded; no silent truncation')
            record.write_text(json.dumps(row, ensure_ascii=False))
            generation = {}
            schema = json_schema if json_schema is not None else (writer_json_schema() if stage == 'memorier' else None)
            if schema is not None:
                from xgrammar.contrib.hf import LogitsProcessor
                signature = digest(schema)
                cached = signature in self.grammars
                if not cached:
                    self.grammars[signature] = compile_json_grammar(
                        self.generator.tokenizer, int(self.model.get_output_embeddings().weight.shape[0]),
                        self.model.generation_config.eos_token_id, schema)
                grammar, decoding = self.grammars[signature]
                row['structured_decoding'] = {**decoding, 'cache_hit': cached,
                    'mask_device': 'cpu',
                    'compile_seconds_this_call': 0.0 if cached else decoding['compile_seconds']}
                # The compiled grammar is reusable; its stateful HF processor is not.
                processor = LogitsProcessor(grammar)
                # ponytail: CPU masking transfers logits per token; use CUDA after its compiler is provisioned.
                generation['logits_processor'] = [lambda input_ids, scores:
                    processor(input_ids, scores.cpu()).to(scores.device)]
                record.write_text(json.dumps(row, ensure_ascii=False))
            if stage == 'memorier':
                from transformers import StoppingCriteria, StoppingCriteriaList
                class RecordProgress(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        count = int(input_ids.shape[-1]) - total
                        if count % 128 == 0:
                            row.update(output_tokens_observed=count,
                                       partial_output_token_ids=input_ids[0][total:].tolist(),
                                       partial_output=self_processor.decode(input_ids[0][total:], skip_special_tokens=True),
                                       generation_elapsed_seconds=time.perf_counter() - started)
                            record.write_text(json.dumps(row, ensure_ascii=False))
                        return False
                # Observe censored output on timeout; never stop or alter token selection here.
                self_processor = self.processor
                generation['stopping_criteria'] = StoppingCriteriaList([RecordProgress()])
            with torch.inference_mode():
                generated = self.model.generate(**inputs.to(self.model.device), do_sample=False,
                                                max_new_tokens=max_new_tokens, **generation)
            row['output_tokens'] = int(generated.shape[-1]) - total
            row['output_token_ids'] = generated[0][total:].tolist()
            row['output'] = self.processor.decode(generated[0][total:], skip_special_tokens=True).strip()
            eos = self.model.generation_config.eos_token_id
            stopped = int(generated[0][-1]) in (eos if isinstance(eos, list) else [eos])
            row['finish_reason'] = 'eos' if stopped else ('length' if row['output_tokens'] >= max_new_tokens else 'stopped')
            if schema is not None:
                check_structured_output_limit(row['output'], row['output_tokens'], max_new_tokens)
            row['status'] = 'OK'
        except Exception as exc:
            row.update(status='ERROR', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            row['elapsed_seconds'] = time.perf_counter() - started
            record.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True))
        return row
