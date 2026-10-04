"""Pure helpers shared by the registered CoT experiment."""
import json


def completed_indices(path):
    """Recover only an interrupted last line; reject corrupted or duplicate records."""
    if not path.exists():
        return set()
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    indices = set()
    offset = 0
    for number, line in enumerate(lines):
        try:
            item = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if number == len(lines)-1 and not line.endswith(b'\n'):
                with path.open('r+b') as stream:
                    stream.truncate(offset)
                break
            raise ValueError('CORRUPT_PREDICTION_LINE') from None
        index = item.get('index')
        if not isinstance(index, int) or isinstance(index, bool) or index in indices:
            raise ValueError('INVALID_OR_DUPLICATE_INDEX')
        if not line.endswith(b'\n'):
            with path.open('ab') as stream:
                stream.write(b'\n')
        indices.add(index)
        offset += len(line)
    return indices


def neutral_instruction(original):
    text = original.replace(
        'Analyze the mammography image and return ONLY valid JSON.',
        'Analyze the mammography image. Your final answer must be valid JSON.')
    text = text.replace('Return ONLY a JSON object with exactly these keys:',
                        'Your final answer must be a JSON object with exactly these keys:')
    text = text.replace('Do not output explanations, markdown, code fences, or extra text.',
                        'The final answer must contain only the JSON object, without markdown or code fences.')
    if text == original or 'ONLY valid JSON' in text or 'Do not output explanations' in text:
        raise ValueError('UNRECOGNIZED_DIRECT_PROMPT')
    return text


def transform_instruction(original, mode):
    if mode == 'original':
        return original
    if mode == 'neutral':
        return neutral_instruction(original)
    if mode == 'thinking':
        from distill_thinking import thinking_instruction
        return thinking_instruction(original)
    raise ValueError('UNKNOWN_INSTRUCTION_MODE')
