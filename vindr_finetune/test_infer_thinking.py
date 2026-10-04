import ast
import json
from pathlib import Path
import re
import unittest

# Test the pure parser without importing the GPU runtime on the Mac.
source=ast.parse(Path(__file__).with_name('infer_lora.py').read_text())
function=next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name=='parse_prediction')
namespace={'json':json,'THINK_RE':re.compile(r'<think>.*?</think>',re.S),
           'FENCE_RE':re.compile(r'```(?:json)?\s*(.*?)```',re.S)}
exec(compile(ast.Module(body=[function],type_ignores=[]),'<parser>','exec'),namespace)
parse=namespace['parse_prediction']


class ThinkingParserTests(unittest.TestCase):
    def test_only_final_answer_is_scored(self):
        raw='<think>Candidate {"breast_birads":5}</think>\n{"breast_birads":1}'
        self.assertEqual(parse(raw,True),{'breast_birads':1})
        self.assertEqual(parse('Candidate {"breast_birads":5}</think>\n{"breast_birads":1}',True),{'breast_birads':1})

    def test_truncated_reasoning_never_becomes_a_prediction(self):
        self.assertIsNone(parse('<think>Candidate {"breast_birads":5}',True))
        self.assertIsNone(parse('<think>Candidate {"breast_birads":5}',False))
        self.assertEqual(parse('Candidate {"breast_birads":5}</think>\n{"breast_birads":1}',False),{'breast_birads':1})
        self.assertIsNone(parse('Reasoning candidate {"breast_birads":5}',True))
        self.assertEqual(parse('{"breast_birads":1}',True),{'breast_birads':1})

    def test_previous_non_thinking_parser_remains_compatible(self):
        self.assertEqual(parse('```json\n{"breast_birads":1}\n```'),{'breast_birads':1})


if __name__=='__main__':unittest.main()
