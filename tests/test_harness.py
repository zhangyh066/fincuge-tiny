# -*- coding: utf-8 -*-
import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from harness import Harness


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(str(text))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(str(i) for i in ids)


class HarnessAnswerTest(unittest.TestCase):
    def setUp(self):
        self.h = Harness.__new__(Harness)
        self.h.tok = FakeTokenizer()
        self.h.versions = {name: f"{name}-test" for name in ("emotion", "newscat", "entity")}
        self.h.used = []
        self.h.logs = []
        self.h._use = lambda task: self.h.used.append(task)
        self.h._log = lambda res, body: self.h.logs.append((res, body))

    def fake_generate(self, result):
        def generate(h, row, max_new):
            h.last_row = row
            return result
        self.h._generate = types.MethodType(generate, self.h)

    def test_entity_requires_event_type_before_inference(self):
        res = self.h.answer("甲公司被查", task_hint="entity")
        self.assertEqual(res["status"], "need_input")
        self.assertEqual(res["missing"], ["event_type"])
        self.assertEqual(res["generation_count"], 0)
        self.assertEqual(res["attempts"], 0)
        self.assertIsNone(res["schema_valid"])
        self.assertEqual(self.h.used, [])

    def test_no_event_type_returns_no_without_inference(self):
        res = self.h.answer("甲公司被查", task_hint="entity", event_type="无")
        self.assertEqual(res["status"], "no_event")
        self.assertEqual(res["answer"], "无")
        self.assertEqual(res["decision_reason"], "no_applicable_event")
        self.assertEqual(res["missing"], [])
        self.assertEqual(res["generation_count"], 0)
        self.assertEqual(self.h.used, [])

    def test_event_type_builds_training_instruction(self):
        self.fake_generate(("信批违规事件的金融主体是甲公司。", 0.95))
        res = self.h.answer("甲公司被查", task_hint="entity", event_type="信批违规")
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["answer"], "甲公司")
        self.assertEqual(res["instruction_source"], "event_type")
        self.assertNotIn("instruction_synthesized", res["flags"])
        self.assertEqual(self.h.last_row["instruction"], "分析以下新闻，抽取信批违规事件相关的主体信息。")
        self.assertEqual(self.h.last_row["input"], "甲公司被查")

    def test_route_text_is_not_added_to_model_input(self):
        def emotion(h, row):
            h.last_row = row
            return "消极", 0.96, {"积极": 0.01, "消极": 0.96, "中性": 0.03}
        self.h._emotion = types.MethodType(emotion, self.h)
        body = "这家公司业绩暴雷，股价要崩"
        res = self.h.answer(body, route_text="判断这段话的情绪")
        self.assertEqual(res["task"], "emotion")
        self.assertEqual(res["answer"], "消极")
        self.assertEqual(self.h.last_row["input"], body)
        self.assertEqual(self.h.last_row["instruction"], "判断以下句子的情感属性。")

    def test_bare_document_returns_need_task(self):
        res = self.h.answer("这家公司业绩暴雷，股价要崩")
        self.assertEqual(res["status"], "need_task")
        self.assertEqual(res["missing"], ["task"])
        self.assertEqual(res["generation_count"], 0)

    def test_unanimous_vote_is_hint_and_records_four_generations(self):
        self.fake_generate(("信批违规事件的金融主体是甲公司。", 0.80))

        def vote(h, row, max_new):
            detail = {
                "samples": [{"entity": "甲公司", "raw": "主体是甲公司。"}] * 3,
                "votes": {"甲公司": 3},
            }
            return "甲公司", 1.0, "主体是甲公司。", detail
        self.h._generate_vote = types.MethodType(vote, self.h)
        res = self.h.answer("甲公司被查", task_hint="entity", event_type="信批违规")
        self.assertEqual(res["decision"], "hint")
        self.assertEqual(res["decision_reason"], "vote_agreement_uncalibrated")
        self.assertEqual(res["confidence_kind"], "vote_agreement")
        self.assertEqual(res["token_confidence"], 0.80)
        self.assertEqual(res["generation_count"], 4)
        self.assertEqual(res["attempts"], 4)
        self.assertEqual(res["vote"]["votes"], {"甲公司": 3})
        self.assertEqual(res["raw"], "主体是甲公司。")

    def test_completed_and_rejected_responses_have_same_fields(self):
        self.fake_generate(("信批违规事件的金融主体是甲公司。", 0.95))
        completed = self.h.answer("甲公司被查", task_hint="entity", event_type="信批违规")
        rejected = self.h.answer("甲公司被查", task_hint="entity")
        self.assertEqual(set(completed), set(rejected))

    def test_request_ids_are_unique(self):
        first = self.h.answer("甲公司被查", task_hint="entity")
        second = self.h.answer("甲公司被查", task_hint="entity")
        self.assertNotEqual(first["request_id"], second["request_id"])


class VoteDetailTest(unittest.TestCase):
    def test_winning_raw_is_not_simply_last_sample(self):
        h = Harness.__new__(Harness)
        h._encode = lambda row: torch.tensor([[0]])
        h.tok = FakeTokenizer()
        h.tok.decode = lambda ids, skip_special_tokens=True: {1: "甲公司", 2: "甲公司", 3: "乙公司"}[int(ids[0])]

        class Model:
            def __init__(self):
                self.outputs = iter([1, 2, 3])

            def generate(self, ids, **kwargs):
                return torch.tensor([[0, next(self.outputs)]])

        h.model = Model()
        entity, conf, raw, detail = h._generate_vote({"input": "甲公司、乙公司"}, 24)
        self.assertEqual(entity, "甲公司")
        self.assertEqual(conf, 2 / 3)
        self.assertEqual(raw, "甲公司")
        self.assertEqual(detail["votes"], {"甲公司": 2, "乙公司": 1})
        self.assertEqual([s["raw"] for s in detail["samples"]], ["甲公司", "甲公司", "乙公司"])


if __name__ == "__main__":
    unittest.main()
