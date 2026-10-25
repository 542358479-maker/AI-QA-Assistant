"""V2.2 结构化输出重试测试（不需要Streamlit/Ollama）。"""
import ast
import json
import re
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch


source = ast.parse(Path(__file__).with_name("app.py").read_text(encoding="utf-8"))
names = {"extract_json", "compact_text", "privacy_evidence", "validate_evidence", "call_ollama"}
funcs = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
cats = ["用户诉求识别与问题处理", "服务态度与沟通", "专业准确性与表达", "风险与流程合规"]
ns = {"json":json,"re":re,"urllib":urllib,"CATEGORIES":cats,
      "ISSUE_LABELS":["不合理承诺","隐私风险"],"OLLAMA_URL":"http://localhost:11434/api/chat",
      "SYSTEM_PROMPT":"测试质检规则"}
exec(compile(ast.Module(body=funcs,type_ignores=[]),"app_v2_2.py","exec"),ns)


def model_json(categories):
    return json.dumps({"summary":"测试", "items":[{"category":c,"score":20,"problem":"未发现明显问题",
            "evidence":"未发现明确证据","suggestion":"无"} for c in categories],
            "issue_labels":[],"critical_violation":False,"risk_reason":"", "requires_review":False},ensure_ascii=False)


class MockHTTPResponse:
    def __init__(self, model_content):
        self.data = json.dumps({"message":{"content":model_content}},ensure_ascii=False).encode()
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def read(self): return self.data


class RetryTests(unittest.TestCase):
    def test_retry_duplicate_dimensions_then_success(self):
        outputs = [model_json([cats[0],cats[0],cats[1],cats[1]]),model_json(cats)]
        def fake_open(req,timeout):
            self.assertEqual(timeout,240)
            return MockHTTPResponse(outputs.pop(0))
        with patch("urllib.request.urlopen",side_effect=fake_open) as opener:
            result = ns["call_ollama"]("客户：你好。\n客服：您好。","qwen3.5:4b")
        self.assertEqual(opener.call_count,2)
        self.assertEqual(result["attempts"],2)
        self.assertEqual(result["total_score"],80)
        self.assertIn("维度不完整或重复",result["retry_reason"])

    def test_success_needs_no_retry(self):
        with patch("urllib.request.urlopen",return_value=MockHTTPResponse(model_json(cats))) as opener:
            result=ns["call_ollama"]("客服：您好。","qwen3.5:4b")
        self.assertEqual(opener.call_count,1)
        self.assertEqual(result["attempts"],1)

    def test_two_bad_outputs_fail_with_reason(self):
        invalid=model_json([cats[0],cats[0],cats[1],cats[1]])
        with patch("urllib.request.urlopen",return_value=MockHTTPResponse(invalid)) as opener:
            with self.assertRaisesRegex(ValueError,"连续两次失败"):
                ns["call_ollama"]("客服：您好。","qwen3.5:4b")
        self.assertEqual(opener.call_count,2)

    def test_connection_error_no_retry(self):
        with patch("urllib.request.urlopen",side_effect=urllib.error.URLError("offline")) as opener:
            with self.assertRaisesRegex(RuntimeError,"连接Ollama失败"):
                ns["call_ollama"]("客服：您好。","qwen3.5:4b")
        self.assertEqual(opener.call_count,1)


if __name__=="__main__": unittest.main()
