"""AI-QA-Assistant V2.2 — WPS 业务场景模拟质检求职演示版。

运行：python -m streamlit run app.py
依赖：streamlit pandas openpyxl（模型服务：本机 Ollama）
注意：这是合成数据演示工具，不代表金山办公内部质检规则。
"""

import io
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:4b"
CATEGORIES = ["用户诉求识别与问题处理", "服务态度与沟通", "专业准确性与表达", "风险与流程合规"]
ISSUE_LABELS = [
    "答非所问", "未解决诉求", "无效追问", "推诿责任", "缺乏共情", "态度冷漠",
    "侮辱用户", "解释不清", "未经核实的产品说明", "不合理承诺", "隐私风险", "升级处理缺失",
]
SYSTEM_PROMPT = """你是办公软件用户服务的模拟质检员。只能依据可见客服对话做初判，不得声称掌握公司内部政策。
务必返回一个JSON对象，不输出Markdown、思考内容或解释。
对以下四维分别给0到25的整数分：
1. 用户诉求识别与问题处理：理解问题、合理追问、提供解决/升级路径。
2. 服务态度与沟通：礼貌、耐心、共情；仅当存在明显冷漠、冒犯、指责等语气证据时扣分。绝对化承诺本身不等于态度差。
3. 专业准确性与表达：说明清楚、区分事实与推测；缺少官方依据时不要断言退款/功能政策。
4. 风险与流程合规：隐私保护、承诺边界、必要升级。不要把不合理承诺当作隐私泄露；没有明确索取密码、验证码、私人信息或泄露信息的证据，不得输出“隐私风险”。
各维分数锚点：23-25优秀，19-22基本合格，13-18中度问题，6-12明显问题，0-5严重问题。
不要凭空推断后台情况、退款条件、文件能否恢复或员工权限。客服合理询问账号类型、版本不应自动扣分。
证据evidence只能是输入中连续出现的原文片段，尽可能逐字复制（含原标点）；未发现问题时写“未发现明确证据”。
严重违规初判critical_violation可针对辱骂威胁、索要密码/验证码、泄露隐私、未经核实保证文件必能恢复或付款必到账等。
只有严重违规候选、重要政策无法核实、重大误导或确实需要人工介入时，才设置requires_review=true。一般正常服务设置false。
JSON结构：
{
  "summary":"总体评价",
  "items":[{"category":"用户诉求识别与问题处理","score":20,"problem":"问题或未发现明显问题","evidence":"原文","suggestion":"建议"}],
  "issue_labels":["不合理承诺"],
  "critical_violation":false,
  "risk_reason":"风险原因",
  "requires_review":false,
  "uncertainties":["需要核实的外部政策或事实"]
}
items必须恰好包含四项，顺序与定义一致；issue_labels只能取上述常用类别中的适用项。
""" + "允许的问题标签：" + "、".join(ISSUE_LABELS)


def extract_json(text: str) -> dict:
    """提取JSON，严格校验维度与可显示的证据，不接受编造的原文。"""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        a, b = cleaned.find("{"), cleaned.rfind("}")
        if a < 0 or b <= a:
            raise ValueError("模型没有返回JSON对象")
        data = json.loads(cleaned[a : b + 1])
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError("JSON缺少items数组")
    raw_items = data["items"]
    cats = [i.get("category") for i in raw_items if isinstance(i, dict)]
    if len(raw_items) != 4 or sorted(cats) != sorted(CATEGORIES):
        raise ValueError(f"维度不完整或重复：{cats}，请重新运行")
    for item in raw_items:
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 25:
            raise ValueError(f"{item['category']}的分数不是0-25整数")
    data["items"] = sorted(raw_items, key=lambda item: CATEGORIES.index(item["category"]))
    data["total_score"] = sum(i["score"] for i in data["items"])
    data["critical_violation"] = data.get("critical_violation") is True
    labels = data.get("issue_labels", [])
    data["issue_labels"] = list(dict.fromkeys(v for v in labels if v in ISSUE_LABELS)) if isinstance(labels, list) else []
    data["summary"] = str(data.get("summary", ""))
    data["risk_reason"] = str(data.get("risk_reason", ""))
    un = data.get("uncertainties", [])
    data["uncertainties"] = [str(x) for x in un] if isinstance(un, list) else []
    if data["critical_violation"]:
        data["risk_level"] = "高风险（待复核）"
    elif data["total_score"] < 60:
        data["risk_level"] = "高风险（待复核）"
    elif data["total_score"] < 80:
        data["risk_level"] = "中风险"
    else:
        data["risk_level"] = "低风险"
    # 复核条件以可核验风险为主，避免模型给每条正常记录都设为True。
    data["requires_review"] = data["critical_violation"] or data["total_score"] < 60
    return data


def compact_text(value: str) -> str:
    """仅为核对原文引用容忍空白与常见中文标点差异，不改变展示的证据。"""
    return re.sub(r"[\s，。！？、；：,.!?;:'‘’“”\"（）()]", "", value)


def privacy_evidence(conversation: str) -> bool:
    """保守筛查明确的密码、验证码、身份信息索取/披露场景。"""
    return bool(re.search(r"密码|验证码|身份证号|手机号|私人文档|隐私|公开群|公开评论区", conversation))


def validate_evidence(result: dict, conversation: str) -> dict:
    bad = []
    # 明确无隐私相关对话证据时，删除模型误加的隐私风险标签。
    if not privacy_evidence(conversation):
        result["issue_labels"] = [x for x in result["issue_labels"] if x != "隐私风险"]
    for item in result["items"]:
        for field in ("problem", "evidence", "suggestion"):
            item[field] = str(item.get(field, ""))
        quote = item["evidence"].strip().strip('“”"')
        if quote and quote != "未发现明确证据" and quote not in conversation and compact_text(quote) not in compact_text(conversation):
            bad.append(item["category"])
            item["evidence"] = f"[引用未通过原文校验] {quote}"
        # 不把缺少隐私证据的泛化问题判断展示成已证实隐私事件。
        if not privacy_evidence(conversation) and "隐私风险" in item["problem"]:
            item["problem"] = item["problem"].replace("隐私风险", "需核实的业务风险")
    result["invalid_evidence_categories"] = bad
    if bad:
        result["requires_review"] = True
    # 风险为中/低且引用有效时，不因为模型任意的复核偏好全量进入队列。
    # 如后续接入业务知识库，可在此加入经核实的政策风险规则。
    return result


def call_ollama(conversation: str, model: str) -> dict:
    """结构化输出错误自动重试一次；连接失败不重试，以免长时间阻塞。"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "请质检以下模拟客服对话：\n【开始】\n" + conversation + "\n【结束】"},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    first_error = None
    for attempt in range(1, 3):
        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                reply = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"连接Ollama失败，请确认服务已启动：{exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("Ollama超时（240秒）") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama服务返回了无效的HTTP响应JSON") from exc

        content = (reply.get("message") or {}).get("content") or ""
        try:
            result = validate_evidence(extract_json(content), conversation)
            result["attempts"] = attempt
            result["retry_reason"] = first_error or ""
            return result
        except (ValueError, json.JSONDecodeError) as exc:
            if attempt == 2:
                raise ValueError(
                    f"结构化输出连续两次失败；第一次：{first_error}；第二次：{exc}"
                ) from exc
            first_error = str(exc)
            # 仅对第二次请求加强格式提醒；原始对话和质检规则保持一致。
            payload["messages"].append({
                "role": "user",
                "content": "上次输出结构有误：" + first_error
                + "。请重新独立分析原对话。items必须恰好四项，category依次且各出现一次："
                + "、".join(CATEGORIES) + "。各项score为0到25的整数，只输出合法JSON。",
            })
    raise RuntimeError("质检未产生结果")


def result_to_row(meta: dict, result: dict, elapsed: float) -> dict:
    row = dict(meta)
    row.update({
        "total_score": result["total_score"],
        "risk_level": result["risk_level"],
        "critical_violation": result["critical_violation"],
        "requires_review": result["requires_review"],
        "issue_labels": "；".join(result["issue_labels"]),
        "summary": result["summary"],
        "risk_reason": result["risk_reason"],
        "invalid_evidence": "；".join(result["invalid_evidence_categories"]),
        "elapsed_seconds": round(elapsed, 2),
        "attempts": result.get("attempts", 1),
        "retry_reason": result.get("retry_reason", ""),
        "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result_json": json.dumps(result, ensure_ascii=False),
        "status": "成功",
        "error": "",
    })
    for item in result["items"]:
        row[item["category"] + "_score"] = item["score"]
    return row


def load_batch(uploaded) -> pd.DataFrame:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded, dtype=str, encoding="utf-8-sig")
    else:
        df = pd.read_excel(uploaded, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"conversation_id", "conversation"}
    if not required.issubset(df.columns):
        raise ValueError("缺少必要列：" + "、".join(sorted(required - set(df.columns))))
    df = df.fillna("")
    df["conversation_id"] = df["conversation_id"].str.strip()
    if (df["conversation_id"] == "").any() or (df["conversation"].str.strip() == "").any():
        raise ValueError("存在空编号或空对话，请先处理")
    if df["conversation_id"].duplicated().any():
        raise ValueError("conversation_id存在重复，请先处理")
    if len(df) > 100:
        raise ValueError("演示版单次最多100条；建议先用6条测试")
    return df


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="质检结果")
    return out.getvalue()


def create_report(df: pd.DataFrame) -> str:
    ok = df[df["status"] == "成功"].copy()
    if ok.empty:
        return "暂无成功质检记录，无法生成报告。"
    labels = ok["issue_labels"].fillna("").str.split("；").explode()
    counts = labels[labels != ""].value_counts().head(5)
    high = int(ok["risk_level"].str.contains("高风险").sum())
    business = ok.groupby("business_type")["total_score"].mean().sort_values() if "business_type" in ok else None
    lines = [
        "# 客服服务质量分析报告（模拟数据）", "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M}", "数据声明：全部数据为模拟客服对话，不代表真实企业服务表现。", "",
        "## 一、质量概况", f"- 成功质检：{len(ok)}条", f"- 平均得分：{ok['total_score'].mean():.1f}/100",
        f"- 高风险候选：{high}条（{high/len(ok):.1%}）", "",
        "## 二、主要问题", 
    ]
    lines.extend([f"- {label}：{int(n)}次" for label, n in counts.items()] or ["- 本批未识别出明确问题标签"])
    if business is not None:
        lines.extend(["", "## 三、业务场景平均得分"])
        lines.extend(f"- {k}：{v:.1f}分" for k, v in business.items())
    lines.extend([
        "", "## 四、整改建议（待人工确认）",
        "1. 复核高风险候选案例，核对原文与产品政策；不得直接把模型初判当作正式定责。",
        "2. 将高频问题对应到具体服务话术、帮助文档或升级流程，安排针对性培训。",
        "3. 后续使用同一套标签和样本口径追踪变化；模拟数据的变化不能证明真实改进效果。",
        "", "## 五、数据限制",
        "本报告仅依据输入对话和模型初判生成；暂无客服后台、真实政策核对或独立人工标注。",
    ])
    return "\n".join(lines)


def display_result(result: dict):
    c1, c2, c3 = st.columns(3)
    c1.metric("综合得分", f"{result['total_score']}/100")
    c2.metric("风险初判", result["risk_level"])
    c3.metric("严重违规候选", "是" if result["critical_violation"] else "否")
    st.write(result["summary"])
    if result["requires_review"]:
        st.warning("该记录建议人工复核，模型初判不等于最终业务结论。")
    if result["risk_reason"]:
        st.markdown("**风险依据：** " + result["risk_reason"])
    st.markdown("**问题标签：** " + ("、".join(result["issue_labels"]) or "未识别"))
    for item in result["items"]:
        with st.expander(f"{item['category']}｜{item['score']}/25", expanded=item["score"] < 19):
            st.write("问题：", item["problem"])
            st.write("证据：", item["evidence"])
            st.write("建议：", item["suggestion"])
    if result["invalid_evidence_categories"]:
        st.error("以下维度的原文引用校验失败：" + "、".join(result["invalid_evidence_categories"]))
    if result["uncertainties"]:
        st.info("待核实：" + "；".join(result["uncertainties"]))


st.set_page_config(page_title="AI-QA-Assistant V2.2", page_icon="🧾", layout="wide")
st.title("AI-QA-Assistant｜智能客服质检与质量分析")
st.caption("办公软件支持场景 · 本地Ollama模型 · 合成数据求职演示｜非企业内部系统")
with st.sidebar:
    model_name = st.text_input("Ollama模型名", value=DEFAULT_MODEL)
    st.info("先运行 Ollama，并确认模型已下载。批量模式逐条调用模型；输出格式错误时自动重试一次。")
    st.caption("提示：模型判断可能出错，政策、退款和隐私事件必须人工复核。")

single, batch, dashboard, report_tab = st.tabs(["单条质检", "Excel批量质检", "质量分析看板", "质量报告"])

with single:
    example = "客户：会员扣费后怎么申请核实？\n客服：肯定今天就能退款，不需要审核。\n客户：能保证吗？\n客服：百分之百保证。"
    conversation = st.text_area("客服对话", value=example, height=180)
    if st.button("开始单条质检", type="primary"):
        if not conversation.strip():
            st.warning("请先输入对话")
        else:
            try:
                with st.spinner("正在调用本地模型..."):
                    t0 = time.perf_counter()
                    answer = call_ollama(conversation, model_name)
                    elapsed = time.perf_counter() - t0
                st.session_state["single_result"] = answer
                st.session_state["single_record"] = result_to_row(
                    {"conversation_id":"SINGLE", "conversation":conversation, "business_type":"手动输入"}, answer, elapsed
                )
            except Exception as exc:
                st.error(f"质检失败：{exc}")
    if "single_result" in st.session_state:
        display_result(st.session_state["single_result"])
        st.download_button("下载单条JSON", json.dumps(st.session_state["single_record"], ensure_ascii=False, indent=2), "single_qa.json", "application/json")

with batch:
    st.markdown("Excel至少包含 `conversation_id` 和 `conversation` 两列；可包含 `agent_id`、`date`、`business_type`。")
    uploaded = st.file_uploader("选择Excel或CSV", type=["xlsx", "csv"])
    if uploaded is not None:
        try:
            source_df = load_batch(uploaded)
            st.success(f"校验通过：共{len(source_df)}条")
            st.dataframe(source_df.head(10), use_container_width=True)
            max_rows = st.number_input("本次处理前N条（建议先选6）", min_value=1, max_value=len(source_df), value=min(6,len(source_df)))
            if st.button("执行批量质检", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                rows = []
                for idx, (_, record) in enumerate(source_df.head(int(max_rows)).iterrows()):
                    meta = record.to_dict()
                    status.write(f"正在处理 {idx+1}/{max_rows}：{meta['conversation_id']}")
                    start = time.perf_counter()
                    try:
                        result = call_ollama(meta["conversation"], model_name)
                        rows.append(result_to_row(meta, result, time.perf_counter()-start))
                    except Exception as exc:
                        rows.append({**meta, "status":"失败", "error":str(exc), "elapsed_seconds":round(time.perf_counter()-start,2)})
                    progress.progress((idx+1)/int(max_rows))
                st.session_state["batch_df"] = pd.DataFrame(rows)
                status.success("本批处理结束。失败条目会保留错误信息，便于排查。")
        except Exception as exc:
            st.error(f"文件或批量处理失败：{exc}")
    if "batch_df" in st.session_state:
        results_df = st.session_state["batch_df"]
        st.dataframe(results_df.drop(columns=["result_json"], errors="ignore"), use_container_width=True)
        st.download_button("下载完整质检结果（Excel）",to_excel_bytes(results_df),"qa_batch_results.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with dashboard:
    if "batch_df" not in st.session_state:
        st.info("请先在批量质检页面生成数据。")
    else:
        data = st.session_state["batch_df"]
        ok = data[data["status"] == "成功"].copy()
        if ok.empty:
            st.warning("当前没有成功质检的记录")
        else:
            high = int(ok["risk_level"].str.contains("高风险").sum())
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("成功质检",len(ok))
            c2.metric("平均分",f"{ok['total_score'].mean():.1f}")
            c3.metric("高风险候选占比",f"{high/len(ok):.1%}")
            c4.metric("待人工复核",int(ok["requires_review"].sum()))
            l,r = st.columns(2)
            with l:
                st.subheader("风险分布")
                st.bar_chart(ok["risk_level"].value_counts())
            with r:
                st.subheader("问题标签频次")
                labels = ok["issue_labels"].fillna("").str.split("；").explode()
                labels = labels[labels != ""]
                if len(labels): st.bar_chart(labels.value_counts())
                else: st.info("暂无问题标签")
            if "business_type" in ok:
                st.subheader("不同业务场景平均分")
                st.bar_chart(ok.groupby("business_type")["total_score"].mean())
            if "agent_id" in ok:
                st.subheader("模拟客服平均分（请结合样本量解读）")
                st.dataframe(ok.groupby("agent_id").agg(样本量=("total_score","size"),平均分=("total_score","mean")).round(1),use_container_width=True)

with report_tab:
    if "batch_df" not in st.session_state:
        st.info("先运行批量质检，才能生成报告。")
    else:
        report = create_report(st.session_state["batch_df"])
        st.markdown(report)
        st.download_button("下载质量报告（Markdown）",report,"qa_quality_report.md","text/markdown")
