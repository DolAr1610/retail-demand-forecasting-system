from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.backend.services.loader import ArtifactLoader
from app.backend.services.predictor import PredictorService
from app.backend.services.llm_service import LLMService
from app.backend.prompts.render import render_prompt


@dataclass
class ParsedAgentContext:
    store_nbr: Optional[int] = None
    item_nbr: Optional[int] = None
    item_alias: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    model_name: Optional[str] = None
    current_stock: Optional[float] = None
    safety_stock: Optional[float] = None
    lead_time_days: Optional[int] = None


class AgentService:
    def __init__(self, loader: ArtifactLoader, predictor: PredictorService, llm: Optional[LLMService] = None):
        self.loader = loader
        self.predictor = predictor
        self.llm = llm

    async def chat(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        history = history or []

        parsed = self._extract_context(message=message, context=context)

        if not self.llm or not self.llm.is_enabled:
            fallback = self._fallback_without_llm(message, parsed)
            return fallback

        
        system_prompt = render_prompt("agent_system_prompt.j2")
        tools = self._build_tools_schema()

        user_prompt = render_prompt(
            "agent_user_prompt.j2",
            user_message=message,
            resolved_context_json=json.dumps(
                {
                    "store_nbr": parsed.store_nbr,
                    "item_nbr": parsed.item_nbr,
                    "item_alias": parsed.item_alias,
                    "date_from": parsed.date_from,
                    "date_to": parsed.date_to,
                    "model_name": parsed.model_name,
                    "current_stock": parsed.current_stock,
                    "safety_stock": parsed.safety_stock,
                    "lead_time_days": parsed.lead_time_days,
                },
                ensure_ascii=False,
                indent=2,
            ),
            available_models_json=json.dumps(
                self.loader.load_available_model_names(),
                ensure_ascii=False,
                indent=2,
            ),
            default_model=self.loader.load_default_model_name(),
        )

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        for h in history[-10:]:
            role = h.get("role", "user")
            content = h.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_prompt})

        used_tools: List[str] = []
        tool_results: List[Dict[str, Any]] = []

        # up to 4 tool rounds
        for _ in range(4):
            response = await self.llm.create_chat_completion(
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=1200,
            )

            choice = response["choices"][0]["message"]
            assistant_message: Dict[str, Any] = {
                "role": "assistant",
            }

            if "content" in choice and choice["content"] is not None:
                assistant_message["content"] = choice["content"]
            else:
                assistant_message["content"] = ""

            if "tool_calls" in choice and choice["tool_calls"]:
                assistant_message["tool_calls"] = choice["tool_calls"]
                messages.append(assistant_message)

                for tool_call in choice["tool_calls"]:
                    fn_name = tool_call["function"]["name"]
                    raw_args = tool_call["function"].get("arguments", "{}")

                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = {}

                    tool_result = self._execute_tool(fn_name, args, parsed)
                    used_tools.append(fn_name)
                    tool_results.append({
                        "tool": fn_name,
                        "args": args,
                        "result": tool_result,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": fn_name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })

                continue

            raw_content = choice.get("content")
            final_answer = raw_content.strip() if isinstance(raw_content, str) else ""

            if not final_answer:
                final_answer = "I could not produce a final answer from the available tool results."

            return {
                "answer": final_answer,
                "detected_intent": "llm_agent",
                "used_tools": used_tools,
                "parsed_context": {
                    "store_nbr": parsed.store_nbr,
                    "item_nbr": parsed.item_nbr,
                    "item_alias": parsed.item_alias,
                    "date_from": parsed.date_from,
                    "date_to": parsed.date_to,
                    "model_name": parsed.model_name,
                    "current_stock": parsed.current_stock,
                    "safety_stock": parsed.safety_stock,
                    "lead_time_days": parsed.lead_time_days,
                },
                "data": {
                    "tool_results": tool_results,
                },
            }

        return {
            "answer": "The agent reached the tool-call limit without producing a final answer.",
            "detected_intent": "llm_agent",
            "used_tools": used_tools,
            "parsed_context": {
                "store_nbr": parsed.store_nbr,
                "item_nbr": parsed.item_nbr,
                "item_alias": parsed.item_alias,
                "date_from": parsed.date_from,
                "date_to": parsed.date_to,
                "model_name": parsed.model_name,
                "current_stock": parsed.current_stock,
                "safety_stock": parsed.safety_stock,
                "lead_time_days": parsed.lead_time_days,
            },
            "data": {
                "tool_results": tool_results,
            },
        }

    # ==================================================
    # TOOL EXECUTION
    # ==================================================
    def _execute_tool(self, fn_name: str, args: Dict[str, Any], parsed: ParsedAgentContext) -> Dict[str, Any]:
        merged = self._merge_tool_args_with_context(args, parsed)

        if fn_name == "tool_predict_item":
            return self.tool_predict_item(merged)

        if fn_name == "tool_explain_prediction":
            return self.tool_explain_prediction(merged)

        if fn_name == "tool_recommend_order":
            return self.tool_recommend_order(merged)

        return {
            "ok": False,
            "error": f"Unknown tool: {fn_name}",
        }

    def _merge_tool_args_with_context(self, args: Dict[str, Any], parsed: ParsedAgentContext) -> ParsedAgentContext:
        merged = ParsedAgentContext(
            store_nbr=self._safe_int(args.get("store_nbr")) if args.get("store_nbr") is not None else parsed.store_nbr,
            item_nbr=self._resolve_item(args.get("item")) if args.get("item") is not None else parsed.item_nbr,
            item_alias=parsed.item_alias,
            date_from=self._normalize_date(args.get("date_from")) if args.get("date_from") else parsed.date_from,
            date_to=self._normalize_date(args.get("date_to")) if args.get("date_to") else parsed.date_to,
            model_name=self._resolve_model_name(args.get("model_name")) if args.get("model_name") else parsed.model_name,
            current_stock=self._safe_float(args.get("current_stock")) if args.get("current_stock") is not None else parsed.current_stock,
            safety_stock=self._safe_float(args.get("safety_stock")) if args.get("safety_stock") is not None else parsed.safety_stock,
            lead_time_days=self._safe_int(args.get("lead_time_days")) if args.get("lead_time_days") is not None else parsed.lead_time_days,
        )
        return merged


    def _build_tools_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "tool_predict_item",
                    "description": "Get forecast for a store, item, date or date range using a selected model.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "store_nbr": {"type": "integer"},
                            "item": {
                                "type": "string",
                                "description": "Item alias like 'Item 5' or numeric item_nbr."
                            },
                            "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                            "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                            "model_name": {"type": "string"},
                        },
                        "required": ["store_nbr", "item", "date_from"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_explain_prediction",
                    "description": "Explain forecast using model feature importance and forecast output.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "store_nbr": {"type": "integer"},
                            "item": {"type": "string"},
                            "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                            "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                            "model_name": {"type": "string"},
                        },
                        "required": ["store_nbr", "item", "date_from"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_recommend_order",
                    "description": "Recommend how much to order using forecast, current stock, safety stock and lead time.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "store_nbr": {"type": "integer"},
                            "item": {"type": "string"},
                            "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                            "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                            "model_name": {"type": "string"},
                            "current_stock": {"type": "number"},
                            "safety_stock": {"type": "number"},
                            "lead_time_days": {"type": "integer"},
                        },
                        "required": ["store_nbr", "item", "date_from"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    # ==================================================
    # REAL TOOLS
    # ==================================================
    def tool_predict_item(self, parsed: ParsedAgentContext) -> Dict[str, Any]:
        missing = []
        if parsed.store_nbr is None:
            missing.append("store")
        if parsed.item_nbr is None:
            missing.append("item")
        if parsed.date_from is None:
            missing.append("date")

        if missing:
            return {
                "ok": False,
                "error": f"Missing required fields for prediction: {', '.join(missing)}.",
                "rows": [],
            }

        model_name = parsed.model_name or self.loader.load_default_model_name()

        df = self.predictor.timeseries(
            store_nbr=int(parsed.store_nbr),
            item_nbr=int(parsed.item_nbr),
            date_from=parsed.date_from,
            date_to=parsed.date_to or parsed.date_from,
            model_name=model_name,
        )

        if df.empty:
            return {
                "ok": False,
                "error": "No prediction rows were returned for the selected store/item/date.",
                "rows": [],
                "model_name": model_name,
            }

        rows = []
        for _, r in df.iterrows():
            rows.append({
                "date": str(pd.Timestamp(r["date"]).date()),
                "store_nbr": int(r["store_nbr"]),
                "item_nbr": int(r["item_nbr"]),
                "pred": float(r["pred"]),
                "actual": float(r["actual"]) if pd.notna(r.get("actual")) else None,
                "abs_error": float(r["abs_error"]) if pd.notna(r.get("abs_error")) else None,
                "ape": float(r["ape"]) if pd.notna(r.get("ape")) else None,
            })

        alias_label = self._item_label(parsed.item_nbr)
        total_pred = float(sum(x["pred"] for x in rows))

        return {
            "ok": True,
            "tool": "tool_predict_item",
            "model_name": model_name,
            "store_nbr": parsed.store_nbr,
            "item_nbr": parsed.item_nbr,
            "item_label": alias_label,
            "date_from": parsed.date_from,
            "date_to": parsed.date_to or parsed.date_from,
            "rows": rows,
            "forecast_total": total_pred,
            "forecast_days": len(rows),
        }

    def tool_explain_prediction(self, parsed: ParsedAgentContext) -> Dict[str, Any]:
        pred_result = self.tool_predict_item(parsed)
        if not pred_result.get("ok"):
            return {
                "ok": False,
                "tool": "tool_explain_prediction",
                "error": pred_result.get("error", "Prediction step failed."),
            }

        model_name = pred_result["model_name"]
        fi_path = self.loader._root() / "tables" / f"feature_importance__{model_name}.csv"

        top_features = []
        if fi_path.exists():
            fi_df = pd.read_csv(fi_path)
            fi_df = fi_df.sort_values("importance", ascending=False).head(8).reset_index(drop=True)

            for _, row in fi_df.iterrows():
                feature_name = str(row["feature"])
                top_features.append({
                    "feature": feature_name,
                    "importance": float(row["importance"]),
                    "business_meaning": self._map_feature_to_business_meaning(feature_name),
                })

        return {
            "ok": True,
            "tool": "tool_explain_prediction",
            "prediction": pred_result,
            "top_features": top_features,
            "note": "This explanation is based on global feature importance, not SHAP local explanation.",
        }

    def tool_recommend_order(self, parsed: ParsedAgentContext) -> Dict[str, Any]:
        if parsed.date_from and not parsed.date_to:
            lead_time_days = parsed.lead_time_days or 1
            end_date = pd.Timestamp(parsed.date_from) + pd.Timedelta(days=max(lead_time_days - 1, 0))
            parsed.date_to = str(end_date.date())

        pred_result = self.tool_predict_item(parsed)
        if not pred_result.get("ok"):
            return {
                "ok": False,
                "tool": "tool_recommend_order",
                "error": pred_result.get("error", "Prediction step failed."),
            }

        forecast_total = float(pred_result["forecast_total"])
        current_stock = float(parsed.current_stock) if parsed.current_stock is not None else 0.0

        if parsed.safety_stock is not None:
            safety_stock = float(parsed.safety_stock)
            safety_source = "user_provided"
        else:
            safety_stock = float(max(1, math.ceil(forecast_total * 0.15)))
            safety_source = "default_15_percent"

        recommended_order_qty = max(0, math.ceil(forecast_total + safety_stock - current_stock))
        reorder_point = float(forecast_total + safety_stock)

        return {
            "ok": True,
            "tool": "tool_recommend_order",
            "prediction": pred_result,
            "forecast_total": forecast_total,
            "current_stock": current_stock,
            "safety_stock": safety_stock,
            "safety_stock_source": safety_source,
            "recommended_order_qty": int(recommended_order_qty),
            "reorder_point": reorder_point,
            "lead_time_days": parsed.lead_time_days or 1,
            "formula": "recommended_order = max(0, forecast_total + safety_stock - current_stock)",
        }

    # ==================================================
    # FALLBACK WITHOUT LLM
    # ==================================================
    def _fallback_without_llm(self, message: str, parsed: ParsedAgentContext) -> Dict[str, Any]:
        result = self.tool_predict_item(parsed)
        answer = (
            "LLM is not configured. "
            "I can still run deterministic forecast tools, but conversational agent mode is disabled."
        )
        return {
            "answer": answer,
            "detected_intent": "fallback",
            "used_tools": ["tool_predict_item"] if result.get("ok") else [],
            "parsed_context": {
                "store_nbr": parsed.store_nbr,
                "item_nbr": parsed.item_nbr,
                "item_alias": parsed.item_alias,
                "date_from": parsed.date_from,
                "date_to": parsed.date_to,
                "model_name": parsed.model_name,
                "current_stock": parsed.current_stock,
                "safety_stock": parsed.safety_stock,
                "lead_time_days": parsed.lead_time_days,
            },
            "data": result,
        }

    # ==================================================
    # CONTEXT EXTRACTION
    # ==================================================
    def _extract_context(self, message: str, context: Dict[str, Any]) -> ParsedAgentContext:
        parsed = ParsedAgentContext()

        parsed.store_nbr = self._safe_int(context.get("store_nbr") or context.get("store"))
        parsed.item_nbr = self._resolve_item(context.get("item_nbr") or context.get("item"))
        parsed.model_name = self._resolve_model_name(context.get("model_name") or context.get("model"))
        parsed.date_from, parsed.date_to = self._extract_dates_from_context(context)
        parsed.current_stock = self._safe_float(context.get("current_stock"))
        parsed.safety_stock = self._safe_float(context.get("safety_stock"))
        parsed.lead_time_days = self._safe_int(context.get("lead_time_days"))

        if parsed.store_nbr is None:
            parsed.store_nbr = self._extract_store_from_text(message)

        if parsed.item_nbr is None:
            item_nbr, item_alias = self._extract_item_from_text(message)
            parsed.item_nbr = item_nbr
            parsed.item_alias = item_alias

        if parsed.model_name is None:
            parsed.model_name = self._extract_model_from_text(message)

        if parsed.date_from is None:
            parsed.date_from, parsed.date_to = self._extract_dates_from_text(message)

        if parsed.current_stock is None:
            parsed.current_stock = self._extract_current_stock_from_text(message)

        if parsed.safety_stock is None:
            parsed.safety_stock = self._extract_safety_stock_from_text(message)

        if parsed.lead_time_days is None:
            parsed.lead_time_days = self._extract_lead_time_from_text(message)

        return parsed

    def _extract_dates_from_context(self, context: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        date_from = context.get("date_from")
        date_to = context.get("date_to")

        if date_from:
            date_from = self._normalize_date(date_from)
        if date_to:
            date_to = self._normalize_date(date_to)

        return date_from, date_to

    def _extract_dates_from_text(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        # 1) ISO dates: 2017-07-18
        matches = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
        if matches:
            if len(matches) == 1:
                return matches[0], matches[0]
            return matches[0], matches[1]

        # 2) dotted/slashed dates: 18.07.2017 or 18/07/2017
        m = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", text)
        if m:
            day, month, year = m.groups()
            try:
                dt = pd.Timestamp(year=int(year), month=int(month), day=int(day))
                s = str(dt.date())
                return s, s
            except Exception:
                pass

        # 3) Ukrainian month names: 18 липня 2017 року
        uk_months = {
            "січня": 1,
            "лютого": 2,
            "березня": 3,
            "квітня": 4,
            "травня": 5,
            "червня": 6,
            "липня": 7,
            "серпня": 8,
            "вересня": 9,
            "жовтня": 10,
            "листопада": 11,
            "грудня": 12,
        }

        m = re.search(
            r"\b(\d{1,2})\s+(січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|листопада|грудня)\s+(\d{4})(?:\s+року)?\b",
            text.lower(),
        )
        if m:
            day, month_name, year = m.groups()
            try:
                dt = pd.Timestamp(year=int(year), month=uk_months[month_name], day=int(day))
                s = str(dt.date())
                return s, s
            except Exception:
                pass

        return None, None

    def _extract_store_from_text(self, text: str) -> Optional[int]:
        patterns = [
            r"(?:store|магазин)\s*#?\s*(\d+)",
            r"\bstore\s+(\d+)\b",
            r"\bмагазин\s+(\d+)\b",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return self._safe_int(m.group(1))
        return None

    def _extract_item_from_text(self, text: str) -> Tuple[Optional[int], Optional[str]]:
        alias_match = re.search(r"\bitem\s+(\d+)\b", text, flags=re.IGNORECASE)
        if alias_match:
            alias_label = f"Item {alias_match.group(1)}"
            return self._resolve_item(alias_label), alias_label

        ua_alias_match = re.search(r"\bтовар[ау]?\s+(\d+)\b", text, flags=re.IGNORECASE)
        if ua_alias_match:
            alias_label = f"Item {ua_alias_match.group(1)}"
            return self._resolve_item(alias_label), alias_label

        item_patterns = [
            r"(?:item_nbr|item id|товар)\s*#?\s*(\d+)",
            r"\bitem\s*\(\s*(\d+)\s*\)",
        ]
        for pattern in item_patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return self._resolve_item(m.group(1)), None

        raw_numbers = re.findall(r"\b\d{5,}\b", text)
        for candidate in raw_numbers:
            resolved = self._resolve_item(candidate)
            if resolved is not None:
                return resolved, None

        return None, None

    def _extract_model_from_text(self, text: str) -> Optional[str]:
        available = self.loader.load_available_model_names()
        text_norm = text.strip().lower()

        for model_name in available:
            if model_name.lower() in text_norm:
                return model_name
        return None

    def _extract_current_stock_from_text(self, text: str) -> Optional[float]:
        patterns = [
            r"(?:current stock|stock|залишок|на складі)\s*[:=]?\s*(\d+(?:\.\d+)?)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return self._safe_float(m.group(1))
        return None

    def _extract_safety_stock_from_text(self, text: str) -> Optional[float]:
        patterns = [
            r"(?:safety stock|страховий запас|страхового запасу)\s*[:=]?\s*(\d+(?:\.\d+)?)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return self._safe_float(m.group(1))
        return None

    def _extract_lead_time_from_text(self, text: str) -> Optional[int]:
        patterns = [
            r"(?:lead time|lead_time|термін поставки)\s*[:=]?\s*(\d+)",
            r"(?:на|for)\s*(\d+)\s*(?:days|днів|day)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return self._safe_int(m.group(1))
        return None

    # ==================================================
    # HELPERS
    # ==================================================
    def _resolve_item(self, raw_item: Any) -> Optional[int]:
        if raw_item is None:
            return None

        raw_str = str(raw_item).strip()
        if not raw_str:
            return None

        if raw_str.isdigit():
            return int(raw_str)

        aliases = self.loader.load_item_aliases()
        alias_map = {
            str(x["display_name"]).strip().lower(): int(x["item_nbr"])
            for x in aliases
        }

        resolved = alias_map.get(raw_str.lower())
        if resolved is not None:
            return resolved

        bracket_match = re.search(r"\((\d+)\)\s*$", raw_str)
        if bracket_match:
            return int(bracket_match.group(1))

        return None

    def _resolve_model_name(self, raw_model: Any) -> Optional[str]:
        if raw_model is None:
            return None
        raw = str(raw_model).strip()
        if not raw:
            return None

        available = self.loader.load_available_model_names()
        for model_name in available:
            if model_name.lower() == raw.lower():
                return model_name
        return None

    def _item_label(self, item_nbr: int) -> str:
        aliases = self.loader.load_item_aliases()
        for row in aliases:
            if int(row["item_nbr"]) == int(item_nbr):
                return f'{row["display_name"]} ({item_nbr})'
        return str(item_nbr)

    def _map_feature_to_business_meaning(self, feature_name: str) -> str:
        f = feature_name.lower()

        if "lag_" in f or "log_lag_" in f:
            return "Historical sales from previous days strongly influence the forecast."
        if "rolling_mean" in f or "rolling_std" in f or "ewm_mean" in f:
            return "Recent sales trend and short-term demand dynamics are important."
        if "promo" in f or "onpromotion" in f:
            return "Promotions can increase or shift demand."
        if "holiday" in f:
            return "Holiday effects can change purchasing behavior."
        if "day_of_week" in f or "dow_" in f or "month_" in f or "week_of_year" in f or "quarter" in f:
            return "Calendar seasonality affects demand."
        if "family" in f or "item_" in f or "store_" in f or "cluster" in f or "city" in f or "state" in f:
            return "Product/store identity and location characteristics matter."
        if "transactions" in f:
            return "Store traffic and transactional activity may affect expected sales."
        if "oil" in f:
            return "External economic signals may indirectly affect demand."

        return "This feature contributes to model prediction quality."

    def _normalize_date(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            return str(pd.Timestamp(value).date())
        except Exception:
            return None

    def _safe_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _safe_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None