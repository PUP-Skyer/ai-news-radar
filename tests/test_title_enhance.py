"""Tests for the LLM title-enhancement step: the enhance-gate, the
micro-crawl + DeepSeek rewrite validation, cache-key stability, and the
DEEPSEEK_API_KEY no-op wiring path. All network/LLM calls are mocked."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from scripts.update_news import (
    TITLE_ENHANCE_CACHE_PREFIX,
    add_title_enhancements,
    enhance_title_deepseek,
    title_needs_enhance,
)


DS_ENV = {"DEEPSEEK_API_KEY": "sk-test"}


def deepseek_ok_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


class TestTitleNeedsEnhanceGate(unittest.TestCase):
    def test_short_english_title_is_gated(self):
        item = {
            "site_id": "producthunt",
            "source_tier": "discussion",
            "title_en": "AI Visibility",
            "title_original": "AI Visibility",
        }
        self.assertTrue(title_needs_enhance(item))

    def test_normal_length_official_title_is_not_gated(self):
        item = {
            "site_id": "official_ai",
            "source_tier": "official",
            "title_en": "OpenAI launches new Codex agent for developers this week",
            "title_original": "OpenAI launches new Codex agent for developers this week",
        }
        self.assertFalse(title_needs_enhance(item))

    def test_year_suffixed_aggregate_title_is_gated(self):
        item = {
            "site_id": "techurls",
            "source_tier": "aggregate",
            "title_en": "Major New AI Regulation Framework Overview And Policy Analysis (2024)",
            "title_original": "Major New AI Regulation Framework Overview And Policy Analysis (2024)",
        }
        self.assertTrue(title_needs_enhance(item))

    def test_short_effective_title_on_gated_tier_is_gated(self):
        item = {
            "site_id": "hackernews",
            "source_tier": "discussion",
            "title_en": "New chip breakthrough announced today",
            "title_original": "New chip breakthrough announced today",
            "title_zh": "新芯片突破",
        }
        self.assertTrue(title_needs_enhance(item))

    def test_curated_tier_never_gated_even_if_short(self):
        item = {
            "site_id": "curated_media",
            "source_tier": "curated",
            "title_en": "AI Visibility",
            "title_original": "AI Visibility",
        }
        self.assertFalse(title_needs_enhance(item))


class TestEnhanceTitleDeepseekValidation(unittest.TestCase):
    def setUp(self):
        self.title = "AI Visibility raises Grok concerns"
        self.context = "AI Visibility is a new product from Grok that monitors brand mentions."

    def test_fabricated_entity_title_rejected(self):
        with patch.dict("os.environ", DS_ENV, clear=True), patch(
            "scripts.update_news.requests.post",
            return_value=deepseek_ok_response("行业迎来新一轮技术变革与市场调整"),
        ):
            result = enhance_title_deepseek(self.title, self.context)
        self.assertIsNone(result)

    def test_too_short_result_rejected(self):
        with patch.dict("os.environ", DS_ENV, clear=True), patch(
            "scripts.update_news.requests.post",
            return_value=deepseek_ok_response("Grok来了"),
        ):
            result = enhance_title_deepseek(self.title, self.context)
        self.assertIsNone(result)

    def test_good_rewrite_accepted(self):
        good = "AI Visibility新品引发对Grok数据来源的担忧"
        with patch.dict("os.environ", DS_ENV, clear=True), patch(
            "scripts.update_news.requests.post",
            return_value=deepseek_ok_response(good),
        ):
            result = enhance_title_deepseek(self.title, self.context)
        self.assertEqual(result, good)

    def test_no_key_returns_none_without_network(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "scripts.update_news.requests.post"
        ) as mock_post:
            result = enhance_title_deepseek(self.title, self.context)
        self.assertIsNone(result)
        mock_post.assert_not_called()


class TestAddTitleEnhancementsWiring(unittest.TestCase):
    def make_item(self, url="https://example.com/a", title_en="AI Visibility"):
        return {
            "id": "item-1",
            "site_id": "producthunt",
            "source_tier": "discussion",
            "url": url,
            "title": title_en,
            "title_en": title_en,
            "title_original": title_en,
        }

    def test_no_key_returns_items_unchanged_and_no_network(self):
        item = self.make_item()
        session = MagicMock()
        cache: dict[str, str] = {}
        with patch.dict("os.environ", {}, clear=True), patch(
            "scripts.update_news.fetch_title_context"
        ) as mock_fetch, patch(
            "scripts.update_news.enhance_title_deepseek"
        ) as mock_enhance:
            out_items, out_cache = add_title_enhancements([item], session, cache)
        self.assertEqual(out_items, [item])
        self.assertEqual(out_cache, {})
        mock_fetch.assert_not_called()
        mock_enhance.assert_not_called()
        session.get.assert_not_called()

    def test_cache_key_is_stable_across_runs_and_skips_second_llm_call(self):
        item = self.make_item()
        session = MagicMock()
        cache: dict[str, str] = {}
        enhanced_title = "AI Visibility发布品牌监测新品"
        with patch.dict("os.environ", DS_ENV, clear=True), patch(
            "scripts.update_news.fetch_title_context", return_value="some page context"
        ) as mock_fetch, patch(
            "scripts.update_news.enhance_title_deepseek", return_value=enhanced_title
        ) as mock_enhance:
            first_items, cache = add_title_enhancements([item], session, cache)

        self.assertEqual(len(cache), 1)
        self.assertTrue(next(iter(cache)).startswith(TITLE_ENHANCE_CACHE_PREFIX))
        self.assertEqual(first_items[0]["title_enhanced_zh"], enhanced_title)
        mock_fetch.assert_called_once()
        mock_enhance.assert_called_once()

        # Second run with the same url+title: same cache key hit, no new LLM/crawl calls.
        second_item = self.make_item()
        with patch.dict("os.environ", DS_ENV, clear=True), patch(
            "scripts.update_news.fetch_title_context"
        ) as mock_fetch2, patch(
            "scripts.update_news.enhance_title_deepseek"
        ) as mock_enhance2:
            second_items, cache = add_title_enhancements([second_item], session, cache)

        self.assertEqual(len(cache), 1)
        self.assertEqual(second_items[0]["title_enhanced_zh"], enhanced_title)
        mock_fetch2.assert_not_called()
        mock_enhance2.assert_not_called()

    def test_negative_cache_on_empty_context_skips_llm_and_is_not_retried(self):
        item = self.make_item()
        session = MagicMock()
        cache: dict[str, str] = {}
        with patch.dict("os.environ", DS_ENV, clear=True), patch(
            "scripts.update_news.fetch_title_context", return_value=""
        ) as mock_fetch, patch(
            "scripts.update_news.enhance_title_deepseek"
        ) as mock_enhance:
            out_items, cache = add_title_enhancements([item], session, cache)

        self.assertNotIn("title_enhanced_zh", out_items[0])
        self.assertEqual(len(cache), 1)
        mock_fetch.assert_called_once()
        mock_enhance.assert_not_called()

        # Re-run: negative cache hit means no repeat crawl.
        second_item = self.make_item()
        with patch.dict("os.environ", DS_ENV, clear=True), patch(
            "scripts.update_news.fetch_title_context"
        ) as mock_fetch2:
            out_items2, cache = add_title_enhancements([second_item], session, cache)
        self.assertNotIn("title_enhanced_zh", out_items2[0])
        mock_fetch2.assert_not_called()

    def test_ungated_item_is_left_untouched(self):
        item = self.make_item(title_en="OpenAI launches new Codex agent for developers this week")
        item["source_tier"] = "official"
        item["site_id"] = "official_ai"
        session = MagicMock()
        cache: dict[str, str] = {}
        with patch.dict("os.environ", DS_ENV, clear=True), patch(
            "scripts.update_news.fetch_title_context"
        ) as mock_fetch, patch(
            "scripts.update_news.enhance_title_deepseek"
        ) as mock_enhance:
            out_items, cache = add_title_enhancements([item], session, cache)
        self.assertNotIn("title_enhanced_zh", out_items[0])
        self.assertEqual(cache, {})
        mock_fetch.assert_not_called()
        mock_enhance.assert_not_called()

    def test_per_run_cap_blocks_further_crawls(self):
        items = [self.make_item(url=f"https://example.com/{i}") for i in range(3)]
        session = MagicMock()
        cache: dict[str, str] = {}
        with patch.dict("os.environ", DS_ENV, clear=True), patch(
            "scripts.update_news.fetch_title_context", return_value="ctx"
        ) as mock_fetch, patch(
            "scripts.update_news.enhance_title_deepseek", return_value="标题增强结果占位文本"
        ):
            add_title_enhancements(items, session, cache, max_new_per_run=1)
        self.assertEqual(mock_fetch.call_count, 1)
        self.assertEqual(len(cache), 1)


if __name__ == "__main__":
    unittest.main()
