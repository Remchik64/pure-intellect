"""Benchmark reporter — красивый отчёт сравнения baseline vs with_memory."""

from .runner import ScenarioResult


def compare(baseline: ScenarioResult, memory: ScenarioResult) -> dict:
    """Сравнить два результата и вычислить delta."""
    b = baseline.to_dict()["metrics"]
    m = memory.to_dict()["metrics"]
    
    recall_delta = m["avg_keyword_recall"] - b["avg_keyword_recall"]
    recall_pct = (recall_delta / max(b["avg_keyword_recall"], 0.001)) * 100
    
    return {
        "scenario": baseline.scenario_name,
        "turns": baseline.total_turns,
        "baseline": b,
        "with_memory": m,
        "delta": {
            "keyword_recall": round(recall_delta, 3),
            "keyword_recall_pct": round(recall_pct, 1),
            "topic_switches_detected": m["topic_switches_detected"],
            "context_restorations": m["context_restorations"],
            "facts_stored": m["total_facts_stored"],
        }
    }


def print_report(results: list[dict]) -> None:
    """Распечатать красивый отчёт в консоль."""
    SEP = "═" * 65
    sep = "─" * 65
    
    print(f"\n{SEP}")
    print("  🧠 PURE INTELLECT — BENCHMARK REPORT")
    print(SEP)
    
    total_recall_delta = 0.0
    
    for r in results:
        b = r["baseline"]
        m = r["with_memory"]
        d = r["delta"]
        
        print(f"\n📋 Сценарий: {r['scenario']}")
        print(f"   Turns: {r['turns']}")
        print(sep)
        
        print(f"  {'Метрика':<35} {'Baseline':>10} {'Memory':>10} {'Delta':>10}")
        print(f"  {sep}")
        
        recall_b = b['avg_keyword_recall']
        recall_m = m['avg_keyword_recall']
        recall_arrow = "▲" if recall_m > recall_b else "▼"
        print(
            f"  {'Keyword Recall (avg)':<35} "
            f"{recall_b:>9.1%} "
            f"{recall_m:>9.1%} "
            f"  {recall_arrow}{d['keyword_recall_pct']:+.0f}%"
        )
        
        coh_b = b['avg_coherence_score']
        coh_m = m['avg_coherence_score']
        print(
            f"  {'Coherence Score (avg)':<35} "
            f"{coh_b:>10.3f} "
            f"{coh_m:>10.3f} "
            f"  {'N/A' if coh_b == 0 else ''}"
        )
        
        print(
            f"  {'Topic Switches Detected':<35} "
            f"{'—':>10} "
            f"{d['topic_switches_detected']:>10} "
        )
        print(
            f"  {'Context Restorations':<35} "
            f"{'—':>10} "
            f"{d['context_restorations']:>10} "
        )
        print(
            f"  {'Facts in Long-term Storage':<35} "
            f"{'0':>10} "
            f"{d['facts_stored']:>10} "
        )
        print(
            f"  {'Peak Working Memory Facts':<35} "
            f"{'0':>10} "
            f"{m['peak_memory_facts']:>10} "
        )
        
        total_recall_delta += d["keyword_recall_pct"]
    
    # Итоговый вердикт
    avg_improvement = total_recall_delta / max(len(results), 1)
    print(f"\n{SEP}")
    print("  📊 ИТОГ")
    print(sep)
    print(f"  Сценариев протестировано: {len(results)}")
    print(f"  Средний прирост Keyword Recall: +{avg_improvement:.1f}%")
    print()
    
    if avg_improvement >= 50:
        verdict = "✅ СИСТЕМА РАБОТАЕТ ЗНАЧИТЕЛЬНО ЛУЧШЕ BASELINE"
    elif avg_improvement >= 20:
        verdict = "✅ СИСТЕМА РАБОТАЕТ ЛУЧШЕ BASELINE"
    elif avg_improvement >= 0:
        verdict = "⚠️  НЕБОЛЬШОЕ УЛУЧШЕНИЕ ОТНОСИТЕЛЬНО BASELINE"
    else:
        verdict = "❌ ТРЕБУЕТ ДОРАБОТКИ"
    
    print(f"  {verdict}")
    print(f"{SEP}\n")
