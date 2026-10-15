"""run_demo.py - run the demo queries from the command line and save a transcript.

    python run_demo.py              # the 5 required queries
    python run_demo.py --all        # 5 required + 4 extra queries
"""
import argparse
import sys

from agent import ask, build_agent
from demo_queries import EXTRA_QUERIES, REQUIRED_QUERIES

sys.stdout.reconfigure(encoding="utf-8")   # so the rupee sign prints on Windows consoles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="also run the 4 extra queries")
    parser.add_argument("--out", default="demo_transcript.md", help="transcript file")
    args = parser.parse_args()

    queries = REQUIRED_QUERIES + (EXTRA_QUERIES if args.all else [])
    agent = build_agent()
    lines, passed = ["# Demo transcript", ""], 0

    for number, (query, expected) in enumerate(queries, start=1):
        result = ask(agent, query)
        used = [c.name for c in result.tool_calls]
        ok = expected in used
        passed += ok
        print(f"\n[{number}] {query}\n    tools used : {used}   expected: {expected}   {'PASS' if ok else 'FAIL'}")
        print(f"    answer     : {result.answer}")
        lines += [f"## {number}. {query}", f"- Expected tool: `{expected}`  |  Tools used: `{used}`  |  {'PASS' if ok else 'FAIL'}",
                  "", result.answer, ""]

    print(f"\nTool-selection accuracy: {passed}/{len(queries)}")
    lines.append(f"**Tool-selection accuracy: {passed}/{len(queries)}**")
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Transcript saved to {args.out}")


if __name__ == "__main__":
    main()
