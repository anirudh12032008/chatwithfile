# might abandon later but keeping it rn for cli

import os 
import sys
import argparse
import rag 

def cmd_index(args):
    rag.build_index(args.docs)

def cmd_chat(args):
    store = rag.load_index()
    while True:
        q = input().strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        reply = rag.answer(store, q, k=args.k, model=args.model)
        print(reply)

def main():
    p = argparse.ArgumentParser(description="Chat")
    sub = p.add_subparsers(dest="command", required=True)
    pi = sub.add_parser("index", help="build index")
    pi.add_argument("--docs", default="docs")
    pi.add_argument("--overlap", type=int, default=150)
    pi.set_defaults(func=cmd_index)
    pc = sub.add_parser("chat")
    pc.add_argument("--model", default=None)
    pc.add_argument("--k", type=int, default=4)
    pc.set_defaults(func=cmd_chat)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()