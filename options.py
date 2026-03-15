import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--device", type=str, default="", help="cuda|cpu|mps|auto (default auto)")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--ckpt", type=str, default="", help="checkpoint path for eval_only")
    return p.parse_args()
