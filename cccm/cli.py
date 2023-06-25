import argparse
import cv2
from cccm.algos.base_algo import BaseAlgo
from cccm.algos.gray_world import GrayWorld

from cccm.read_raw_image import read_raw_image


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cccm",
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height",type=int,  required=True)
    parser.add_argument("--blacklevel",type=int,  required=True)
    parser.add_argument("--bpp",type=int,  required=True)
    parser.add_argument("--bit_depth",type=int,  required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--pattern", default="RG")

    return parser

def algos_resolver(algo: str) -> BaseAlgo:
    algos = {
        "gray_world": GrayWorld
    }
    res = algos.get(algo)
    if not res:
        raise ValueError()
    return res


def main():
    args = get_parser().parse_args()

    image = read_raw_image(
        args.input,
        args.width,
        args.height,
        args.blacklevel,
        args.bpp,
        args.bit_depth,
    )
    demosaiced_image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
    algo = algos_resolver(args.algo)
    result_image = algo.apply(demosaiced_image)
    cv2.imwrite(f"./{args.output}.png", result_image)
