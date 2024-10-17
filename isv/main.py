import json
import os
import sys
from dataclasses import asdict

import annotation

from isv.src import features, prediction


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Predict pathogenicity from annotated CNV.")
    parser.add_argument("input", help="Annotated CNV stored as json")
    parser.add_argument("--output", help="Path to store the prediction JSON. Else prints to stdout.", default=None)
    args = parser.parse_args()

    whole_cnv_annotation = annotation.Annotation.load_from_json(args.input)
    cnv_features = features.get(whole_cnv_annotation)

    pred_json = prediction.predict(cnv_features, whole_cnv_annotation.cnv.cnv_type)

    if args.output:
        path = os.path.abspath(args.output)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(pred_json), f, indent=2)
    else:
        print(json.dumps(asdict(pred_json), indent=2), file=sys.stdout)


if __name__ == "__main__":
    main()
