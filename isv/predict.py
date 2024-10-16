import json
import sys
from dataclasses import asdict

import annotation

from isv.src import isv_attributes, prediction


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Predict pathogenicity from annotated CNV.")
    parser.add_argument("input", help="Annotated CNV stored as json")
    parser.add_argument("--output", help="Path to store the prediction JSON. Else prints to stdout.", default=None)
    args = parser.parse_args()

    annot = annotation.Annotation.load_from_json(args.input)
    isv_annot = isv_attributes.get_annotation_attributes(annot)

    pred_json = prediction.predict(isv_annot, annot.cnv.cnv_type)

    if args.output:
        pred_json.store_as_json(args.output)
    else:
        print(json.dumps(asdict(pred_json), indent=2), file=sys.stdout)


if __name__ == "__main__":
    main()
