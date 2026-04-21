from argparse import ArgumentParser
from benchmark import benchmark

"""
The structure of the {GT_ROOT} can be either of the follow two structures.
{GT_ROOT} and {PRED_ROOT} should be of the same format.

1. VRS test structure (same as SAM2's SAV test)
    {GT_ROOT}  # gt root folder
        ├── {video_id}
        │     ├── 000               # all masks associated with obj 000
        │     │    ├── {frame_id}.png    # mask for object 000 in {frame_id} (binary mask)
        │     │    └── ...
        │     ├── 001               # all masks associated with obj 001
        │     ├── 002               # all masks associated with obj 002
        │     └── ...
        ├── {video_id}
        ├── {video_id}
        └── ...

2. Similar to DAVIS structure:

    {GT_ROOT}  # gt root folder
        ├── {video_id}
        │     ├── {frame_id}.png          # annotation in {frame_id} (may contain multiple objects)
        │     └── ...
        ├── {video_id}
        ├── {video_id}
        └── ...
"""


parser = ArgumentParser()
parser.add_argument(
    "--gt_root",
    required=True,
    help="Path to the GT folder.",
)
parser.add_argument(
    "--pred_root",
    required=True,
    help="Path to a folder containing folders of masks to be evaluated, with exactly the same structure as gt_root",
)
parser.add_argument(
    "-n", "--num_processes", default=16, type=int, help="Number of concurrent processes"
)
parser.add_argument(
    "-s",
    "--strict",
    help="Make sure every video in the gt_root folder has a corresponding video in the prediction",
    action="store_true",
)
parser.add_argument(
    "-q",
    "--quiet",
    help="Quietly run evaluation without printing the information out",
    action="store_true",
)
# https://github.com/davisvideochallenge/davis2017-evaluation/blob/d34fdef71ce3cb24c1a167d860b707e575b3034c/davis2017/evaluation.py#L85
parser.add_argument(
    "--do_not_skip_first_and_last_frame",
    help="We don't skip the first and the last annotated frames in evaluation.",
    action="store_true",
)
parser.add_argument(
    "--category",
    default="robot",
    help="object type: arm, gripper, robot",
)


if __name__ == "__main__":
    args = parser.parse_args()
    benchmark(
        [args.gt_root],
        [args.pred_root],
        args.strict,
        args.num_processes,
        verbose=not args.quiet,
        skip_first_and_last=not args.do_not_skip_first_and_last_frame,
        category=args.category,
    )
