import argparse
import os

from demo_class import RunDemo


def get_parser():
    parser = argparse.ArgumentParser(description="Fault classification for PV panels Demo")
    parser.add_argument(
        "--input_path",
        default="demo_images",
        help="Path with the demo images."
    )
    parser.add_argument(
        "--output_path",
        default="demo_outputs",
        help="Path to save output results."
    )
    parser.add_argument(
        "--segmentation_model",
        default=os.path.abspath(os.path.join("models", "UNet_segmentation_model.h5")),
        help="Path .h5 model for segmentation."
    )
    parser.add_argument(
        "--bin_classification_model",
        default=os.path.abspath(os.path.join("models", "bin_classification_model.h5")),
        help="Path .h5 model for binary classification."
    )
    parser.add_argument(
        "--quat_classification_model",
        default=os.path.abspath(os.path.join("models", "quat_classification_model.h5")),
        help="Path .h5 model for quaternary classification."
    )
    parser.add_argument(
        "--mode",
        default="all",
        help="Mode to use (all, segmentation, bin_classification, quat_classification)"
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.input_path:
        # Get the absolute paths for images and masks
        abs_path, _, images = next(os.walk(args.input_path))
        images = [os.path.abspath(os.path.join(abs_path, image)) for image in images]
        print(f"{len(images)} images were found!")

        # Create a DemoRun object
        demo = RunDemo(images=images,
                       segmentation_model=args.segmentation_model,
                       bin_classification_model=args.bin_classification_model,
                       quat_classification_model=args.quat_classification_model,
                       output_path=args.output_path)

        # Run mode
        mode = args.mode

        if mode == "segmentation":
            demo.run_segmentation()
        elif mode == "bin_classification":
            demo.run_bin_classification()
        elif mode == "quat_classification":
            demo.run_quat_classification()
        elif mode == "all":
            demo.run_segmentation()
            demo.run_bin_classification()
            demo.run_quat_classification()
        else:
            AttributeError("Select a valid mode.")
