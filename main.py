import argparse
from src.queue import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="""
A Qwen 2.1 image-generation job scheduler.

Usage:
  main.py --worker
      Start the background worker.

  main.py --submit --prompt "An orange on a table"
      Submit a new image-generation job.

  main.py --submit --img_input table.png --prompt "Add an orange on the table"
      Submit a new image-generation job.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Action arguments
    parser.add_argument("--worker", action="store_true", default=False, help="Whether to start the worker")
    parser.add_argument("--submit", action="store_true", default=False, help="Whether to submit a job")

    # Model input parameters
    parser.add_argument("--prompt", type=str, default="", help="Prompt for the model")
    parser.add_argument("--img_input", type=str, default="", help="Image to edit in /input directory")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=30)
    return parser.parse_args()

def main(**args):
    if args["worker"]:
        worker()
    if args["submit"]:
        submit_job(args["prompt"],
               args["img_input"],
               args["width"],
               args["height"],
               args["steps"])

if __name__ == "__main__":
    args = vars(parse_args())
    main(**args)
