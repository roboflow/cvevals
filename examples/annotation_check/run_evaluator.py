
from annotation_eval import AnnotationEval
import argparse
'''
from roboflow import Roboflow
rf = Roboflow(api_key="")
project = rf.workspace("").project("")
dataset = project.version().download("voc")
'''


# translate above to argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    "--local_data_folder",
    type=str,
    required=True,
    help="Absolute path to local data folder of images and annotations ",
)

parser.add_argument(
    "--roboflow_data_folder",
    type=str,
    required=True,
    help="Absolute path to roboflow uploads of images and annotations ",
)

args = parser.parse_args()

LOCAL_FOLDER = args.local_data_folder
ROBOFLOW_FOLDER = args.roboflow_data_folder

if __name__ == "__main__":
    eval = AnnotationEval(
        local_folder = LOCAL_FOLDER,
        roboflow_folder = ROBOFLOW_FOLDER)
    eval.collect_images_labels()
    eval.run_eval_loop()
