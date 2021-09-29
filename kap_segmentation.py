import torch
from model import MattingNetwork
import argparse

from inference import convert_video

def kap_convert(input_source, output_location):
    model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
    model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))

    convert_video(
        model,                           # The model, can be on any device (cpu or cuda).
        input_source=input_source,        # A video file or an image sequence directory.
        output_type='png_sequence',             # Choose "video" or "png_sequence"
        output_composition=output_location, # File path if video; directory path if png sequence.
        #output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
        downsample_ratio=None,           # A hyperparameter to adjust or use None for auto.
        seq_chunk=12,                    # Process n frames at once for better parallelism.
    )

parser = argparse.ArgumentParser(description='Inference video')

parser.add_argument('--input_source', type=str, required=True)
parser.add_argument('--output_location', type=str, required=True)

args = parser.parse_args()

kap_convert(args.input_source, args.output_location)