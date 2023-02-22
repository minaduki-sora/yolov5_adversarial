import os.path as osp
from subprocess import Popen, PIPE


def ffmpeg_create_video_from_image_dir(source_image_dir: str, target_video_path: str, etxn: str = ".jpg", framerate: int = 30) -> None:
    """
    Create a video from images in source_image_dir
    """
    source = osp.join(source_image_dir, f"*{etxn}")
    cmd = ["ffmpeg", "-y",
           "-framerate", str(framerate),
           "-pattern_type", "glob",
           "-i", source,
           target_video_path]
    output, error = Popen(
        cmd, universal_newlines=True, stdin=PIPE, stdout=PIPE, stderr=PIPE).communicate()


def ffmpeg_combine_two_vids(vid1: str, vid2: str, target_video_path: str) -> None:
    """
    Attaches two videos side by side horizontally and saves to target_video_path
    """
    cmd = ["ffmpeg", "-y",
           "-i", vid1, "-i", vid2,
           "-filter_complex", "hstack",
           target_video_path]
    output, error = Popen(
        cmd, universal_newlines=True, stdin=PIPE, stdout=PIPE, stderr=PIPE).communicate()


def ffmpeg_combine_three_vids(vid1: str, vid2: str, vid3: str, target_video_path: str) -> None:
    """
    Attaches three videos side by side horizontally and saves to target_video_path
    """
    filter_pat = "[1:v][0:v]scale2ref=oh*mdar:ih[1v][0v];[2:v][0v]scale2ref=oh*mdar:ih[2v][0v];[0v][1v][2v]hstack=3,scale='2*trunc(iw/2)':'2*trunc(ih/2)'"
    cmd = ["ffmpeg", "-y",
           "-i", vid1, "-i", vid2, "-i", vid3,
           "-filter_complex", filter_pat,
           target_video_path]
    output, error = Popen(
        cmd, universal_newlines=True, stdin=PIPE, stdout=PIPE, stderr=PIPE).communicate()
