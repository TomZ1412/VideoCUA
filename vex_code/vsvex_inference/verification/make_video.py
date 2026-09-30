import json
import os
import subprocess
from tqdm import tqdm
import time


def get_audio_duration(audio_path):
    """Uses ffprobe to get the exact duration of an audio file."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ]
    print(audio_path)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

def create_video_ffmpeg_overlap(image_list, audio_list, output_name):
    has_extra_image = len(image_list) > len(audio_list)
    temp_dir = "temp_overlap"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        N = len(audio_list)
        video_list_path = os.path.join(temp_dir, "video_list.txt")
        audio_list_path = os.path.join(temp_dir, "audio_list.txt")
        
        # 1. Probe audio durations and build the instruction lists
        with open(video_list_path, "w") as vf, open(audio_list_path, "w") as af:
            for i in range(N):
                dur = get_audio_duration(audio_list[i])
                
                # Image duration is 0.5s shorter than audio (min 0.1s to prevent FFmpeg errors)
                vid_dur = max(0.1, dur - 0.5)
                
                img_path = os.path.abspath(image_list[i]).replace("\\", "/")
                aud_path = os.path.abspath(audio_list[i]).replace("\\", "/")
                
                vf.write(f"file '{img_path}'\nduration {vid_dur}\n")
                af.write(f"file '{aud_path}'\n")
            
            if has_extra_image:
                # The last image covers the remaining 0.5s of the last audio + 3s pause
                # Math: We cut 0.5s N times. We add (0.5 * N) + 3.0 to the last image to balance it out.
                last_img_dur = 3.0 + 0.5 * N
                last_img_path = os.path.abspath(image_list[-1]).replace("\\", "/")
                vf.write(f"file '{last_img_path}'\nduration {last_img_dur}\n")

        # 2. Create Master Audio Track (Re-encode to uniform 16kHz AAC)
        master_audio = os.path.join(temp_dir, "master_audio.aac")
        cmd_audio = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", audio_list_path,
            "-c:a", "aac", "-ar", "16000", "-b:a", "128k",
            master_audio
        ]
        subprocess.run(cmd_audio, check=True)

        # 3. Create Master Video Track (Encode images to 720p 8fps)
        master_video = os.path.join(temp_dir, "master_video.mp4")
        cmd_video = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", video_list_path,
            "-vf", "scale=-2:720",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-r", "8",
            master_video
        ]
        subprocess.run(cmd_video, check=True)

        # 4. Mux them together (Instantaneous stream copy)
        # Because the video track is slightly longer, the audio will end naturally 
        # and the video will just hold the last frame for the remaining 3 seconds.
        cmd_mux = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", master_video, "-i", master_audio,
            "-c", "copy", "-map", "0:v", "-map", "1:a",
            output_name
        ]
        subprocess.run(cmd_mux, check=True)

    finally:
        # Cleanup temp files
        for f in [video_list_path, audio_list_path, master_audio, master_video]:
            if os.path.exists(f): os.remove(f)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)

# --- Main Execution ---
if __name__ == "__main__":
    # print("Sleep 18000 seconds")
    # time.sleep(18000)
    # with open("traj_data_audio_vs2_desc.json") as fin:
    with open("traj_data_audio_vs2_agentnet_cuagym220_dualboth_fixed.json") as fin:
        data = json.load(fin)

    os.makedirs("videos_vs2_agentnet_cuagym220_dualboth_fixed", exist_ok=True)
    
    for datapiece in tqdm(data, desc="Generating Videos"):
        imagelist = [step["screenshot"] for step in datapiece["steps"]]
        audiolist = [step["audio"] for step in datapiece["steps"]]
        imagelist = [datapiece["init_screenshot"]] + imagelist
        
        output_path = f"videos_vs2_agentnet_cuagym220_dualboth_fixed/{datapiece['task']}.mp4"
        
        create_video_ffmpeg_overlap(imagelist, audiolist, output_path)
        datapiece["video"] = output_path

    with open("traj_data_video_vs2_agentnet_cuagym220_dualboth_fixed.json", "w") as fout:
        json.dump(data, fout, indent=4)