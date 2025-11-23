from moviepy.editor import VideoFileClip

input_video = "First.mp4"
output_video = "CFirst.mp4"

# Duration to keep (1 minute 30 seconds = 90 seconds)
end_time = 90  

# Load video
clip = VideoFileClip(input_video)

# Crop from 0 to 1:30
cropped = clip.subclip(0, end_time)

# Save the output
cropped.write_videofile(output_video, codec="libx264", audio_codec="aac")

print("Cropping completed successfully!")
