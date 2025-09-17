from youtube_transcript_api import YouTubeTranscriptApi

print("Attributes of YouTubeTranscriptApi class:")
print(dir(YouTubeTranscriptApi))

print("\nAttributes of a YouTubeTranscriptApi instance:")
try:
    print(dir(YouTubeTranscriptApi()))
except Exception as e:
    print(f"Failed to create instance: {e}")
