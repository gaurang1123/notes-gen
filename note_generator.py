import os
import re
import json
import argparse
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from tqdm import tqdm

def get_gemini_api_key():
    """Gets the Gemini API key from the environment variables."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    return api_key

def get_transcript(video_url: str) -> str:
    """
    Retrieves the transcript for a YouTube video.

    Args:
        video_url: The URL of the YouTube video.

    Returns:
        The transcript as a single string.
    """
    try:
        video_id = video_url.split("v=")[1].split("&")[0]
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        try:
            transcript = transcript_list.find_transcript(['en'])
        except NoTranscriptFound:
            # If English not found, take the first available transcript
            transcript = list(transcript_list)[0]

        fetched_transcript = transcript.fetch()
        transcript_text = " ".join([d['text'] for d in fetched_transcript])
        return transcript_text
    except NoTranscriptFound:
        raise NoTranscriptFound("No transcript found for this video.")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def preprocess_transcript(text: str) -> str:
    """
    Preprocesses the transcript by inserting punctuation.

    Args:
        text: The raw transcript text.

    Returns:
        The preprocessed text with punctuation.
    """
    # A simple regex to add periods at the end of sentences.
    # This looks for a lowercase letter followed by an uppercase letter,
    # which often indicates a sentence break without punctuation.
    text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)
    return text

def chunk_text(text: str, max_chunk_size: int = 8000) -> list[str]:
    """
    Splits the text into chunks of a specified maximum size.

    Args:
        text: The text to be chunked.
        max_chunk_size: The maximum size of each chunk.

    Returns:
        A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    for sentence in text.split("."):
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + "."
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + "."
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

PROMPT_START = """
Act as an expert academic note-taker and content summarizer. Below is the first part of a long lecture transcript. Your task is to analyze this text and generate a comprehensive, structured, and detailed set of notes. The notes must be **written exclusively in simple English**, regardless of the input language.

Crucial Instructions:
* Analyze the Provided Content: Carefully read the entire text. Identify key themes, definitions, examples, and rules.
* Knowledge Augmentation: Supplement the notes with your own knowledge to enrich the content where the transcript is brief.
* Note Structure: Organize the notes with clear headings, bolded key terms, bulleted/numbered lists, and proper formatting for examples and formulas.
* Final Output: Provide the detailed notes for this section. Do not provide a final summary or conclusion.
"""

PROMPT_CONTINUE = """
Act as an expert academic note-taker and content summarizer. Below is another part of a long lecture transcript. Your task is to analyze this new content and seamlessly incorporate it into the existing notes, ensuring context is maintained and the final document is cohesive. The notes must be **written exclusively in simple English**, regardless of the input language.

Crucial Instructions:
* Analyze and Integrate: Carefully read the new text and integrate the information into the notes you have already created. Add new sections, sub-points, definitions, and examples as needed.
* Maintain Context: Do not treat this as a new request. Build directly upon the notes from the previous transcript part, which are provided below.
* Knowledge Augmentation: Continue to use your own knowledge to supplement and enrich the notes where necessary.
* Final Output: Provide the updated, more comprehensive set of notes. Do not provide a final summary or conclusion until all parts are provided.

---
Notes generated so far:
{existing_notes}
---
New Transcript Part:
{new_transcript_chunk}
"""

def generate_notes(video_url: str, resume_data=None) -> str:
    """
    Generates academic notes from a YouTube video.

    Args:
        video_url: The URL of the YouTube video.
        resume_data: Data to resume from a previous session.

    Returns:
        The generated notes as a string.
    """
    try:
        api_key = get_gemini_api_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')

        if resume_data:
            transcript_chunks = resume_data['transcript_chunks']
            notes = resume_data['notes']
            start_chunk = resume_data['chunk_index']
        else:
            print("Fetching transcript...")
            transcript = get_transcript(video_url)
            print("Preprocessing transcript...")
            preprocessed_transcript = preprocess_transcript(transcript)
            print("Chunking transcript...")
            transcript_chunks = chunk_text(preprocessed_transcript)
            notes = ""
            start_chunk = 0

        with tqdm(total=len(transcript_chunks), desc="Generating Notes", initial=start_chunk) as pbar:
            for i, chunk in enumerate(transcript_chunks[start_chunk:]):
                current_chunk_index = start_chunk + i
                try:
                    if current_chunk_index == 0:
                        prompt = PROMPT_START + chunk
                    else:
                        prompt = PROMPT_CONTINUE.format(existing_notes=notes, new_transcript_chunk=chunk)

                    response = model.generate_content(prompt)
                    notes = response.text
                    pbar.update(1)

                    # Save progress
                    progress = {
                        "video_url": video_url,
                        "transcript_chunks": transcript_chunks,
                        "notes": notes,
                        "chunk_index": current_chunk_index + 1
                    }
                    with open("progress.json", "w") as f:
                        json.dump(progress, f)

                except Exception as e:
                    print(f"An error occurred during note generation: {e}")
                    print("Saving progress and exiting.")
                    return notes

        return notes

    except Exception as e:
        return f"An error occurred: {e}"


def main():
    """Main function to run the program."""
    parser = argparse.ArgumentParser(description="YouTube Academic Note Generator")
    parser.add_argument("video_url", help="The URL of the YouTube video.", nargs='?')
    args = parser.parse_args()

    print("YouTube Academic Note Generator")
    resume_data = None
    video_url = args.video_url

    if os.path.exists("progress.json"):
        with open("progress.json", "r") as f:
            progress = json.load(f)

        if not video_url or video_url == progress['video_url']:
            while True:
                choice = input("A previous session was found. Do you want to (r)esume or (s)tart over? ").lower()
                if choice in ['r', 's']:
                    break
                print("Invalid choice. Please enter 'r' or 's'.")

            if choice == 'r':
                resume_data = progress
                video_url = progress['video_url']
                print("Resuming note generation...")
            else:
                os.remove("progress.json")
                print("Starting over...")
        else:
            print("A new video URL was provided, starting over...")
            os.remove("progress.json")


    if not video_url:
        video_url = input("Enter the YouTube video URL: ")

    notes = generate_notes(video_url, resume_data)

    if "An error occurred" not in notes:
        output_filename = "academic_notes.txt"
        with open(output_filename, "w") as f:
            f.write(notes)
        print(f"\nNotes successfully generated and saved to {output_filename}")
        if os.path.exists("progress.json"):
            os.remove("progress.json")
    else:
        print(f"\n{notes}")


if __name__ == "__main__":
    main()
