import re
import gradio as gr
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi

# Load summarization model
summarizer = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn"
)


def extract_video_id(url):
    """
    Extract video id from YouTube URL
    """
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"

    match = re.search(regex, url)

    if match:
        return match.group(1)

    return None


def get_transcript(video_id):
    """
    Fetch transcript using youtube-transcript-api v1.2.4
    """
    api = YouTubeTranscriptApi()

    transcript = api.fetch(
        video_id,
        languages=["en"]
    )

    transcript_text = " ".join(
        snippet.text for snippet in transcript
    )

    return transcript_text


def summarize_long_text(text):
    """
    Split large transcript into chunks
    and summarize each chunk.
    """

    chunk_size = 1000

    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]

    summaries = []

    for chunk in chunks:

        result = summarizer(
            chunk,
            max_length=120,
            min_length=40,
            do_sample=False
        )

        summaries.append(
            result[0]["summary_text"]
        )

    combined_summary = " ".join(summaries)

    if len(combined_summary) > 1000:

        final_summary = summarizer(
            combined_summary,
            max_length=150,
            min_length=50,
            do_sample=False
        )

        return final_summary[0]["summary_text"]

    return combined_summary


def youtube_summarizer(video_url):

    try:

        video_id = extract_video_id(video_url)

        if not video_id:
            return "❌ Invalid YouTube URL"

        transcript_text = get_transcript(video_id)

        if not transcript_text:
            return "❌ No transcript found"

        summary = summarize_long_text(transcript_text)

        return summary

    except Exception as e:
        return f"❌ Error: {str(e)}"


# Gradio Interface
demo = gr.Interface(
    fn=youtube_summarizer,
    inputs=gr.Textbox(
        label="YouTube Video URL",
        placeholder="Paste YouTube URL here..."
    ),
    outputs=gr.Textbox(
        label="Summary",
        lines=10
    ),
    title="YouTube Transcript Summarizer",
    description="Paste a YouTube URL and get an AI-generated summary."
)

if __name__ == "__main__":
    demo.launch()
