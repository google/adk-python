# How to Fix the Audio Transcription Issue

The audio transcription feature in the AI Novel Editor is not working because the `GOOGLE_API_KEY` has not been set. This key is required to use Google's Gemini API for audio transcription.

## To fix this, you need to:

1.  **Obtain a `GOOGLE_API_KEY`:**
    *   Go to the [Google AI for Developers](https://makersuite.google.com/app/apikey) website.
    *   Sign in with your Google account.
    *   Click on **"Create API key"**.

2.  **Set the `GOOGLE_API_KEY` in your environment:**
    *   In the `ai-novel-editor` directory, you will find a file named `.env.example`.
    *   Make a copy of this file and rename it to `.env`.
    *   Open the `.env` file and you will see the following line:
        ```
        GOOGLE_API_KEY=""
        ```
    *   Paste your newly created API key inside the quotes.

3.  **Restart the application:**
    *   Once you have saved the `.env` file, restart the Streamlit application.

The audio transcription feature should now work correctly.
