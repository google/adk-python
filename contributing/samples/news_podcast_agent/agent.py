from typing import Dict, List, Optional
import pathlib
import wave
import re
from urllib.parse import urlparse
from datetime import datetime, timedelta
import base64
import json
import time

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import ToolContext
from google.adk.tools.google_search_agent_tool import google_search
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import html2text
from bs4 import BeautifulSoup
import yfinance as yf
import requests
from urllib.parse import urljoin

class NewsletterStory(BaseModel):
    """A single news story extracted from a newsletter."""
    newsletter_name: str = Field(description="Name of the newsletter source (e.g., 'Morning Brew', 'The Hustle')")
    newsletter_sender: str = Field(description="Email sender of the newsletter")
    story_title: str = Field(description="Title or headline of the story")
    company: str = Field(description="Company name associated with the story (e.g., 'Nvidia', 'OpenAI'). Use 'N/A' if not applicable.")
    ticker: str = Field(description="Stock ticker for the company (e.g., 'NVDA'). Use 'N/A' if private or not found.")
    summary: str = Field(description="A brief, one-sentence summary of the news story.")
    why_it_matters: str = Field(description="A concise explanation of the story's significance or impact.")
    financial_context: str = Field(description="Current stock price and change, e.g., '$950.00 (+1.5%)'. Use 'No financial data' if not applicable.")
    received_date: str = Field(description="Date when the newsletter was received")
    process_log: str = Field(description="Processing notes about how this story was extracted")

class NewsletterReport(BaseModel):
    """A structured report of newsletter-based news."""
    title: str = Field(default="Newsletter News Report", description="The main title of the report.")
    report_summary: str = Field(description="A brief, high-level summary of the key findings from newsletters.")
    stories: List[NewsletterStory] = Field(description="A list of the individual news stories found in newsletters.")
    newsletters_processed: List[str] = Field(description="List of newsletter names that were processed")
    total_newsletters: int = Field(description="Total number of newsletters processed")

# Gmail API configuration
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
NEWSLETTER_SENDERS = [
    'morningbrew.com', 'thehustle.co', 'axios.com', 'techcrunch.com',
    'venturebeat.com', 'theverge.com', 'arstechnica.com', 'wired.com',
    'bloomberg.com', 'reuters.com', 'wsj.com', 'ft.com', 'cnn.com',
    'bbc.com', 'npr.org', 'nytimes.com', 'washingtonpost.com',
    'theguardian.com', 'forbes.com', 'businessinsider.com', 'cnbc.com',
    'marketwatch.com', 'yahoo.com', 'msn.com', 'tldr.tech', 'tldrnewsletter.com'
]

def get_today_date():
    """Get today's date in the format required by Gmail API."""
    try:
        # Get current date
        today = datetime.now()
        
        # Format for Gmail API (YYYY/MM/DD)
        formatted_date = today.strftime('%Y/%m/%d')
        
        # Also try alternative formats in case Gmail API is picky
        alt_formats = [
            today.strftime('%Y/%m/%d'),
            today.strftime('%Y-%m-%d'),
            today.strftime('%Y%m%d')
        ]
        
        return {
            "status": "success",
            "primary_date": formatted_date,
            "alt_dates": alt_formats,
            "raw_date": today.isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get date: {str(e)}"
        }

def get_gmail_service():
    """Initialize Gmail API service with OAuth2 authentication."""
    creds = None
    token_file = 'token.json'
    credentials_file = 'credentials.json'
    
    # Load existing credentials
    if pathlib.Path(token_file).exists():
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    
    # If no valid credentials, request authorization
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not pathlib.Path(credentials_file).exists():
                return {"status": "error", "message": "credentials.json file not found. Please download it from Google Cloud Console."}
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for next run
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
    
    try:
        service = build('gmail', 'v1', credentials=creds)
        return {"status": "success", "service": service}
    except Exception as e:
        return {"status": "error", "message": f"Failed to initialize Gmail service: {str(e)}"}

def extract_html_part(part):
    """Recursively extract HTML content from MIME parts."""
    if part.get('mimeType') == 'text/html':
        if 'data' in part.get('body', {}):
            return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
    elif part.get('mimeType') == 'text/plain':
        if 'data' in part.get('body', {}):
            return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')

    # Check subparts recursively
    if 'parts' in part:
        for subpart in part['parts']:
            html_content = extract_html_part(subpart)
            if html_content:
                return html_content

    return None

def extract_both_formats(payload):
    """Extract both HTML and plain text versions from email payload."""
    html_content = None
    plain_content = None

    def extract_recursive(part):
        nonlocal html_content, plain_content

        if part.get('mimeType') == 'text/html':
            if 'data' in part.get('body', {}):
                html_content = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
        elif part.get('mimeType') == 'text/plain':
            if 'data' in part.get('body', {}):
                plain_content = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')

        if 'parts' in part:
            for subpart in part['parts']:
                extract_recursive(subpart)

    extract_recursive(payload)
    return html_content, plain_content

def extract_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML content, preserving line breaks."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text with separator to preserve structure
        text = soup.get_text(separator='\n')

        # Clean up excessive blank lines but preserve single line breaks
        lines = [line.strip() for line in text.splitlines()]
        # Remove excessive consecutive blank lines
        cleaned_lines = []
        prev_blank = False
        for line in lines:
            if line:
                cleaned_lines.append(line)
                prev_blank = False
            elif not prev_blank:
                cleaned_lines.append('')
                prev_blank = True

        return '\n'.join(cleaned_lines)
    except Exception as e:
        return html_content  # Return original if parsing fails

def is_valid_newsletter(sender: str, subject: str, content: str, headers: List[Dict] = None) -> bool:
    """Check if an email is a valid newsletter (not promotional/webinar)."""
    sender_lower = sender.lower()
    subject_lower = subject.lower()
    content_lower = content.lower()

    # Check for List-ID header (present in almost all newsletters)
    if headers:
        list_id = next((h['value'] for h in headers if h['name'] == 'List-ID'), None)
        if list_id:
            return True  # If List-ID exists, it's likely a newsletter

    # Check if it's from a known newsletter domain - if yes, auto-accept
    is_newsletter_domain = any(domain in sender_lower for domain in NEWSLETTER_SENDERS)
    if is_newsletter_domain:
        return True  # Trust known newsletter sources

    # For unknown senders, be more selective
    # Exclude obvious promotional keywords (reduced list)
    promotional_keywords = [
        'webinar registration', 'join our webinar', 'register now', 'limited time offer',
        'exclusive offer', 'claim your discount', 'act now', 'hurry',
        'hackathon registration', 'event registration', 'rsvp now'
    ]

    # Newsletter indicators
    newsletter_keywords = [
        'newsletter', 'daily', 'weekly', 'digest', 'roundup', 'briefing',
        'news', 'report', 'summary', 'recap', 'update', 'bulletin'
    ]

    # Check if it contains strong promotional content (subject only for unknown senders)
    is_promotional = any(keyword in subject_lower for keyword in promotional_keywords)

    # Check if it looks like a newsletter
    looks_like_newsletter = any(keyword in subject_lower for keyword in newsletter_keywords)

    # Special case for TLDR newsletters - always valid
    is_tldr = 'tldr' in sender_lower or 'tldr' in subject_lower
    if is_tldr:
        return True

    # It's a valid newsletter if it looks like a newsletter and is not strongly promotional
    return looks_like_newsletter and not is_promotional

def parse_newsletter_content(email_content: str, sender: str, subject: str, plain_text: str = None) -> List[Dict]:
    """Parse newsletter content to extract individual stories.

    Args:
        email_content: HTML version of email (for URL extraction)
        sender: Email sender
        subject: Email subject
        plain_text: Plain text version of email (for story parsing, optional)
    """
    stories = []

    try:
        # First extract URLs from HTML before converting to text
        soup = BeautifulSoup(email_content, 'html.parser')
        url_map = {}  # Map story titles to URLs
        story_urls = []  # Ordered list of story URLs

        # Find all links in the HTML - TLDR newsletters have article links in a specific pattern
        for link in soup.find_all('a', href=True):
            link_text = link.get_text().strip()
            href = link['href']

            # Skip tracking/unsubscribe links
            if any(skip in href for skip in ['unsubscribe', 'tldr.tech', 'actions.tldrnewsletter', 'preferences']):
                continue

            # Store the mapping of text to URL
            if link_text and href and href.startswith('http'):
                url_map[link_text] = href
                # Also store in ordered list for positional matching
                story_urls.append(href)

        # DEBUG: Log extracted URLs
        print(f"DEBUG: Extracted {len(url_map)} URLs from newsletter HTML ({len(story_urls)} story URLs)")
        if url_map:
            for i, (link_text, url) in enumerate(list(url_map.items())[:5]):  # Show first 5
                print(f"  URL {i+1}: '{link_text[:60]}...' -> {url[:80]}...")
        else:
            print("  WARNING: No URLs found in newsletter HTML!")

        # Use plain text for parsing if available, otherwise extract from HTML
        if plain_text:
            clean_text = plain_text
            print(f"DEBUG: Using plain text version for story parsing ({len(clean_text)} chars)")
        else:
            clean_text = extract_text_from_html(email_content)
            print(f"DEBUG: Using HTML-to-text conversion for story parsing ({len(clean_text)} chars)")

        # For TLDR newsletters, parse the structured format BEFORE normalizing whitespace
        # TLDR format: "ALL CAPS HEADLINE (X MINUTE READ) [link] \n Content here..."
        if 'tldr' in sender.lower():
            # Clean up zero-width chars but preserve newlines
            clean_text = re.sub(r'‌', '', clean_text)  # Remove zero-width chars
            clean_text = re.sub(r' +', ' ', clean_text)  # Multiple spaces to single space

            # Split text into lines first to preserve structure
            lines = clean_text.split('\n')

            i = 0
            while i < len(lines) and len(stories) < 5:
                line = lines[i].strip()

                # Look for headline pattern: ALL CAPS with (X MINUTE READ) and [link]
                # Example: "CHATGPT ATLAS (4 MINUTE READ) [5]"
                headline_pattern = r'^([A-Z][A-Z\s&\',AI-]+)\s*\((\d+)\s+MINUTE\s+READ\)\s*\[\d+\]'
                match = re.match(headline_pattern, line)

                if match:
                    headline = match.group(1).strip()
                    read_time = match.group(2)

                    # Collect content from following lines until next headline or empty lines
                    content_lines = []
                    j = i + 1
                    while j < len(lines):  # Remove line limit - get all content
                        next_line = lines[j].strip()

                        # Stop at next headline or section marker
                        if re.match(headline_pattern, next_line):
                            break
                        if re.match(r'^[🚀🧠💼📱🎯🔥]+\s*$', next_line):  # Emoji section markers
                            break
                        if re.match(r'^[A-Z\s&]+$', next_line) and len(next_line) > 20:  # Section headers
                            break

                        if next_line and len(next_line) > 20:  # Skip very short lines
                            content_lines.append(next_line)

                        j += 1

                    if content_lines:
                        content = ' '.join(content_lines)

                        # Keep full content - no summary truncation
                        # (Removed 3-sentence summary extraction)

                        # Extract company names
                        company_patterns = [
                            r'\b(Google|Amazon|Microsoft|Apple|Meta|OpenAI|Anthropic|Tesla|Nvidia|AMD|Intel|AWS|DeepSeek|ChatGPT)\b',
                            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:announced|launched|released|unveiled|introduced|revealed)',
                            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\'s\s+',
                        ]

                        company = "N/A"
                        for pattern in company_patterns:
                            company_match = re.search(pattern, content)
                            if company_match:
                                company = company_match.group(1)
                                break

                        # Try to find URL for this story
                        story_url = None

                        # Method 1: Try exact headline match
                        for link_text, url in url_map.items():
                            if headline.lower() in link_text.lower() or link_text.lower() in headline.lower():
                                story_url = url
                                break

                        # Method 2: Try content-based matching (find keywords in link text)
                        if not story_url:
                            headline_words = set(headline.lower().split())
                            best_match = 0
                            best_url = None
                            for link_text, url in url_map.items():
                                link_words = set(link_text.lower().split())
                                match_count = len(headline_words & link_words)
                                if match_count > best_match and match_count >= 2:  # At least 2 words match
                                    best_match = match_count
                                    best_url = url
                            story_url = best_url

                        # Method 3: Positional matching - use story index to get URL from ordered list
                        if not story_url and story_urls and len(stories) < len(story_urls):
                            story_url = story_urls[len(stories)]

                        print(f"DEBUG: Story '{headline[:50]}...' content length: {len(content)} chars, URL: {story_url[:80] if story_url else 'None'}...")  # Debug trace
                        stories.append({
                            "title": headline,  # Keep full title
                            "content": content,  # Keep full content - no truncation
                            "company": company,
                            "newsletter": sender,
                            "subject": subject,
                            "url": story_url if story_url else "N/A"  # Include article URL
                        })
                        i = j  # Skip to where we left off
                    else:
                        # No content found for this headline, move to next line
                        i += 1
                else:
                    i += 1

        else:
            # Generic parsing for other newsletters
            # Normalize whitespace for non-TLDR newsletters
            clean_text = re.sub(r'\s+', ' ', clean_text)
            clean_text = re.sub(r'\s*‌\s*', ' ', clean_text)

            # Split on paragraph breaks and bullet points
            story_sections = re.split(r'(?:\n\s*\n|[•·▪▫]\s+|\d+\.\s+)', clean_text)

            for section in story_sections:
                section = section.strip()
                if len(section) < 100:  # Skip short sections
                    continue

                lines = [l.strip() for l in section.split('.') if l.strip()]
                title = lines[0][:200] if lines else "Untitled Story"

                # Look for company mentions
                company_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', section)
                company = company_match.group(1) if company_match else "N/A"

                stories.append({
                    "title": title,
                    "content": section[:1000],
                    "company": company,
                    "newsletter": sender,
                    "subject": subject
                })

                if len(stories) >= 5:
                    break

        return stories if stories else [{
            "title": subject,
            "content": clean_text[:500],
            "company": "N/A",
            "newsletter": sender,
            "subject": subject
        }]

    except Exception as e:
        return [{"title": "Parse Error", "content": f"Failed to parse newsletter: {str(e)}", "company": "N/A", "newsletter": sender, "subject": subject}]

def test_gmail_connection(tool_context: ToolContext) -> Dict[str, any]:
    """Test Gmail API connection and get basic inbox info."""
    try:
        # Initialize Gmail service
        gmail_result = get_gmail_service()
        if gmail_result["status"] != "success":
            return gmail_result
        
        service = gmail_result["service"]
        
        # Get user profile
        profile = service.users().getProfile(userId='me').execute()
        email = profile.get('emailAddress', 'Unknown')
        
        # Get today's date
        today = datetime.now().strftime('%Y/%m/%d')
        
        # Get recent emails (last 10)
        results = service.users().messages().list(userId='me', maxResults=10).execute()
        messages = results.get('messages', [])
        
        recent_emails = []
        for message in messages[:5]:  # Just get first 5 for testing
            try:
                msg = service.users().messages().get(userId='me', id=message['id']).execute()
                headers = msg['payload'].get('headers', [])
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
                date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown Date')
                
                recent_emails.append({
                    "subject": subject,
                    "sender": sender,
                    "date": date
                })
            except Exception as e:
                continue
        
        return {
            "status": "success",
            "email": email,
            "today_date": today,
            "recent_emails": recent_emails,
            "total_recent": len(messages)
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Failed to test Gmail connection: {str(e)}"
        }

def fetch_newsletters_from_inbox(tool_context: ToolContext) -> Dict[str, any]:
    """Fetch newsletters from Gmail inbox for the current day."""
    newsletters = []
    process_log = []
    
    try:
        # Initialize Gmail service
        gmail_result = get_gmail_service()
        if gmail_result["status"] != "success":
            process_log.append(f"Gmail service error: {gmail_result.get('message', 'Unknown error')}")
            return {
                "status": "error",
                "message": gmail_result.get('message', 'Unknown error'),
                "newsletters": [],
                "process_log": process_log,
                "total_processed": 0
            }
        
        service = gmail_result["service"]
        
        # Get today's date using our dedicated function
        date_result = get_today_date()
        if date_result["status"] != "success":
            return {
                "status": "error",
                "message": f"Failed to get date: {date_result['message']}",
                "newsletters": [],
                "process_log": [f"Date error: {date_result['message']}"],
                "total_processed": 0
            }
        
        today = date_result["primary_date"]
        query = f'newer_than:1d'
        
        # First try: Search for emails from known newsletter domains
        sender_query = ' OR '.join([f'from:{sender}' for sender in NEWSLETTER_SENDERS])
        query_with_senders = f'{query} AND ({sender_query})'
        
        # Search for emails
        results = service.users().messages().list(userId='me', q=query_with_senders, maxResults=50).execute()
        messages = results.get('messages', [])
        
        process_log.append(f"Found {len(messages)} emails from known newsletter domains")
        
        # If no emails found, try broader search for newsletter-like emails
        if len(messages) == 0:
            process_log.append("No emails from known domains, trying broader search...")
            # Search for emails that might be newsletters (contain "newsletter", "daily", "weekly", etc.)
            broader_query = f'{query} AND (subject:newsletter OR subject:daily OR subject:weekly OR subject:digest OR subject:roundup OR subject:briefing OR from:tldr OR subject:tldr)'
            results = service.users().messages().list(userId='me', q=broader_query, maxResults=50).execute()
            messages = results.get('messages', [])
            process_log.append(f"Found {len(messages)} emails from broader search")
        
        # If still no emails, try even broader search
        if len(messages) == 0:
            process_log.append("Still no emails, trying even broader search...")
            # Just search for recent emails and filter manually
            results = service.users().messages().list(userId='me', q=query, maxResults=100).execute()
            messages = results.get('messages', [])
            process_log.append(f"Found {len(messages)} total recent emails to filter manually")

        # Final count after all search attempts
        process_log.append(f"Total messages to process: {len(messages)}")
        
        for message in messages:
            try:
                # Get full message
                msg = service.users().messages().get(userId='me', id=message['id']).execute()
                
                headers = msg['payload'].get('headers', [])
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
                date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown Date')
                
                # Extract both HTML and plain text versions
                html_body, plain_body = extract_both_formats(msg['payload'])

                # DEBUG: Check what we extracted
                print(f"DEBUG: Extracted HTML: {len(html_body) if html_body else 0} chars, Plain: {len(plain_body) if plain_body else 0} chars")

                # Use whichever is available (prefer HTML for URL extraction initially)
                body = html_body or plain_body or ""

                # DEBUG: Check if body contains HTML tags
                has_html_tags = bool(re.search(r'<[^>]+>', body[:500]))  # Check first 500 chars
                print(f"DEBUG: Email from {sender[:30]}... has HTML tags: {has_html_tags}, body length: {len(body)}")

                # Check if this is a valid newsletter (not promotional)
                is_valid = is_valid_newsletter(sender, subject, body, headers)
                process_log.append(f"Email from {sender}: '{subject}' - Valid: {is_valid}")

                if not is_valid:
                    process_log.append(f"Skipped promotional email from {sender}: {subject}")
                    continue

                # Parse newsletter content - pass both HTML and plain text versions
                # HTML for URL extraction, plain text for story parsing
                stories = parse_newsletter_content(html_body or body, sender, subject, plain_body)
                
                newsletters.append({
                    "sender": sender,
                    "subject": subject,
                    "date": date,
                    "stories": stories
                })
                
                process_log.append(f"Processed newsletter from {sender}: {len(stories)} stories found")
                
            except Exception as e:
                process_log.append(f"Error processing message {message['id']}: {str(e)}")
                continue
        
        return {
            "status": "success",
            "newsletters": newsletters,
            "process_log": process_log,
            "total_processed": len(newsletters)
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Failed to fetch newsletters: {str(e)}",
            "newsletters": [],
            "process_log": [f"Error: {str(e)}"],
            "total_processed": 0
        }

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Helper function to save audio data as a wave file"""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
        

async def generate_podcast_audio(podcast_script: str, tool_context: ToolContext, filename: str = "'ai_today_podcast") -> Dict[str, str]:
    """
    Generates audio from a podcast script using Gemini API and saves it as a WAV file.
    Includes retry logic for handling API overload errors.

    Args:
        podcast_script: The conversational script to be converted to audio.
        tool_context: The ADK tool context.
        filename: Base filename for the audio file (without extension).

    Returns:
        Dictionary with status and file information.
    """
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            # Get API key from environment
            import os
            api_key = os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")

            client = genai.Client(api_key=api_key)
            prompt = f"TTS the following conversation between Joe and Jane:\n\n{podcast_script}"

            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                            speaker_voice_configs=[
                                types.SpeakerVoiceConfig(speaker='Joe',
                                                         voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Kore'))),
                                types.SpeakerVoiceConfig(speaker='Jane',
                                                         voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Puck')))
                            ]
                        )
                    )
                )
            )

            data = response.candidates[0].content.parts[0].inline_data.data

            if not filename.endswith(".wav"):
                filename += ".wav"

            # ** BUG FIX **: This logic now runs for all cases, not just when the extension is added.
            current_directory = pathlib.Path.cwd()
            file_path = current_directory / filename
            wave_file(str(file_path), data)

            return {
                "status": "success",
                "message": f"Successfully generated and saved podcast audio to {file_path.resolve()}",
                "file_path": str(file_path.resolve()),
                "file_size": len(data)
            }

        except Exception as e:
            error_msg = str(e)

            # Check if it's a 503/overload error
            if "503" in error_msg or "overloaded" in error_msg.lower() or "UNAVAILABLE" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"⚠️  API overloaded, retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return {"status": "error", "message": f"Audio generation failed after {max_retries} attempts: API overloaded. Please try again later."}

            # For other errors, don't retry
            return {"status": "error", "message": f"Audio generation failed: {error_msg[:200]}"}

def get_financial_context(tickers: List[str]) -> Dict[str, str]:
    """
    Fetches the current stock price and daily change for a list of stock tickers.
    """
    financial_data: Dict[str, str] = {}

    # Filter out invalid tickers upfront
    valid_tickers = [ticker.upper().strip() for ticker in tickers 
                    if ticker and ticker.upper() not in ['N/A', 'NA', '']]
    
    if not valid_tickers:
        return {ticker: "No financial data" for ticker in tickers}
        
    for ticker_symbol in valid_tickers:
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            change_percent = info.get("regularMarketChangePercent")
            
            if price is not None and change_percent is not None:
                change_str = f"{change_percent * 100:+.2f}%"
                financial_data[ticker_symbol] = f"${price:.2f} ({change_str})"
            else:
                financial_data[ticker_symbol] = "Price data not available."
        except Exception:
            financial_data[ticker_symbol] = "Invalid Ticker or Data Error"
            
    return financial_data

def extract_links_from_html(html_content: str, base_url: str = "") -> List[Dict[str, str]]:
    """Extract all links from HTML content."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []

        for link in soup.find_all('a', href=True):
            url = link['href']
            text = link.get_text(strip=True)

            # Skip empty links, anchors, and mailto links
            if not url or url.startswith('#') or url.startswith('mailto:'):
                continue

            # Make relative URLs absolute
            if base_url and not url.startswith('http'):
                url = urljoin(base_url, url)

            # Filter out common newsletter junk links
            skip_patterns = ['unsubscribe', 'preferences', 'privacy', 'terms', 'manage-email', 'email-settings']
            if any(pattern in url.lower() for pattern in skip_patterns):
                continue

            if url.startswith('http'):
                links.append({
                    "url": url,
                    "text": text[:200] if text else "No text"
                })

        return links
    except Exception as e:
        return []

def fetch_article_content(url: str, tool_context: ToolContext) -> Dict[str, any]:
    """
    Fetch and extract main content from a web article URL.
    Returns the article title, content, and metadata.
    """
    try:
        # Set a timeout and user agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title = soup.find('title')
        title_text = title.get_text(strip=True) if title else "No title"

        # Try to find article content using common tags
        article_content = None
        for tag in ['article', 'main', 'div[class*="content"]', 'div[class*="article"]']:
            article_content = soup.select_one(tag)
            if article_content:
                break

        # If no article tag found, use body
        if not article_content:
            article_content = soup.find('body')

        # Remove script, style, nav, footer, ads
        if article_content:
            for element in article_content(['script', 'style', 'nav', 'footer', 'aside', 'header']):
                element.decompose()

            # Get text content
            text = article_content.get_text(separator='\n', strip=True)

            # Clean up excessive whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            clean_text = '\n'.join(lines)

            # Truncate to reasonable length (first 2000 characters)
            if len(clean_text) > 2000:
                clean_text = clean_text[:2000] + "..."

            return {
                "status": "success",
                "url": url,
                "title": title_text,
                "content": clean_text,
                "length": len(clean_text)
            }
        else:
            return {
                "status": "error",
                "message": "Could not find article content",
                "url": url
            }

    except requests.Timeout:
        return {
            "status": "error",
            "message": "Request timed out",
            "url": url
        }
    except requests.RequestException as e:
        return {
            "status": "error",
            "message": f"Failed to fetch article: {str(e)[:200]}",
            "url": url
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing article: {str(e)[:200]}",
            "url": url
        }

def save_news_to_markdown(filename: str, content: str) -> Dict[str, str]:
    """
    Saves the given content to a Markdown file in the current directory.
    """
    try:
        if not filename.endswith(".md"):
            filename += ".md"
        current_directory = pathlib.Path.cwd()
        file_path = current_directory / filename
        file_path.write_text(content, encoding="utf-8")
        return {
            "status": "success",
            "message": f"Successfully saved news to {file_path.resolve()}",
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to save file: {str(e)}"}


# Script Generator Agent - Creates the podcast script
script_generator_agent = Agent(
    name="script_generator_agent",
    model="gemini-2.5-flash",
    instruction="""
    You are a Podcast Script Writer. Your single job is to take newsletter data and additional research context,
    and create a complete, engaging podcast script between two co-hosts: Joe and Jane.

    **Your Input:**
    You will receive:
    1. Newsletter data with stories, companies, and financial context
    2. Additional research context from web searches and article fetches
    3. The number of stories to cover

    **Your Output:**
    A complete podcast script ready for audio generation.

    **CRITICAL LENGTH REQUIREMENTS:**
    - The script MUST be AT LEAST 4000-6000 words (approximately 20-30 minutes of audio)
    - Cover ALL stories provided - do NOT skip any
    - Each story needs 10-12 exchanges MINIMUM between Joe and Jane
    - If you have 10 stories, that's 100-120 exchanges just for stories, plus opening (5 exchanges) and closing (5 exchanges)
    - Count your exchanges as you write to ensure you meet minimums

    **CRITICAL FORMATTING REQUIREMENTS:**
    The script MUST use this EXACT format for EVERY line of dialogue:

    Joe: [Complete sentence with proper punctuation.]
    Jane: [Complete sentence with proper punctuation.]
    Joe: [Complete sentence with proper punctuation.]
    Jane: [Complete sentence with proper punctuation.]

    DO NOT use any other format. DO NOT add stage directions, timestamps, or scene descriptions.

    **Script Structure:**
    1. Opening (5 exchanges welcoming listeners, setting the stage)
    2. For EACH story (10-12 exchanges per story):
       - Joe introduces the headline with excitement and context
       - Jane asks a clarifying or challenging question
       - Joe provides more details from the research
       - Jane offers analysis or counterpoint
       - They discuss implications together with specific examples
       - They debate different perspectives
       - Jane brings up concerns or alternative viewpoints
       - Joe responds with data or optimistic takes
       - They explore "what this means for" various stakeholders
       - They discuss future implications
       - They transition naturally to next story
    3. Closing (5 exchanges thanking listeners, teasing next episode)

    **Dialogue Quality Requirements - ABSOLUTELY CRITICAL:**
    - Both hosts MUST contribute EQUALLY - Joe and Jane MUST have the EXACT SAME number of lines
    - After writing the script, COUNT: Joe's lines = Jane's lines (50/50 split is MANDATORY)
    - Each story MUST have 10-12 exchanges MINIMUM (not 6-8, but 10-12!)
    - Jane MUST speak in COMPLETE, SUBSTANTIVE sentences - NO short affirmations like "yes", "right", "interesting"
    - Jane's lines should be AS LONG as Joe's lines - she is an equal co-host, not a sidekick
    - Include natural reactions: "Wait, really?", "Hold on, that's interesting because...", "But here's what worries me..."
    - Add disagreements and friendly debates where Jane challenges Joe's optimism
    - Use follow-up questions and expansions from BOTH hosts
    - Never cut stories short - complete the full discussion with proper transitions
    - If Joe has 60 lines, Jane MUST have 60 lines - no exceptions!

    **Character Personalities:**
    - Joe: Enthusiastic tech optimist, gets excited about innovation, sees possibilities, focuses on benefits
    - Jane: Analytical skeptic, asks tough questions, considers risks, provides balance

    **Example CORRECT Exchange (10+ exchanges per story):**
    Joe: "Welcome back everyone! I've got some breaking news that's going to blow your mind - OpenAI just launched ChatGPT Atlas, their very own AI-powered web browser!"
    Jane: "Okay, so let me get this straight - they're going head-to-head with Chrome, Edge, and Safari? That's a massive undertaking for a company that's been focused on APIs and language models."
    Joe: "Exactly! And here's the kicker - it has a sidebar with ChatGPT that's contextually aware of everything on your screen. It can help you research, summarize articles, even help you write emails based on what you're reading."
    Jane: "That does sound convenient, but I have to ask - how does this actually change the browser wars? Google has been dominating with Chrome for years, and they have deep integration with their services."
    Joe: "Great question! The thing is, Google must be nervous because Microsoft already has Copilot baked into Edge. This could really fragment the market and force everyone to up their AI game."
    Jane: "Right, but let's think about the average user for a second. Do we really need another browser? Most people are comfortable with what they have. What makes Atlas special enough to convince someone to switch?"
    Joe: "Fair point! But imagine this - you're doing research for a project, and instead of copy-pasting into a separate chat window, the AI is right there, understanding the context of every tab you have open."
    Jane: "Hmm, I see the appeal for productivity, especially for researchers or writers. Though I'm wondering about privacy implications - is ChatGPT seeing and analyzing everything you browse? That's a lot of personal data."
    Joe: "That's a great question and one I looked into. OpenAI says it's all optional and you control what gets shared. But yeah, that's definitely something privacy-conscious users will scrutinize closely."
    Jane: "And what about performance? Chrome is already a memory hog. Are we talking about an even heavier browser with all this AI running in the background?"
    Joe: "Actually, early reports suggest it's surprisingly lightweight because they optimized it from the ground up. But you're right to be concerned - we'll need to see real-world usage data."
    Jane: "Alright, well I'm cautiously interested. Let's see how this plays out. Now, speaking of AI integration, let's move to our next story about Amazon's automation push..."

    **Example WRONG Exchange (TOO SHORT - DO NOT DO THIS):**
    Joe: "OpenAI launched ChatGPT Atlas."
    Jane: "That's interesting."
    Joe: "It has AI features."
    Jane: "Cool, next story."

    **MANDATORY FINAL STEP:**
    After creating the script, you MUST save it using the `save_news_to_markdown` tool with filename `podcast_script.md`.
    Then return the COMPLETE script text so it can be passed to the audio generation agent.
    """,
    tools=[save_news_to_markdown],
)

# Audio Generator Agent - Converts script to audio
podcaster_agent = Agent(
    name="podcaster_agent",
    model="gemini-2.5-flash",
    instruction="""
    You are an Audio Generation Specialist. Your single task is to take a provided text script
    and convert it into a multi-speaker audio file using the `generate_podcast_audio` tool.

    CRITICAL: You must pass the ENTIRE script to the audio generation tool, not a summary or shortened version.

    Workflow:
    1. Receive the text script from the user or another agent.
    2. Immediately call the `generate_podcast_audio` tool with the COMPLETE provided script and the filename of 'ai_today_podcast'
    3. Report the result of the audio generation back to the user.
    """,
    tools=[generate_podcast_audio],
)

root_agent = Agent(
    name="newsletter_podcast_producer",
    model="gemini-2.5-flash", 
    instruction="""
    **Your Core Identity:**
    You are a Newsletter Podcast Producer. Your job is to orchestrate a complete workflow: fetch newsletters from the user's inbox, extract news stories, compile a report, write a script, and generate a podcast audio file, all while keeping the user informed.

    **Crucial Rules:**
    1.  **Resilience is Key:** If you encounter an error or cannot find specific information for one item (like fetching a stock ticker), you MUST NOT halt the entire process. Use a placeholder value like "Not Available", and continue to the next step. Your primary goal is to deliver the final report and podcast, even if some data points are missing.
    2.  **Newsletter Focus:** Your research is based on newsletters received in the user's inbox today. Focus on extracting meaningful business and tech news stories from newsletter content.
    3.  **User-Facing Communication:** Your interaction has only two user-facing messages: the initial acknowledgment and the final confirmation. All complex work must happen silently in the background between these two messages.

    **Understanding Newsletter Tool Outputs:**
    The `fetch_newsletters_from_inbox` tool returns a JSON object with these keys:
    1.  `newsletters`: A list of newsletter objects with sender, subject, date, and stories.
    2.  `process_log`: A list of strings describing the processing actions performed.
    3.  `total_processed`: Number of newsletters processed.

    **Required Conversational Workflow:**
    1.  **Acknowledge and Inform:** The VERY FIRST thing you do is respond to the user with: "Okay, I'll scan your inbox for today's newsletters, extract the key news stories, research them in-depth, enrich them with financial data where available, and compile a podcast for you. This might take a moment."

    2.  **Fetch Newsletters (Background Step):** Immediately after acknowledging, use the `fetch_newsletters_from_inbox` tool to get today's newsletters from the user's inbox.

    3.  **Analyze & Extract Stories (Internal Step):** Process newsletter content to identify individual news stories, company names, and potential stock tickers. If a company is not publicly traded or a ticker cannot be found, use 'N/A'.

    4.  **MANDATORY Deep Research (Background Step) - ABSOLUTELY CRITICAL:**
        **YOU MUST EXECUTE THIS STEP - NO EXCEPTIONS:**

        For EVERY story (not just top 5-10, but ALL stories), you MUST:

        a) **Extract URLs from newsletter content:**
           - Look for article links in the newsletter HTML/text
           - These are usually "Read more" links or embedded in story titles
           - Each TLDR newsletter story has a link - you MUST find and extract them

        b) **Fetch full article content:**
           - Call `fetch_article_content` with each extracted URL
           - This gives you the COMPLETE article text, not just the newsletter snippet
           - The newsletter only has 2-3 sentences - the FULL article has 10-20 paragraphs
           - Log each fetch: "Fetching article: [URL]"

        c) **Search for additional context:**
           - Use `google_search` with the story title to find related articles, analysis, reactions
           - This provides broader context and multiple perspectives
           - Log each search: "Searching for: [query]"

        d) **Record all findings:**
           - Store all fetched content and search results
           - Pass ALL of this enriched data to the script generator
           - The script generator needs FULL articles to create detailed discussions

        **ENFORCEMENT:**
        - You MUST fetch at least 10 article URLs (one per story minimum)
        - If you skip this step, the podcasts will be SHORT and INCOMPLETE
        - Log EVERY research action in process_log
        - If a tool fails, log it and try the next story - but DO NOT skip research entirely

    5.  **Get Financial Data (Background Step):** Call the `get_financial_context` tool with the extracted tickers. If the tool returns "Not Available" for any ticker, you will accept this and proceed. Do not stop or report an error.

    6.  **Structure the Report (Internal Step):** Use the `NewsletterReport` schema to structure all gathered information. Include any additional context gathered from article fetching or web searches. If financial data was not found for a story, you MUST use "Not Available" in the `financial_context` field. You MUST also populate the `process_log` field with processing notes, including all research actions taken.

    7.  **Format for Markdown (Internal Step):** Convert the structured `NewsletterReport` data into a well-formatted Markdown string. This MUST include a section at the end called "## Newsletter Processing Notes" where you list the items from the `process_log`.

    8.  **Save the Report (Background Step):** Save the Markdown string using `save_news_to_markdown` with the filename `newsletter_report.md`.

    9.  **CRITICAL: Delegate Script Creation - DO NOT CREATE SCRIPTS YOURSELF:**
        You MUST NOT create podcast scripts inline. You MUST call the `script_generator_agent` tool.

        Pass to the script generator agent:
        - A summary of all newsletters and stories you found
        - All research context you gathered (article fetches, search results)
        - Explicit instruction: "Create a complete podcast script with 10-12 exchanges per story, ensuring Joe and Jane speak equally"

        The script_generator_agent will:
        - Create the full podcast script (4000-6000 words)
        - Save it to `podcast_script.md`
        - Return the complete script text to you

        **MANDATORY**: You must wait for the script_generator_agent to complete and return the script before proceeding.

    10. **Generate Audio (Background Step):** After receiving the complete script from script_generator_agent, call the `podcaster_agent` tool with the EXACT script text you received. DO NOT modify, summarize, or truncate it in any way.

    11. **Final Confirmation:** After the audio is successfully generated, your final response to the user MUST be: "All done! I've processed your newsletters, researched the stories in-depth, compiled the report, saved it to `newsletter_report.md`, delegated script creation to the script generator (saved to `podcast_script.md`), and generated the podcast audio file for you."
    """,
    tools=[
        fetch_newsletters_from_inbox,
        fetch_article_content,
        google_search,
        get_financial_context,
        save_news_to_markdown,
        AgentTool(agent=script_generator_agent),
        AgentTool(agent=podcaster_agent)
    ],
    output_schema=NewsletterReport,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)