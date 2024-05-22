import streamlit as st
import imaplib
import email
import datetime
import pytz
from email.utils import parsedate_to_datetime
import email.header 
from groq import Groq
import os
import itertools

import tiktoken
import re  # For regular expressions

from bs4 import BeautifulSoup  # Import BeautifulSoup
from email.header import decode_header

# Groq API credentials
os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Helper function to decode email headers and content
def decode_email_part(part):
    """Efficiently decodes email parts, including encoded-word headers."""
    # Check if 'part' is already a string, avoiding the error
    if isinstance(part, str):
        return part  # Already decoded
    
    payload = part.get_payload(decode=True)
    charset = part.get_content_charset() or "utf-8"  

    if payload:
        try:
            decoded_header, encoding = email.header.decode_header(payload)[0]
            if isinstance(decoded_header, bytes):
                return decoded_header.decode(encoding or charset)
            return decoded_header 
        except UnicodeDecodeError:
            return payload.decode("utf-8", errors="replace")  
    return ""

def extract_plain_text(email_message):
    if not isinstance(email_message, str):  # Ensure it's a string
        email_message = str(email_message)  # Convert to string if needed
    
    soup = BeautifulSoup(email_message, "html.parser")
    summary_elements = soup.select("div.article-summary p")

    text_parts = [element.get_text(separator=" ", strip=True) for element in summary_elements]
    return " ".join(text_parts)


def strip_email_content(content):
    # Decode if bytes, handling potential errors
    content = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content

    # Remove headers (up to the first empty line)
    content = re.sub(r"(.*?\n\n)", "", content, count=1) 
    # Remove quoted text
    content = re.sub(r"(On.*?wrote:.*?\n)", "", content, flags=re.DOTALL)
    # Remove signatures (simple example)
    content = re.sub(r"(--\n.*)", "", content, flags=re.DOTALL)
    return content

# Helper functions for LLM
call_count = 0  # Initialize a global call counter

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    # Explicitly specify the cl100k_base tokenizer
    encoding = tiktoken.get_encoding("cl100k_base")  
    num_tokens = len(encoding.encode(string))
    return num_tokens

def summarize_email(email_content, model_selected, max_tokens=32000):
    global call_count  # Access the global counter
    call_count += 1   # Increment the counter each time the function is called

    total_tokens = num_tokens_from_string(email_content, "cl100k_base")

    try:
        # If the content is too long, split it into chunks
        if total_tokens > max_tokens:
            chunks = []
            current_chunk = ""
            current_chunk_tokens = 0
            for word in email_content.split():
                word_tokens = num_tokens_from_string(word, "cl100k_base")
                if current_chunk_tokens + word_tokens <= max_tokens:
                    current_chunk += word + " "
                    current_chunk_tokens += word_tokens
                else:
                    chunks.append(current_chunk)
                    current_chunk = word + " "
                    current_chunk_tokens = word_tokens
            if current_chunk:  # Add the last chunk if it's not empty
                chunks.append(current_chunk)

            summaries = []
            for chunk in chunks:
                # Extract plain text from the chunk
                email_message = email.message_from_string(chunk)
                plain_text_chunk = extract_plain_text(email_message)
                
                # Modified Prompt
                prompt = f"""
                        Summarize this part of the email exchange:
                        "{plain_text_chunk}"
                    """

                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_selected,
                    temperature=0,
                    max_tokens=25000,
                    top_p=1
                )
                summaries.append(response.choices[0].message.content)

            # Combine summaries
            combined_summary = "\n\n".join(summaries)
            return combined_summary

        # If the content is within the limit, just summarize it as before
        else:
            prompt = f"""
            Summarize all the email exchange so I can clearly understand the email thread. \
            Please make your summary a combination of all emails put together that explain and provide the full picture and tell the story of the emails exchange.\
            In the email content you may find duplicate of the same email since you may have emails that are reply that include previous emails.\
            If you find groups of email that are about a specific topic, regroup them by theme in your summary and list all the themes you encounter in the email thread extracted.\
            Output your summary of the emails content within three backtick, i.e. "```" \
            
            Here's the content you need to summarize:\
            "{email_content}"
            """
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_selected,
                temperature=0,
                max_tokens=25000,
                top_p=1
            )
            return response.choices[0].message.content
        
    finally:  # Always decrement, even if there's an error during summarization
        call_count -= 1


def article_key_takeaways(article_content, model_selected):
    prompt = f"""
    Summarize and extract the main key takeawyas of this text. Output the key takwaways within three backtick, i.e. "```" \
    
    Here's the content of the text you need to analyze:\
    "{article_content}"
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_selected,
        temperature=0,
        max_tokens=3000,
        top_p=1
    )
    return response.choices[0].message.content
def stock_market_sentiment(article_content, model_selected):
    prompt = f"""
    Based on the content of this news article, how would you describe the potential impact on stock market sentiment? \
    
    Here's the content of the text you need to analyze:\
    "{article_content}"
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_selected,
        temperature=0,
        max_tokens=100,
        top_p=1
    )
    return response.choices[0].message.content




# Create a Streamlit app
st.title("Gmail Email Extractor & Summarizer")
st.markdown("Extract, filter, and summarize emails from your Gmail.")


# Get the user's Gmail address and password
#email_address = st.text_input("Enter your Gmail address", value = "sattvamtl@gmail.com")
#password = st.text_input("Enter your Gmail password", type="password", value = "jqkr ooko qfjz pixv")

email_address = st.text_input("Enter your Gmail address", value = "smartineau00@gmail.com")
password = st.text_input("Enter your Gmail password", type="password", value = "mvpu hbpu owuu yjhl")

# Calculate initial dates
today = datetime.date.today() + datetime.timedelta(days=1)
one_week_ago = today - datetime.timedelta(days=110)


# Get the user's filters
subject = st.text_input("Filter by subject")
sender = st.text_input("Filter by sender", value = "happykaty@hotmail.com")
start_date = st.date_input("Start date", value=one_week_ago)
end_date = st.date_input("End date", value=today)
#keywords = st.text_input("Filter by keywords (separated by ';')")

# Initiali#ze the IMAP connection
mail = imaplib.IMAP4_SSL('imap.gmail.com')

# Initialize dictionary to store email content and received datetime
email_data = {}

# Get the user's input
if st.button("Extract & Summarize"):
    with st.spinner("Connecting to Gmail and fetching emails..."):
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        mail.login(email_address, password)
        mail.select('inbox')

        # Dynamic Search Criteria (Refined)
        search_criteria = [
            f'(SUBJECT "{subject}")' if subject else None,
            f'(FROM "{sender}")' if sender else None,
            f'(SINCE {start_date.strftime("%d-%b-%Y")})' if start_date else None,
            f'(BEFORE {end_date.strftime("%d-%b-%Y")})' if end_date else None,
        ]

        # Keyword Handling (Fixed)
        #if keywords:
        #    keyword_criteria = [f'(BODY "{kw.lower()}")' for kw in keywords.split(';') if kw.strip()]  # Clean keywords
        #    search_criteria.extend(keyword_criteria)

        search_query = " ".join(filter(None, search_criteria)) or "ALL"  # Safely combine criteria
        _, response = mail.search(None, search_query)  # Perform search with adjusted query
        email_ids = response[0].split()
        local_timezone = pytz.timezone('America/New_York')

    # Efficient Email Processing
    filtered_email_data = {}
    for email_id in email_ids:
        _, response = mail.fetch(email_id, '(RFC822)')
        email_message = email.message_from_bytes(response[0][1])

        # Decode Email Subject for filtering
        decoded_subject = decode_header(email_message['Subject'])[0][0]
        subject = decoded_subject.decode() if isinstance(decoded_subject, bytes) else decoded_subject

        # Filtering (integrated into email processing) - use decoded_subject
        #(keywords and not any(kw.lower() in decode_email_part(email_message).lower() for kw in keywords.split(';'))) or \
        if (subject and subject.lower() not in decoded_subject.lower()) or \
        (sender and sender.lower() not in email_message["From"].lower()) or \
        (start_date and parsedate_to_datetime(email_message['Date']).astimezone(local_timezone).date() < start_date) or \
        (end_date and parsedate_to_datetime(email_message['Date']).astimezone(local_timezone).date() > end_date):
            continue

        # Extract Plain Text from Email Parts (Improved)
        plain_text_parts = []
        for part in email_message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    plain_text_parts.append(payload.decode("utf-8", errors="replace"))

        # Combine Plain Text Parts
        plain_text = "\n\n".join(plain_text_parts)
        
        # Apply Stripping to Further Reduce Unnecessary Content
        plain_text = strip_email_content(plain_text)

        # Store Cleaned Plain Text
        filtered_email_data[email_id] = {
            "content": plain_text,  # Store only the extracted plain text
            "received_datetime": parsedate_to_datetime(email_message['Date'])
        }

        # Decode Subject and From
        subject = decode_email_part(email_message['Subject'])
        sender = decode_email_part(email_message['From'])
        
        # Get received date with timezone adjustment and formatting
        received_date = parsedate_to_datetime(email_message['Date']).astimezone(local_timezone)
        formatted_date = received_date.strftime("%Y-%m-%d %H:%M:%S %Z") # format date here

        # Display email information in Streamlit
        st.write(f"Subject: {subject}")  # Use the decoded subject
        st.write(f"From: {sender}")  # Use the decoded sender
        st.write(f"Received: {formatted_date}")

        # Extract and display plain text preview
        preview_text = ""
        for part in email_message.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    preview_text = payload.decode("utf-8", errors="replace")
                break  # Stop after finding the first text part
        st.write(f"Preview: {preview_text[:1000]}...")  

        st.write("---")

    # Summarize Filtered Emails
    if filtered_email_data:  # Check if any emails were found
        with st.spinner("Summarizing emails..."):
            email_content_list = [email["content"] for email in filtered_email_data.values()]  
            email_content_str = "\n\n".join(email_content_list)  # Join emails with newlines
            
            LLM_Response = summarize_email(email_content_str, "llama3-70b-8192")

            # Improved Summary Display with Text Wrapping
            st.write("---")
            st.write("\n")
            st.subheader("Email Thread Summary:")

            display_output = st.text_area("\n", value=LLM_Response, height=500)
    else:
        st.write("No emails found matching the criteria.")
    
    # After the button is clicked and email processing is done
    #st.write(f"The 'client.chat.completions.create' function was called {call_count} times.")

    # Close the IMAP connection
    mail.close()
    mail.logout()
