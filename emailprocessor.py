# Code for email handling and manipulations
import email
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
import quopri
import base64
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class EmailProcessor:
    """
    Comprehensive email processing class for spam filtering
    Handles header removal, multipart emails, and text extraction
    """

    def __init__(self):
        # Common email headers to remove/filter
        self.standard_headers = [
            'return-path', 'delivered-to', 'received', 'from', 'to', 'cc', 'bcc',
            'subject', 'date', 'message-id', 'reply-to', 'in-reply-to', 'references',
            'mime-version', 'content-type', 'content-transfer-encoding',
            'x-mailer', 'x-priority', 'x-spam-flag', 'x-spam-level', 'x-spam-status',
            'precedence', 'list-id', 'list-unsubscribe', 'user-agent',
            'thread-topic', 'thread-index', 'organization', 'sender'
        ]

        # Patterns for email forwarding and reply chains
        self.forwarding_patterns = [
            r'-----Original Message-----',
            r'From:.*?(?=\n\S|\n\n|\Z)',
            r'Sent:.*?(?=\n\S|\n\n|\Z)',
            r'To:.*?(?=\n\S|\n\n|\Z)',
            r'Subject:.*?(?=\n\S|\n\n|\Z)',
            r'Forwarded-by:.*?(?=\n\S|\n\n|\Z)',
            r'Begin forwarded message:',
            r'---------- Forwarded message ----------',
            r'On .* wrote:',
            r'From fork-admin@.*?(?=\n\S|\n\n|\Z)',
            r'Return-Path:.*?(?=\n\S|\n\n|\Z)',
            r'Delivered-To:.*?(?=\n\S|\n\n|\Z)',
            r'Received:.*?(?=\n\S|\n\n|\Z)'
        ]

    def parse_email_from_file(self, file_path: str) -> email.message.Message:
        """Parse email from file with proper encoding handling"""
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return email.message_from_file(f)
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['utf-8', 'ascii', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return email.message_from_file(f)
                except UnicodeDecodeError:
                    continue
            # If all fail, use binary mode
            with open(file_path, 'rb') as f:
                return email.message_from_bytes(f.read())

    def remove_standard_headers(self, email_text: str) -> str:
        """Remove standard email headers from raw email text"""
        lines = email_text.split('\n')
        cleaned_lines = []
        in_header = True

        for line in lines:
            if in_header:
                # Check if this line is a header
                if ':' in line and not line.startswith(' ') and not line.startswith('\t'):
                    header_name = line.split(':', 1)[0].lower().strip()
                    if header_name in self.standard_headers:
                        continue  # Skip this header
                    else:
                        # This might be content, switch to body mode
                        in_header = False
                        cleaned_lines.append(line)
                elif line.strip() == '':
                    # Empty line typically separates headers from body
                    in_header = False
                elif line.startswith(' ') or line.startswith('\t'):
                    # Continuation of previous header, skip if we're still in header mode
                    continue
                else:
                    # This is likely content
                    in_header = False
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def remove_forwarding_headers(self, text: str) -> str:
        """Remove forwarding and reply chain headers"""
        cleaned_text = text

        for pattern in self.forwarding_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)

        # Remove multiple consecutive newlines
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)

        return cleaned_text.strip()

    def decode_content(self, payload: str, encoding: str) -> str:
        """Decode email content based on encoding"""
        if encoding is None:
            return payload

        encoding = encoding.lower()

        try:
            if encoding == 'quoted-printable':
                return quopri.decodestring(payload.encode()).decode('utf-8', errors='ignore')
            elif encoding == 'base64':
                return base64.b64decode(payload).decode('utf-8', errors='ignore')
            elif encoding in ['7bit', '8bit', 'binary']:
                return payload
            else:
                return payload
        except Exception:
            return payload

    def extract_text_from_simple_email(self, msg: email.message.Message) -> str:
        """Extract text from simple (non-multipart) email"""
        payload = msg.get_payload()

        if isinstance(payload, str):
            encoding = msg.get('Content-Transfer-Encoding')
            return self.decode_content(payload, encoding)
        else:
            return str(payload)

    def extract_text_from_multipart(self, msg: email.message.Message) -> Dict[str, str]:
        """
        Extract text from multipart email
        Returns dictionary with different content types
        """
        text_parts = {
            'plain': [],
            'html': [],
            'other': []
        }

        for part in msg.walk():
            # Skip the multipart container itself
            if part.get_content_maintype() == 'multipart':
                continue

            content_type = part.get_content_type()

            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    payload = part.get_payload()

                if isinstance(payload, bytes):
                    # Try to decode bytes to string
                    charset = part.get_content_charset() or 'utf-8'
                    try:
                        text_content = payload.decode(charset, errors='ignore')
                    except (UnicodeDecodeError, LookupError):
                        text_content = payload.decode('latin-1', errors='ignore')
                else:
                    text_content = str(payload)

                # Categorize content
                if content_type == 'text/plain':
                    text_parts['plain'].append(text_content)
                elif content_type == 'text/html':
                    text_parts['html'].append(text_content)
                else:
                    text_parts['other'].append(f"[{content_type}] {text_content[:100]}...")

            except Exception as e:
                print(f"Error processing part {content_type}: {e}")
                continue

        return text_parts

    def clean_html_content(self, html_text: str) -> str:
      """HTML tag removal using BeautifulSoup"""
      try:
          # Parse HTML with BeautifulSoup
          soup = BeautifulSoup(html_text, 'html.parser')

          # Remove script and style elements completely
          for script in soup(["script", "style"]):
              script.decompose()

          # Get text content
          text = soup.get_text()

          # Clean up whitespace
          lines = (line.strip() for line in text.splitlines())
          chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
          text = ' '.join(chunk for chunk in chunks if chunk)

          return text

      except Exception as e:
          # Fallback to your original method if BeautifulSoup fails
          print(f"BeautifulSoup parsing failed, using fallback: {e}")
          return self.clean_html_content_backup(html_text)

    def clean_html_content_backup(self, html_text: str) -> str:
      """backup process in case something happens to beautiful soup"""
      # Remove HTML tags
      clean_text = re.sub(r'<[^>]+>', '', html_text)

      # Decode HTML entities
      html_entities = {
          '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
          '&apos;': "'", '&nbsp;': ' ', '&copy;': '©', '&reg;': '®'
      }

      for entity, char in html_entities.items():
          clean_text = clean_text.replace(entity, char)

    def process_email(self, msg: email.message.Message, prefer_html: bool = False) -> str:
        """
        Main method to process an email and extract clean text

        Args:
            msg: Email message object
            prefer_html: Whether to prefer HTML content over plain text

        Returns:
            Cleaned text content
        """
        if msg.is_multipart():
            # Handle multipart email
            text_parts = self.extract_text_from_multipart(msg)

            # Choose which content to use
            if prefer_html and text_parts['html']:
                content = '\n'.join(text_parts['html'])
                content = self.clean_html_content(content)
            elif text_parts['plain']:
                content = '\n'.join(text_parts['plain'])
            elif text_parts['html']:
                content = '\n'.join(text_parts['html'])
                content = self.clean_html_content(content)
            else:
                content = '\n'.join(text_parts['other'])
        else:
            # Handle simple email
            content = self.extract_text_from_simple_email(msg)

            # If it's HTML, clean it
            if msg.get_content_type() == 'text/html':
                content = self.clean_html_content(content)

        # Remove forwarding headers and clean up
        content = self.remove_forwarding_headers(content)

        # Additional cleanup
        content = self.final_cleanup(content)

        return content

    def final_cleanup(self, text: str) -> str:
        """Final text cleanup operations"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common email signatures
        text = re.sub(r'\n--\s*\n.*', '', text, flags=re.DOTALL)

        # Remove URLs (optional - might want to keep for spam detection)
        # text = re.sub(r'https?://\S+', '[URL]', text)

        # Remove email addresses (optional)
        # text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

        return text.strip()

    def analyze_email_structure(self, msg: email.message.Message) -> Dict:
        """Analyze email structure for debugging"""
        analysis = {
            'is_multipart': msg.is_multipart(),
            'content_type': msg.get_content_type(),
            'headers': dict(msg.items()),
            'parts': []
        }

        if msg.is_multipart():
            for i, part in enumerate(msg.walk()):
                if part.get_content_maintype() == 'multipart':
                    continue

                part_info = {
                    'part_number': i,
                    'content_type': part.get_content_type(),
                    'content_disposition': part.get('Content-Disposition'),
                    'content_transfer_encoding': part.get('Content-Transfer-Encoding'),
                    'charset': part.get_content_charset(),
                    'size': len(str(part.get_payload()))
                }
                analysis['parts'].append(part_info)

        return analysis

# Example usage and testing
def demonstrate_email_processing():
    """Demonstrate email processing with examples"""

    processor = EmailProcessor()

    # Example 1: Simple email with headers
    print("=== EXAMPLE 1: Simple Email with Headers ===")

    simple_email_text = """From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Wed, 24 Jul 2002 03:23:20 -0700
Message-ID: <12345@example.com>
Content-Type: text/plain

This is the actual email content.
It should be preserved after header removal.
"""

    cleaned_simple = processor.remove_standard_headers(simple_email_text)
    print("Original:")
    print(simple_email_text)
    print("\nCleaned:")
    print(cleaned_simple)

    # Example 2: Email with forwarding headers
    print("\n=== EXAMPLE 2: Email with Forwarding Headers ===")

    forwarded_email = """-----Original Message-----
From: razor-use@listserver.com
Sent: Wednesday, July 24, 2002 3:23 AM
To: recipient@example.com
Subject: FW: Important Message

Forwarded-by: Rob Windsor <windsor@warthog.co>

From fork-admin@xent.com Wed Jul 24 03:23:20 2002
Return-Path: <fork-admin@xent.com>
Delivered-To: yyyy@localhost.netnoteinc.com
Received: from localhost (localhost [127.0.0.1])

This is the actual message content that we want to keep.
Everything above should be filtered out.
"""

    cleaned_forwarded = processor.remove_forwarding_headers(forwarded_email)
    print("Original:")
    print(forwarded_email)
    print("\nCleaned:")
    print(cleaned_forwarded)

    # Example 3: Create and process a multipart email
    print("\n=== EXAMPLE 3: Multipart Email ===")

    # Create a sample multipart email
    multipart_msg = MIMEMultipart('alternative')
    multipart_msg['Subject'] = 'Test Multipart Email'
    multipart_msg['From'] = 'sender@example.com'
    multipart_msg['To'] = 'recipient@example.com'

    # Plain text part
    plain_text = """This is the plain text version of the email.
It contains the main message content.
Some important information here."""

    # HTML part
    html_text = """<html>
<body>
<h1>This is the HTML version</h1>
<p>This is the <b>HTML</b> version of the email.</p>
<p>It contains <em>formatted</em> content.</p>
<p>Some <strong>important</strong> information here.</p>
</body>
</html>"""

    # Add parts
    plain_part = MIMEText(plain_text, 'plain')
    html_part = MIMEText(html_text, 'html')
    multipart_msg.attach(plain_part)
    multipart_msg.attach(html_part)

    # Process the multipart email
    analysis = processor.analyze_email_structure(multipart_msg)
    print("Email Structure Analysis:")
    print(f"Is multipart: {analysis['is_multipart']}")
    print(f"Content type: {analysis['content_type']}")
    print(f"Number of parts: {len(analysis['parts'])}")

    for part in analysis['parts']:
        print(f"  Part {part['part_number']}: {part['content_type']}")

    # Extract text content
    extracted_text = processor.process_email(multipart_msg, prefer_html=False)
    print(f"\nExtracted text (prefer plain):")
    print(extracted_text)

    extracted_html = processor.process_email(multipart_msg, prefer_html=True)
    print(f"\nExtracted text (prefer HTML, cleaned):")
    print(extracted_html)

def process_real_email_file(file_path: str):
    """Process a real email file"""
    processor = EmailProcessor()

    try:
        # Parse email
        msg = processor.parse_email_from_file(file_path)

        # Analyze structure
        analysis = processor.analyze_email_structure(msg)
        print(f"Email Analysis for {file_path}:")
        print(f"Is multipart: {analysis['is_multipart']}")
        print(f"Content type: {analysis['content_type']}")

        if analysis['is_multipart']:
            print("Parts:")
            for part in analysis['parts']:
                print(f"  {part['content_type']} (size: {part['size']})")

        # Extract clean text
        clean_text = processor.process_email(msg)
        print(f"\nCleaned text (first 500 chars):")
        print(clean_text[:500])

        return clean_text

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Integration with spam filter
def integrate_with_spam_filter():
    """Show how to integrate email processing with spam filtering"""

    print("\n=== INTEGRATION WITH SPAM FILTER ===")

    processor = EmailProcessor()

    # Example of processing multiple emails for spam filtering
    def process_email_dataset(email_files: List[str]) -> List[Dict]:
        """Process multiple emails and prepare for spam filtering"""

        processed_emails = []

        for file_path in email_files:
            try:
                # Parse email
                msg = processor.parse_email_from_file(file_path)

                # Extract clean text
                clean_text = processor.process_email(msg)

                # Determine label from file path (for SpamAssassin dataset)
                label = 1 if 'spam' in file_path else 0

                processed_emails.append({
                    'file_path': file_path,
                    'text': clean_text,
                    'label': label,
                    'is_multipart': msg.is_multipart(),
                    'content_type': msg.get_content_type()
                })

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        return processed_emails

    print("Use process_email_dataset() to prepare emails for spam filtering")
    print("The cleaned text will be much better for machine learning models!")

if __name__ == "__main__":
    # Run demonstrations
    demonstrate_email_processing()
    integrate_with_spam_filter()

    print("\n=== SUMMARY ===")
    print("Email processing handles:")
    print("1. Standard header removal (From, To, Subject, etc.)")
    print("2. Forwarding header cleanup (Original Message blocks)")
    print("3. Multipart email parsing (plain text + HTML)")
    print("4. Content decoding (base64, quoted-printable)")
    print("5. HTML tag removal and entity decoding")
    print("6. Final text cleanup and normalization")
