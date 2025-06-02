import os
import warnings
# Suppress urllib3 SSL warning
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

import requests
import json
import base64
import sys
import re
from urllib.parse import urlparse, unquote

from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat

import argparse

from rich.console import Console
from rich.markdown import Markdown

GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")

def fetch_file_from_github_api(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    path_parts = parsed.path.strip("/").split("/")

    try:
        owner = path_parts[0]
        repo = path_parts[1]
        commit = path_parts[3]
        filepath = "/".join(path_parts[4:])
        filepath = unquote(filepath)

        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{filepath}?ref={commit}"
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }

        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("encoding") == "base64":
                return base64.b64decode(data["content"]).decode("utf-8")
            else:
                return data["content"]
        else:
            print(f"Failed to fetch {filepath} from GitHub API: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Error while parsing URL or fetching file: {e}")
        return ""

def is_test_file(filename: str) -> bool:
    """
    Determines if a file is a test file based on its name or path.
    """
    # Convert to lowercase for case-insensitive matching
    filename_lower = filename.lower()
    
    # Check for test directories
    if filename_lower.startswith('test/') or filename_lower.startswith('tests/'):
        return True
        
    # Check for test file extensions
    if '.test.' in filename_lower or '.spec.' in filename_lower:
        return True
    
    # Check other patterns
    test_patterns = [
        '/test/', '/tests/', 'test_', '_test.', 
        'spec.', 'spec/', '_spec.', '/spec/', '__tests__'
    ]
    
    return any(pattern in filename_lower for pattern in test_patterns)

def get_github_commit(commit_url: str) -> tuple:
    """
    Fetches the commit details from GitHub and returns formatted text including file content.

    Args:
        commit_url (str): The URL of the GitHub commit 
                         (e.g., https://github.com/owner/repo/commit/hash).

    Returns:
        tuple: (formatted_text: str, commit_json: dict)
    """
    # Parse the URL to extract owner, repo, and commit hash
    parsed = urlparse(commit_url)
    path_parts = parsed.path.strip("/").split("/")
    
    if len(path_parts) < 4 or path_parts[2] != 'commit':
        raise ValueError("Invalid commit URL format")
    
    owner = path_parts[0]
    repo = path_parts[1]
    commit_sha = path_parts[3]

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Get commit details
    api_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"
    response = requests.get(api_url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to get commit details: {response.status_code} - {response.text}")
        return "Failed to fetch commit details", None

    commit_data = response.json()
    
    result_parts = []
    
    # Add commit metadata
    result_parts.append(f"COMMIT: {commit_data['sha']}")
    result_parts.append(f"AUTHOR: {commit_data['commit']['author']['name']} <{commit_data['commit']['author']['email']}>")
    result_parts.append(f"DATE: {commit_data['commit']['author']['date']}")
    result_parts.append(f"MESSAGE: {commit_data['commit']['message']}")
    result_parts.append("===================================\n")
    
    # Process files in the commit
    for file_info in commit_data.get('files', []):
        filename = file_info.get('filename', '')
        if is_test_file(filename):
            continue
            
        # Add file metadata
        file_part = f"FILE: {file_info['filename']}\n"
        file_part += f"STATUS: {file_info['status']}\n"
        file_part += f"ADDITIONS: {file_info.get('additions', 0)}, DELETIONS: {file_info.get('deletions', 0)}\n"
        
        # Add patch information
        if 'patch' in file_info:
            file_part += "PATCH:\n"
            patch = file_info['patch']
            if '\n' in patch:
                # Indent each line of the patch for better readability
                patch_lines = patch.split('\n')
                formatted_patch = '\n'.join(['    ' + line for line in patch_lines])
                file_part += formatted_patch + '\n'
            else:
                file_part += f"    {patch}\n"
        
        # Fetch and add the file content for modified files
        if file_info['status'] in ['modified', 'added'] and 'raw_url' in file_info:
            file_content = fetch_file_from_github_api(file_info['raw_url'])
            file_part += "SOURCE:\n"
            if file_content:
                if '\n' in file_content:
                    # Indent each line of the source for better readability
                    content_lines = file_content.split('\n')
                    formatted_content = '\n'.join(['    ' + line for line in content_lines])
                    file_part += formatted_content + '\n'
                else:
                    file_part += f"    {file_content}\n"
            else:
                file_part += "    <Failed to fetch file content>\n"
        
        file_part += "===================================\n"
        result_parts.append(file_part)
    
    formatted_text = "\n".join(result_parts) if result_parts else "No files found in commit"
    
    return formatted_text, commit_data

def parse_arguments():
    """Parse command line arguments."""

    epilog = """
    Environment variables:
      GITHUB_ACCESS_TOKEN - Your GitHub API token (required)
      
    Example usage:
      python commit.py https://github.com/owner/repo/commit/abc123
      python commit.py https://github.com/owner/repo/commit/abc123 --summary project_summary.txt
      python commit.py https://github.com/owner/repo/commit/abc123 --output report.md
    """
    
    parser = argparse.ArgumentParser(
        description="Argus commit security analyzer - analyzes a single Git commit for security vulnerabilities",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("commit_url", 
                        help="URL to the GitHub commit (e.g., https://github.com/owner/repo/commit/hash)")
    
    parser.add_argument("--summary",
                        required=False,
                        help="Path to file with project summary for additional context")

    parser.add_argument("--debug",
                        action="store_true",
                        help="Enable debug mode for LLM interactions")
    
    parser.add_argument("--output",
                        required=False,
                        help="Path to save the security report (if not specified, prints to console)")
    
    return parser.parse_args()

def extract_markdown_content(text):
    """
    Extracts the content from the first markdown code block in a multiline string.
    
    Args:
        text (str): A multiline string that may contain markdown code blocks.
        
    Returns:
        str: The content between the first ```markdown and ``` tags, or the original text if not found.
    """
    import re
    
    # Look for content between ```markdown and ```
    match = re.search(r'```markdown\s+(.*?)```', text, re.DOTALL)
    
    if match:
        # Return just the content between the tags, without the ```markdown and ``` markers
        return match.group(1).strip()
    else:
        return text

def main():
    args = parse_arguments()
    
    if not GITHUB_TOKEN:
        print("Error: GITHUB_ACCESS_TOKEN environment variable is required")
        return 1
    
    print(f"Fetching commit from GitHub: {args.commit_url}")
    
    try:
        # Get commit information
        commit_info, commit_json = get_github_commit(args.commit_url)
        
        if not commit_info or not commit_json:
            print("Error: Failed to fetch commit information")
            return 1
        
    except Exception as e:
        print(f"Error processing commit: {str(e)}")
        return 1
    
    # Read project summary if provided
    summary = ""
    if args.summary:
        try:
            with open(args.summary, 'r') as file:
                summary = file.read().strip()
        except Exception as e:
            print(f"Warning: Could not read summary file: {str(e)}")
    
    # Prepare system prompt
    system = "Your goal is to thoroughly review code changes for security vulnerabilities."
    if summary:
        system += f" Here is your brief technical architecture summary of the system you are working on: {summary}"

    # Initialize the security reviewer agent
    security_reviewer = Agent(
        name="commit security reviewer", 
        role="You are the best in the world application security engineer which is doing top notch security code audits. Be concise and professional",
        model=Gemini(id="gemini-2.5-pro-preview-03-25", system_prompt=system),
        markdown=True,
        instructions="""
        Analyze each file change carefully, looking for:
        - Injection vulnerabilities (SQL, command, etc.)
        - Authentication/authorization issues
        - Data validation problems
        - Insecure cryptography
        - Hardcoded secrets or credentials
        - Race conditions
        - Insecure dependencies
        - Other security concerns

        Also analyse each file change carefully for any functional or logical bugs.
        
        If you find NO major security concerns, explicitly state: "NO MAJOR SECURITY CONCERNS FOUND"
        Be thorough but concise in your analysis.
        Ignore all changes in tests.
        """,
        expected_output="""
        A professional properly formatted security audit report:

        # Security Audit Report - Commit Analysis

        ## Commit Information
        - **SHA**: {commit SHA}
        - **Author**: {author name and email}
        - **Date**: {commit date}
        - **Message**: {commit message}

        ## Summary of Changes
        {Overall summary of what this commit does}

        ## Files Modified
        {List of files changed with brief description of changes in each}

        ## Security Assessment
        {Overall security assessment of the commit}

        ## Security Findings
        {For each finding: file path, vulnerability class, full description, severity, potential impact, recommended fix, code example if relevant}

        ## Security Concerns
        {Any other security concerns without exact vulnerabilities if exists, with description, impact and recommendations}

        ## Conclusion
        {Overall conclusion about the security posture of this commit}

        {Optional string 'NO MAJOR SECURITY CONCERNS FOUND' if applicable}
        """,
        debug_mode=args.debug
    )

    # Prepare the review prompt
    review_prompt = f"""
    Please perform a thorough security review of the following Git commit.
    Focus on identifying any security vulnerabilities or concerns in the code changes.
    
    The commit URL is: {args.commit_url}
    
    Here is the commit information and code changes:
    
    {commit_info}
    
    Provide a well-formatted markdown report focusing on security aspects.
    At the very end add string 'NO MAJOR SECURITY CONCERNS FOUND' if nothing serious found.
    Provide only markdown report, nothing like "Ok, here is your report" at the beginning. Also no need for ```markdown wrapper.
    """
    
    # Get security analysis
    print("Performing security analysis...")
    security_analysis = security_reviewer.run(review_prompt).content
    
    # Extract markdown content
    report_content = extract_markdown_content(security_analysis)
    
    # Output the report
    if args.output:
        # Save to file
        try:
            with open(args.output, 'w') as f:
                f.write(report_content)
            print(f"\nSecurity report saved to: {args.output}")
        except Exception as e:
            print(f"Error saving report to file: {str(e)}")
            # Fall back to console output
            console = Console()
            md = Markdown(report_content)
            console.print(md)
    else:
        # Print to console
        console = Console()
        md = Markdown(report_content)
        console.print(md)
    
    print("\nDone")
    return 0

if __name__ == "__main__":
    sys.exit(main())