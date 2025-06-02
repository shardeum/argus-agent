# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- `python agent.py --pull-request <PR_URL>` - Review a specific GitHub pull request
- `python agent.py --slack-channel <CHANNEL_ID>` - Monitor Slack channel for review requests
- `python agent.py --linear-ticket <TICKET_ID>` - Review Linear ticket and associated PRs
- `python agent.py --summary <FILE_PATH>` - Run review with project summary context
- `python agent.py --debug` - Enable debug mode for LLM interactions
- `python agent.py --open-vs-code` - Clone repo and open in VS Code after review

### Testing
- No automated tests currently exist in the codebase

### Dependencies
- Install dependencies: `pip install -r requirements.txt`
- Key dependencies: agno (LLM framework), slack-sdk, rich (UI), google-genai, openai

### Required Environment Variables
- `GITHUB_ACCESS_TOKEN` - GitHub API access token
- `SLACK_BOT_TOKEN` - Slack bot token (for channel monitoring)
- `LINEAR_API_KEY` - Linear API key (optional, for ticket integration)

## Architecture Overview

Argus is a Python-based security code reviewer that uses LLMs to analyze pull requests and code changes for potential security vulnerabilities.

### Core Components

**agent.py** (862 lines) - Main application containing:
- `ArgusAgent` class: Core agent implementation using the agno framework
- `SlackFetcher` class: Handles Slack channel monitoring and message parsing
- `LinearFetcher` class: Integrates with Linear API to fetch tickets and PRs
- Security prompt templates for comprehensive vulnerability analysis

### Key Functionality

1. **Multiple Input Sources**:
   - Direct GitHub PR URLs
   - Slack channel monitoring for review requests
   - Linear ticket integration
   - Project summary files for context

2. **Security Analysis Focus**:
   - SQL/NoSQL injection vulnerabilities
   - Command injection and code execution risks
   - Authentication and authorization flaws
   - Data validation and sanitization issues
   - Hardcoded secrets and credentials
   - Race conditions and concurrency issues
   - Insecure dependencies and configurations
   - Error handling and information disclosure

3. **Report Generation**:
   - Severity levels (Critical/High/Medium/Low)
   - Detailed explanations with file locations
   - Remediation suggestions
   - Summary statistics

### Workflow

1. **Input Processing**: Accepts PR URLs, Slack messages, or Linear tickets
2. **Code Fetching**: Retrieves PR diff and file contents via GitHub API
3. **Context Building**: Optionally includes project summaries for better understanding
4. **LLM Analysis**: Sends code and context to Gemini/OpenAI for security review
5. **Report Formatting**: Structures findings with severity, descriptions, and fixes
6. **Optional Actions**: Can clone repository and open in VS Code

### Key Design Patterns

- Uses agno framework for LLM agent abstraction
- Extensive prompt engineering for security-focused analysis
- Rich terminal output for formatted reports
- Modular fetchers for different input sources
- Environment-based configuration for API keys