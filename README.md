# Argus - helps you to run security code review with LLM

```
usage: agent.py [-h] [--pull-request PULL_REQUESTS] [--slack-channel SLACK_CHANNEL] [--summary SUMMARY] [--linear-ticket LINEAR_TICKET] [--debug]
                [--open-vs-code]

Argus llm code reviewer

options:
  -h, --help            show this help message and exit
  --pull-request PULL_REQUESTS
                        URL to the GitHub pull request (can be specified multiple times)
  --slack-channel SLACK_CHANNEL
                        Slack channel id to fetch review requests from
  --summary SUMMARY     Path to file with project summary
  --linear-ticket LINEAR_TICKET
                        Linear ticket to review
  --debug               Enable debug mode for llm
  --open-vs-code        Clone repo, checkout branch and open vscode

    Environment variables:
      GITHUB_ACCESS_TOKEN - Your GitHub API token
      SLACK_BOT_TOKEN - Your Slack API token
      LINEAR_API_KEY - Linear API token (optional)
      
    Note: Token environment variables are required for API access.
```
