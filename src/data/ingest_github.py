# src/data/ingest_github.py

import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


BASE_URL = "https://api.github.com"
OWNER = "facebook"
REPO = "react"
DEFAULT_MAX_ISSUES = 2000
OUTPUT_PATH = "data/raw/react_issues.json"


def load_token() -> str:
    """
    Load GitHub token from environment (.env) and return it.
    Exits the program if the token is missing.
    """
    load_dotenv()  # loads variables from .env into os.environ
    token = os.environ.get("GITHUB_TOKEN")

    if not token:
        print("ERROR: GITHUB_TOKEN not found in environment. "
              "Set it in a .env file or your shell.")
        sys.exit(1)

    return token


def build_headers(token: str) -> Dict[str, str]:
    """
    Build HTTP headers for GitHub REST API requests.
    """
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def normalize_issue(raw_issue: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a raw GitHub issue JSON object into a flat dict
    with only the fields we care about.

    Returns None for pull requests so the caller can skip them.
    According to the GitHub REST API, issues with a 'pull_request'
    field are PRs. [web:47][web:104]
    """
    # Skip pull requests
    if "pull_request" in raw_issue:
        return None

    labels = [lab.get("name", "") for lab in raw_issue.get("labels", [])]

    user = raw_issue.get("user") or {}
    author_login = user.get("login")

    normalized = {
        "id": raw_issue.get("number"),
        "title": raw_issue.get("title") or "",
        "body": raw_issue.get("body") or "",
        "state": raw_issue.get("state"),
        "created_at": raw_issue.get("created_at"),
        "updated_at": raw_issue.get("updated_at"),
        "closed_at": raw_issue.get("closed_at"),
        "labels": labels,
        "author": author_login,
        "comments_count": raw_issue.get("comments"),
        "url": raw_issue.get("html_url"),
    }

    return normalized


def fetch_issues(
    owner: str,
    repo: str,
    headers: Dict[str, str],
    max_issues: int = DEFAULT_MAX_ISSUES,
) -> List[Dict[str, Any]]:
    """
    Fetch up to `max_issues` issues from the given repository using
    the GitHub REST API, handling pagination.

    Uses /repos/{owner}/{repo}/issues with state=all, per_page=100,
    and page=X. [web:92][web:103]
    """
    issues: List[Dict[str, Any]] = []
    page = 1

    url = f"{BASE_URL}/repos/{owner}/{repo}/issues"

    print(f"Fetching issues from {owner}/{repo} ...")

    while len(issues) < max_issues:
        params = {
            "state": "all",      # open + closed [web:47]
            "per_page": 100,     # max page size [web:92]
            "page": page,
        }

        response = requests.get(url, headers=headers, params=params, timeout=30)

        if response.status_code == 403:
            # Check for rate limiting [web:96][web:108]
            remaining = response.headers.get("X-RateLimit-Remaining")
            reset_time = response.headers.get("X-RateLimit-Reset")
            print("ERROR: Received 403 Forbidden (possibly rate limited).")
            print(f"X-RateLimit-Remaining: {remaining}")
            print(f"X-RateLimit-Reset (epoch): {reset_time}")
            break

        if response.status_code != 200:
            print(f"ERROR: GitHub API returned status {response.status_code}")
            print(f"Response: {response.text[:500]}")
            break

        page_items = response.json()
        if not isinstance(page_items, list):
            print("ERROR: Unexpected response format (not a list).")
            break

        if not page_items:
            # No more pages [web:103][web:106]
            print("No more issues to fetch (empty page).")
            break

        print(f"Fetched page {page}, {len(page_items)} items.")

        for raw_issue in page_items:
            normalized = normalize_issue(raw_issue)
            if normalized is None:
                # Skip pull requests
                continue

            issues.append(normalized)

            if len(issues) >= max_issues:
                break

        if len(page_items) < 100:
            # Likely no more pages
            print("Last page had fewer than 100 items, stopping pagination.")
            break

        page += 1

    print(f"Total normalized issues collected: {len(issues)}")
    return issues


def save_issues(issues: List[Dict[str, Any]], path: str) -> None:
    """
    Save the list of issues to a JSON file at `path`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(issues)} issues to {path}")


def main() -> None:
    token = load_token()
    headers = build_headers(token)
    issues = fetch_issues(OWNER, REPO, headers, max_issues=DEFAULT_MAX_ISSUES)
    save_issues(issues, OUTPUT_PATH)


if __name__ == "__main__":
    main()
