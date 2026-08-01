import base64
import requests
import streamlit as st

def commit_file(filepath: str, content: str, message: str):
    token = st.secrets["GITHUB_TOKEN"]
    repo = st.secrets["GITHUB_REPO"]  # e.g. "yourname/contradiction-detective"
    api_url = f"https://api.github.com/repos/{repo}/contents/{filepath}"
    headers = {"Authorization": f"token {token}"}

    resp = requests.get(api_url, headers=headers)
    sha = resp.json().get("sha") if resp.status_code == 200 else None

    payload = {"message": message, "content": base64.b64encode(content.encode()).decode()}
    if sha:
        payload["sha"] = sha

    requests.put(api_url, headers=headers, json=payload)