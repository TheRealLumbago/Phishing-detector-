import re
from collections import deque

class PhishingAgent:
    def __init__(self):
        # High-risk phishing phrases (graph nodes)
        self.phishing_graph = {
            "start": ["verify", "suspended", "password", "urgent", "click"],
            "verify": ["verify your account", "confirm your password"],
            "suspended": ["account has been suspended"],
            "password": ["confirm your password"],
            "urgent": ["urgent action required"],
            "click": ["click here to verify"]
        }

        # Your risk keywords
        self.high_risk_keywords = [
            "verify your account", "your account has been suspended",
            "confirm your password", "update your billing",
            "click here to verify", "unauthorized login", "urgent action required"
        ]

        self.medium_keywords = [
            "verify", "suspended", "urgent", "limited", "password",
            "bank", "billing", "security alert"
        ]

        self.trusted_domains = [
            r"google\.com", r"amazon\.com", r"microsoft\.com", r"apple\.com"
        ]

    #BFS SEARCH FUNCTION
    def bfs_search(self, email_text):
        email_text = email_text.lower()

        queue = deque([("start", ["start"])])
        visited = set()

        while queue:
            node, path = queue.popleft()

            # if a phishing keyword in path appears in email text → FOUND
            if node != "start" and node in email_text:
                return True, path

            visited.add(node)

            for next_node in self.phishing_graph.get(node, []):
                if next_node not in visited:
                    queue.append((next_node, path + [next_node]))

        return False, []


    # Your Feature Extractor 
    def extract_features(self, text):
        text = text.lower()

        # Detect links
        links = re.findall(r"http[s]?://[^\s]+", text)
        num_links = len(links)

        # Count keywords
        high_risk_count = sum(1 for kw in self.high_risk_keywords if kw in text)
        medium_count = sum(1 for kw in self.medium_keywords if kw in text)

        has_html = int(bool(re.findall(r"<[^>]+>", text)))

        trusted_link_found = any(
            re.search(domain, link) for link in links for domain in self.trusted_domains
        )

        return {
            "num_links": num_links,
            "high_risk": high_risk_count,
            "medium_risk": medium_count,














            
            "has_html": has_html,
            "trusted_link": trusted_link_found
        }


    def classify(self, text):
        features = self.extract_features(text)
        bfs_found, path = self.bfs_search(text)

        score = 0

        if bfs_found:
            score += 5

        score += features["high_risk"] * 5
        score += features["medium_risk"] * 2
        score += (2 if features["num_links"] > 0 else 0)
        score -= (2 if features["trusted_link"] else 0)
        score += features["has_html"]

        return "Phishing Email" if score >= 6 else "Safe Email"

# Example usage
agent = PhishingAgent()

phishing_email = """
Your account has been suspended. Please click here to verify your identity:
https://Phising-site.com
"""

safe_email = """
Hello team,
Just a reminder about the meeting scheduled for tomorrow at 10 AM.
"""

print(safe_email, agent.classify(safe_email))
print(phishing_email, agent.classify(phishing_email))