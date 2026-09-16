"""
Where the Support menu points.

Kept in their own file so changing a link never means touching UI code.
`is_placeholder` still guards every entry, so an unfilled one explains itself
instead of opening a dead page.
"""

BUY_ME_A_COFFEE = "https://buymeacoffee.com/btspodcastph"
YOUTUBE = "https://www.youtube.com/@marineearthscience"
# The share link carries a tracking parameter; the bare show URL is cleaner
# and works the same.
SPOTIFY = "https://open.spotify.com/show/4NTLrSfceKjpFvZWflzBJj"
FACEBOOK = "https://www.facebook.com/btspodcastph"

# Google Drive form used to collect problem reports.
REPORT_FORM = "https://forms.gle/G1YgvvmcjYtQwEhQ8"

PODCAST_NAME = "Behind The Science Podcast"

# Menu order. Buy Me a Coffee sits on its own above the rest because it is the
# one that is actually being asked for.
SUPPORT_MENU = [
    ("Buy me a coffee", BUY_ME_A_COFFEE),
    (None, None),                        # separator
    ("Watch on YouTube", YOUTUBE),
    ("Listen on Spotify", SPOTIFY),
    ("Facebook", FACEBOOK),
]


def is_placeholder(url):
    return not url or "example.com" in url
