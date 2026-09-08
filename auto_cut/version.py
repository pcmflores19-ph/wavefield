"""Single source of the version number - the app, the About box and the
installer all read it from here so they cannot drift apart."""

__version__ = "1.1.1"

APP_NAME = "Wavefield"
PROJECT_URL = "https://github.com/pcmflores19-ph/wavefield"

# Wavefield bundles pedalboard and rnnoise, both GPL, so the built application
# is GPL-3. That obliges us to make the source available to anyone who receives
# the program. The repository is public, so this is satisfied the easy way -
# GPL-3 section 6(d), a public network location - and everyone can simply go
# and read it. The written offer under section 6(b) that used to stand here was
# only ever needed because the repository was private.
SOURCE_URL = PROJECT_URL

# Kept for anyone who cannot use GitHub and would rather ask a person.
SOURCE_CONTACT_URL = "https://www.facebook.com/btspodcastph"
