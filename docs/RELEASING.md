# Releasing

How to put a new version on GitHub. The first release is the long one; after
that it is four commands and a form.

## Before you start

**Check the username in `auto_cut/version.py`.**

```python
PROJECT_URL = "https://github.com/pcmflores19-ph/wavefield"
```

This is not decorative. It drives *Help ▸ Project page* and, more importantly,
*Help ▸ Check for updates*, which turns it into a GitHub API call. If your
GitHub username is not `pcmflores19-ph`, change it here first or the update check
will silently never find anything.

## One-time setup

### 1. Create the repository

On [github.com/new](https://github.com/new):

| Field | Value |
|---|---|
| Repository name | `wavefield` |
| Description | Removes dead air from multitrack podcast recordings |
| Visibility | Public |
| Initialize with README | **No** |
| Add .gitignore | **No** |
| Choose a licence | **No** |

Say no to all three — the repository already has a README, a `.gitignore` and
a LICENSE, and letting GitHub create its own gives you a conflict on the first
push for no benefit.

### 2. Push

```bash
git remote add origin https://github.com/pcmflores19-ph/wavefield.git
git push -u origin main
```

Git will ask you to sign in; a browser window handles it.

### 3. Fill in the About panel

On the repository page, the gear icon beside **About**:

- **Description**: Removes dead air from multitrack podcast recordings —
  exports to DaVinci Resolve or WAV
- **Topics**: `podcast`, `audio-editing`, `davinci-resolve`, `whisperx`,
  `python`, `vst3`, `video-editing`

Topics are how people actually find a repository like this.

## Every release

### 1. Set the version

`auto_cut/version.py` is the only place it lives — the About box, the
installer filename and the update check all read it from there.

```python
__version__ = "1.1.0"
```

Roughly: bug fixes bump the last number, new features the middle one.

### 2. Build

```bash
python packaging/build.py
```

Icon, plugins, PyInstaller, Inno Setup. Out comes
`packaging/output/Wavefield-Setup-<version>.exe`, around 100 MB.

**Install it and run a real episode before going further.** A frozen build can
fail in ways running from source never does.

### 3. Commit and tag

```bash
git add -A
git commit -m "Release 1.1.0"
git tag v1.1.0
git push && git push --tags
```

The tag must be `v` followed by the version. The update check strips the `v`
and compares the rest against `__version__`, so `v1.1.0` matches `1.1.0`. A tag
named anything else will not be recognised.

### 4. Publish it

**Releases ▸ Draft a new release** on the repository page:

- **Tag**: pick the `v1.1.0` you just pushed
- **Title**: `Wavefield 1.1.0`
- **Attach the installer**: drag `Wavefield-Setup-1.1.0.exe` into the box. This
  matters — without an attached file people have to build it themselves.
- **Paste the checksum** into the release notes. `build.py` prints it at the
  end, as `SHA-256: ...`. People use it to confirm their download is really
  the file you published — the only way to check that until signing lands.
- **Publish release**

Then check *Help ▸ Check for updates* in an older copy. It should offer the new
version. That is the whole feature working end to end.

## Notes

- The installer is ~100 MB, well under GitHub's 2 GB per-file limit. It is
  never committed to git — only attached to a release.
- **Windows will warn** the first time someone runs it: *"Windows protected
  your PC"*, because the executable is not code-signed. They click **More
  info ▸ Run anyway**. Signing needs a certificate at roughly $200/year, which
  is hard to justify for a free tool — but say so in the release notes so it
  does not look alarming.
- **Everything here is GPL-3** — the source and the built installer alike,
  because Wavefield bundles pedalboard and rnnoise, which are GPL. See
  [LICENSE](../LICENSE). This project was briefly and wrongly described as MIT;
  do not reintroduce that anywhere, in the repository or in release notes.
