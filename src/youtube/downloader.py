import yt_dlp

# This file is to download youtube video for testing.

ydl_opts = {
    'outtmpl': 'video/%(title)s.%(ext)s',
    'format': 'best[height<=1024]'
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=ZEcwuJ9Pr_c'])