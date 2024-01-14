Detect loops in audio files.

Useful for [certain](https://rainbowdragoneyes.bandcamp.com/album/the-messenger-original-soundtrack-disc-ii-the-future) kinds of [tracks](https://music.apple.com/us/artist/falcom-sound-team-jdk/120174391).

For example, if a track consists of these parts:

    [ head ] [ loop start | loop middle | loop end ] [ loop start ] [ tail ]

This program can figure out the start and end of `[ loop ]` so the track can be played like:

    [ head ] [ loop ] [ loop ] [ loop ] ...

Requires `ffmpeg` binaries:
- `ffmpeg`: Decode audio files.
- `ffprobe` and `ffplay`: Read channel information, play audio streams (optional, only used by playback).

## Examples

Analyze a track. This usually takes less than 1 second:

    $ loop-detect track.flac
    {"start": 6.3, "end": 111.9, "confidence": 0.88}

Analyze, then play in an infinite gapless loop:

    $ loop-detect -p track.flac
    ...

Analyze multiple files:

    $ loop-detect -q *.flac
    {"a.flac":{"start":4.2,"end":106.6,"confidence":0.99},"b.flac":null}

To get indented JSON output, use:

    $ loop-detect -q *.flac | python3 -m json.tool

## Library interface

This project also provides a library interface for the loop detection algorithm. I intend to integrate it with [foobar2000](https://www.foobar2000.org/).

## License

MIT
