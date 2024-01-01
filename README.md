Detect loops in audio files.

Useful for [certain](https://rainbowdragoneyes.bandcamp.com/album/the-messenger-original-soundtrack-disc-ii-the-future) kinds of [tracks](https://music.apple.com/us/artist/falcom-sound-team-jdk/120174391).

For example, if a track consists of 3 parts:

    [ head ] [ loop body ] [ tail ]

This program can figure out the start and end of `[ loop body ]` so the track can be played like:

    [ head ] [ loop body ] [ loop body ] [ loop body ] ...

Requires `ffmpeg` binaries:
- `ffmpeg`: Decode audio files.
- `ffprobe` and `ffplay`: Read channel information, play audio streams (optional, only used by playback).

## Examples

Analyze a track. This usually takes less than 1 second:

    $ loop-detect track.flac
    {"start": 6.315828, "end": 111.915825, "confidence": 1035}

Analyze, then play in a infinite gapless loop:

    $ loop-detect -p track.flac
    ...

## Library interface

This project also provides a library interface for the loop detection algorithm. I intend to integrate it with [foobar2000](https://www.foobar2000.org/).

## License

MIT
