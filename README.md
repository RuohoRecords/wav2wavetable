# wav2wavetable

Process a recorded note (an audio waveform) from a synthesizer or other source into single-cycle waveforms, 
ready to be imported into a Wavetable synth like Serum, Phaseplant, Vital, etc.

### Usage
```python3 wav2wavetable.py <recorded-note-audio-filename>```

### Algorithm
- Read a WAV into memory
- Detect individual cycles regardless of complexity and zero crossings, using [autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation)
- Find [prominent peaks in the autocorrelation function](https://stackoverflow.com/questions/1713335/peak-finding-algorithm-for-python-scipy), 
these correspond to the main period, and our single-cycle cut points
- Snip the WAV at the single cycle points and export all new files to disk

### Practical Challenges:
- Many synthesizers do not generated notes with equal-sized cycles, that fit an exact (or integral) number of samples.
- Individual cycles can therefore be of different (though not fully random) lengths
- Individual cycles can also have {no, a single, or dozens of} zero crossings, making visual/manual waveform analysis tedious 😅😅

### Manual workflow
- Record a couple seconds of a held note on a synth (say Reaktor's Oki Computer)
- Chop it into cycles
- Export each cycle individually from left to right
- Import (or drag) batch of single-cycle WAVs into Serum


