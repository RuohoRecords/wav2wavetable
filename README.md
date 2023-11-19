## wav2wavetable

Experiments to process recorded notes from a synthesizer or other source into single-cycle waveforms, ready to be imported into a Wavetable synth like Serum, Vital, etc.

Pracical Challenges:
- Many synthesizers do not generated notes with equal-sized cycles, that fit an exact (or integral) number of samples.
- Individual cycles can therefore be of different (though not fully random) lengths
- Individual cycles can also have {no, a single, or dozens of} zero crossings, making visual/manual waveform analysis tedious 😅😅

### Manual workflow
- Record a couple seconds of a held note on a synth (say Reaktor's Oki Computer)
- Chop it into cycles
- Export each cycle individually from left to right
- Import (or drag) batch of single-cycle WAVs into Serum

### Tools

Automatic tools need to 
- Reads a WAV into memory
- Detect individual cycles regardless of complexity and zero crossings
- Snip the WAV at individual cycle points and export to a new file on disk
