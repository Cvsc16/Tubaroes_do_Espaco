## Prompt to Paste into Claude

(Copy the text below into your conversation with Claude and attach the image `comparative_analysis.png`.)

---

**Subject:** Spectrogram Analysis for Signal Detection

Hi Claude, I need your help interpreting the attached image, which compares two audio recordings.

**Project context:**
We want to find a specific sound (the “signature” in `assinatura.wav`) inside a longer, noisier track (`final track.wav`). The image shows a comparative analysis of both sounds.

**Request:**

1. **Look at the bottom-right chart (“Signature - Power Spectral Density”).** What is the frequency band (Hz on the X-axis) where the orange line shows the highest energy concentration? Please provide approximate minimum and maximum values for that main band.

2. **Now analyze the middle-left chart (“Final Track - Spectrogram”).** Using the frequency band you identified in step 1, at which times (seconds on the X-axis) do you see a bright “blob” or burst of energy matching that same band?

I need this information to design an audio filter and continue the project. Thanks!

---

## Why I framed the prompt this way

The prompt is intentionally direct so Claude returns the exact parameters we need for the next technical step.

- **Question 1 (Analyze the Signature):** forces Claude to extract the frequency “fingerprint” of the target sound, giving us the band for our filter (e.g., “the main energy lies between 400 Hz and 1500 Hz”).
- **Question 2 (Spot the Signature in the Full Track):** asks Claude to act as a “visual detector,” locating the same band in the full spectrogram. That yields approximate timestamps for the bite events, which we can compare with our own detection script later.

With Claude’s answers, we will have the exact parameters to finalize the algorithm.
