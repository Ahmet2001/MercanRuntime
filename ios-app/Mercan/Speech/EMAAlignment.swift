import Foundation

/// The EMA Lightning aligner's external input tensors, calculated exactly from
/// the original PyTorch Acoustic.sound_stage and Aligner.forward formulas.
/// The Core ML EMASound conversion accepts h, cg, fg, allow and noise, rather
/// than the internal PyTorch cw/dur/fw/fp/mask inputs.
struct EMAAlignment {
    let cg: [Float]      // [1, L]: character positions
    let fg: [Float]      // [1, T]: frame positions
    let allow: [Float]   // [1, 1, T, L]: attention window mask

    static func make(
        durations: [Float],
        characterWords: [Int],
        wordStarts: [Int],
        frameWords: [Int],
        framePositions: [Float],
        lookback: Int,
        lookahead: Int,
        letterPositions: Bool
    ) -> EMAAlignment {
        let length = durations.count
        let wordCount = (characterWords.max() ?? -1) + 1
        precondition(length > 0 && wordCount > 0)
        precondition(characterWords.count == length && wordStarts.count == length)
        precondition(frameWords.count == framePositions.count)

        // PyTorch: c = dur.clamp(min=1e-4) * mask
        // cp = ((done - before.gather(1,wstart) - 0.5*c) /
        //       total.clamp(min=1e-8)).clamp(0,1) * mask
        let c = durations.map { max($0, 0.0001) }
        var before = [Float](repeating: 0, count: length)
        var wordDurations = [Float](repeating: 0, count: wordCount)
        var running: Float = 0
        for i in 0..<length {
            before[i] = running
            running += c[i]
            wordDurations[characterWords[i]] += c[i]
        }

        var wordLengths = [Int](repeating: 0, count: wordCount)
        for word in characterWords { wordLengths[word] += 1 }
        var wordOffsets = [Int](repeating: 0, count: wordCount)
        for word in 1..<wordCount {
            wordOffsets[word] = wordOffsets[word - 1] + wordLengths[word - 1]
        }

        var cg = [Float](repeating: 0, count: length)
        for i in 0..<length {
            let word = characterWords[i]
            let start = wordStarts[i]
            let cp = max(0, min(1,
                (before[i] + 0.5 * c[i] - before[start]) /
                max(wordDurations[word], 1e-8)
            ))
            cg[i] = letterPositions
                ? Float(wordOffsets[word]) + cp * Float(wordLengths[word])
                : Float(word) + cp
        }

        var fg = [Float]()
        fg.reserveCapacity(frameWords.count)
        var allow = [Float]()
        allow.reserveCapacity(frameWords.count * length)
        for (t, frameWord) in frameWords.enumerated() {
            fg.append(letterPositions
                ? Float(wordOffsets[frameWord]) + framePositions[t] * Float(wordLengths[frameWord])
                : Float(frameWord) + framePositions[t]
            )
            for charWord in characterWords {
                let delta = charWord - frameWord
                allow.append((delta >= -lookback && delta <= lookahead) ? 1 : 0)
            }
        }
        return EMAAlignment(cg: cg, fg: fg, allow: allow)
    }
}
