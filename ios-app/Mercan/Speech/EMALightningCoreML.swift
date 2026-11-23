import CoreML
import Foundation

/// iOS-native host for the three published EMA Lightning Core ML stages.
/// This is a separate voice engine: it never interprets a .mercan LLM model.
/// Predictions run on this actor, not on the main/UI thread.
actor EMALightningCoreML {
    private var textModel: MLModel?
    private var soundModel: MLModel?
    private var decoderModel: MLModel?
    private var vocabulary: [String: Int] = [:]
    private let sampleRate = 48_000
    private let hop = 1_920
    private let latentSize = 64

    private func loadIfNeeded() throws {
        if textModel != nil { return }
        guard #available(iOS 26.0, *) else { throw EMAError.unsupportedSystem }
        guard EMAAssets.isInstalled else { throw EMAError.missingModels }

        let root = EMAAssets.directory
        let json = try Data(contentsOf: root.appendingPathComponent("voice.json"))
        guard let metadata = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw EMAError.badMetadata
        }
        let dictionaries = [metadata,
                            metadata["model"] as? [String: Any] ?? [:],
                            metadata["config"] as? [String: Any] ?? [:]]
        if let mapping = dictionaries.compactMap({ $0["stoi"] as? [String: Int] }).first {
            vocabulary = mapping
        } else {
            let letters = dictionaries.compactMap { object -> [String]? in
                for key in ["vocab", "alphabet", "characters"] {
                    if let list = object[key] as? [String] { return list }
                    if let text = object[key] as? String { return text.map(String.init) }
                }
                return nil
            }.first
            guard let letters, !letters.isEmpty else { throw EMAError.badMetadata }
            vocabulary = Dictionary(uniqueKeysWithValues: letters.enumerated().map { ($0.element, $0.offset) })
        }
        guard !vocabulary.isEmpty else { throw EMAError.badMetadata }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly // Compatible with Elphi's Core ML validation path.
        textModel = try MLModel(contentsOf: root.appendingPathComponent("EMAText.mlmodelc"), configuration: configuration)
        soundModel = try MLModel(contentsOf: root.appendingPathComponent("EMASound.mlmodelc"), configuration: configuration)
        decoderModel = try MLModel(contentsOf: root.appendingPathComponent("EMADecoder.mlmodelc"), configuration: configuration)
    }

    /// The caller gives complete sentences. Sentences over the model's nominal
    /// ten-second window are broken at word boundaries before running inference.
    func synthesize(_ sentence: String) throws -> [Float] {
        try loadIfNeeded()
        let normalized = normalize(sentence)
        guard !normalized.isEmpty else { return [] }
        let pieces = split(normalized, maxCharacters: 170)
        var output: [Float] = []
        for (index, piece) in pieces.enumerated() {
            let audio = try render(piece)
            output.append(contentsOf: audio)
            if index + 1 < pieces.count {
                output.append(contentsOf: repeatElement(0, count: sampleRate / 8))
            }
        }
        return output
    }

    private func normalize(_ source: String) -> String {
        var text = source.replacingOccurrences(of: "İ", with: "i")
            .replacingOccurrences(of: "I", with: "ı")
            .lowercased(with: Locale(identifier: "tr_TR"))
            .replacingOccurrences(of: "’", with: "'")
            .replacingOccurrences(of: "—", with: "-")
            .replacingOccurrences(of: "…", with: ".")
        // Lightweight Turkish number expansion. Full normalizer-tr parity
        // requires a separate native Rust bridge.
        let regex = try? NSRegularExpression(pattern: #"\b\d+\b"#)
        let ranges = regex?.matches(in: text, range: NSRange(text.startIndex..., in: text)) ?? []
        let formatter = NumberFormatter()
        formatter.numberStyle = .spellOut
        formatter.locale = Locale(identifier: "tr_TR")
        for match in ranges.reversed() {
            guard let range = Range(match.range, in: text),
                  let number = Int(text[range]),
                  let spoken = formatter.string(from: NSNumber(value: number)) else { continue }
            text.replaceSubrange(range, with: spoken)
        }
        let approved = text.map { character -> String in
            let value = String(character)
            return vocabulary[value] != nil ? value : " "
        }.joined()
        return approved.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    private func split(_ text: String, maxCharacters: Int) -> [String] {
        let words = text.split(separator: " ")
        var pieces: [String] = []
        var current = ""
        for word in words {
            let candidate = current.isEmpty ? String(word) : current + " " + word
            if candidate.count > maxCharacters && !current.isEmpty {
                pieces.append(finish(current))
                current = String(word)
            } else {
                current = candidate
            }
        }
        if !current.isEmpty { pieces.append(finish(current)) }
        return pieces
    }

    private func finish(_ value: String) -> String {
        let terminal = value.last.map { ".!?".contains($0) } ?? false
        if terminal { return value }
        return value + (vocabulary["."] != nil ? "." : "")
    }

    private func render(_ spoken: String) throws -> [Float] {
        guard let textModel, let soundModel, let decoderModel else { throw EMAError.missingModels }
        let characters = spoken.map(String.init)
        let count = characters.count
        guard count > 0 else { return [] }

        let ids = characters.map { Float(vocabulary[$0] ?? 1) }
        let ones = [Float](repeating: 1, count: count)
        let textOutput = try predict(textModel, inputs: [
            "ids": ([1, count], ids),
            "mask": ([1, count], ones)
        ])
        let outputs = textOutput.featureNames.compactMap { textOutput.featureValue(for: $0)?.multiArrayValue }
        guard let features = outputs.first(where: { $0.shape.count == 3 }),
              let durations = outputs.first(where: { $0.shape.count == 2 }) else {
            throw EMAError.incompatibleModel("EMAText outputs must include features [1,L,D] and durations [1,L]")
        }
        let hidden = features.shape.last?.intValue ?? 0
        guard hidden > 0 else { throw EMAError.incompatibleModel("empty text features") }
        var wordOfChar: [Int] = []
        var wordStart: [Int] = []
        var word = 0
        var start = 0
        for i in 0..<count {
            if i > 0 && characters[i - 1] == " " && characters[i] != " " {
                word += 1
                start = i
            }
            wordOfChar.append(word)
            wordStart.append(start)
        }
        var framesPerWord = [Float](repeating: 0, count: word + 1)
        for i in 0..<count { framesPerWord[wordOfChar[i]] += durations[i].floatValue }
        let frameCounts = framesPerWord.map { min(250, max(1, Int($0.rounded()))) }
        let totalFrames = min(3000, frameCounts.reduce(0, +))
        guard totalFrames > 0 else { return [] }
        var frameWord: [Float] = []
        var framePosition: [Float] = []
        for w in frameCounts.indices {
            for f in 0..<frameCounts[w] where frameWord.count < totalFrames {
                frameWord.append(Float(w))
                framePosition.append(Float(f) / Float(frameCounts[w]))
            }
        }

        let noise = gaussianNoise(count: 4 * totalFrames * latentSize)
        let latentProvider = try predict(soundModel, inputs: [
            "h": ([1, count, hidden], (0..<features.count).map { features[$0].floatValue }),
            "dur": ([1, count], (0..<count).map { durations[$0].floatValue }),
            "mask": ([1, count], ones),
            "cw": ([1, count], wordOfChar.map(Float.init)),
            "wstart": ([1, count], wordStart.map(Float.init)),
            "fw": ([1, totalFrames], frameWord),
            "fp": ([1, totalFrames], framePosition),
            "fmask": ([1, totalFrames], [Float](repeating: 1, count: totalFrames)),
            "noise": ([1, 4, totalFrames, latentSize], noise)
        ])
        guard let latents = latentProvider.featureNames.compactMap({
            latentProvider.featureValue(for: $0)?.multiArrayValue
        }).first else { throw EMAError.incompatibleModel("EMASound has no latents output") }

        let flat = (0..<latents.count).map { latents[$0].floatValue }
        var samples: [Float] = []
        var position = 0
        while position < totalFrames {
            let end = min(totalFrames, position + (position == 0 ? 25 : 100))
            let contextStart = max(0, position - 8)
            let contextEnd = min(totalFrames, end + 8)
            let n = contextEnd - contextStart
            let z = Array(flat[(contextStart * latentSize)..<(contextEnd * latentSize)])
            let output = try predict(decoderModel, inputs: [
                "z": ([1, n, latentSize], z),
                "lengths": ([1], [Float(n)])
            ])
            guard let audio = output.featureNames.compactMap({
                output.featureValue(for: $0)?.multiArrayValue
            }).first else { throw EMAError.incompatibleModel("EMADecoder has no audio output") }
            let lower = (position - contextStart) * hop
            let upper = min(audio.count, (end - contextStart) * hop)
            if lower < upper {
                samples.append(contentsOf: (lower..<upper).map { max(-1, min(1, audio[$0].floatValue)) })
            }
            position = end
        }
        return samples
    }

    /// Core ML conversion can fold padding masks and the decoder's length
    /// argument into the graph. Supply these only if the compiled model actually
    /// declares them. All other inputs remain mandatory and validated.
    private func predict(_ model: MLModel, inputs: [String: ([Int], [Float])]) throws -> MLFeatureProvider {
        let declared = model.modelDescription.inputDescriptionsByName
        let foldedInputs: Set<String> = ["mask", "fmask", "lengths"]
        var features: [String: MLFeatureValue] = [:]
        for (name, value) in inputs {
            guard let description = declared[name] else {
                if foldedInputs.contains(name) { continue }
                throw EMAError.incompatibleModel(
                    "unexpected input \(name); Core ML accepts: \(declared.keys.sorted().joined(separator: ", "))"
                )
            }
            guard let constraint = description.multiArrayConstraint else {
                throw EMAError.incompatibleModel("input \(name) is not a tensor")
            }
            let (shape, scalars) = value
            let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: constraint.dataType)
            guard array.count == scalars.count else {
                throw EMAError.incompatibleModel("incorrect \(name) tensor shape")
            }
            for i in scalars.indices { array[i] = NSNumber(value: scalars[i]) }
            features[name] = MLFeatureValue(multiArray: array)
        }
        let missing = declared.keys.filter { features[$0] == nil }.sorted()
        guard missing.isEmpty else {
            throw EMAError.incompatibleModel("Core ML requires: \(missing.joined(separator: ", "))")
        }
        return try model.prediction(from: MLDictionaryFeatureProvider(dictionary: features))
    }

    private func gaussianNoise(count: Int) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(count)
        while values.count < count {
            let a = max(1e-9, Double.random(in: 0..<1))
            let b = Double.random(in: 0..<1)
            let radius = sqrt(-2 * log(a))
            let angle = 2 * Double.pi * b
            values.append(Float(radius * cos(angle)))
            if values.count < count { values.append(Float(radius * sin(angle))) }
        }
        return values
    }
}
