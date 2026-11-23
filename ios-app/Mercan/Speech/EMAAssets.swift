import CoreML
import CryptoKit
import Foundation

/// EMA Lightning: 5.6M acoustic model + 3M HiFi-GAN decoder.
/// Distribution: elphi-si/elphi (Apache-2.0), pinned to the published file hashes.
/// Downloads weights only; text and audio never leave the device.
enum EMAAssets {
    struct Asset {
        let path: String
        let bytes: Int
        let sha256: String
    }

    static let assets: [Asset] = [
        .init(path: "EMAText.mlpackage/Data/com.apple.CoreML/model.mlmodel", bytes: 29_210, sha256: "24ec2f1acc76eb34fa8979e34e2b29d6fb668533794050957e251ac8e2b3b1ee"),
        .init(path: "EMAText.mlpackage/Data/com.apple.CoreML/weights/weight.bin", bytes: 4_792_000, sha256: "f2ec3166e629929b159fbe12b06cb445a202a5a16e35e78b3b5c30dbefa60db3"),
        .init(path: "EMAText.mlpackage/Manifest.json", bytes: 617, sha256: "94ca43b45ecdbefc8d250ded9b325132ecb241ce4df42c5d17ee4c67003b755b"),
        .init(path: "EMASound.mlpackage/Data/com.apple.CoreML/model.mlmodel", bytes: 275_635, sha256: "b38839114c7b430312881e27d007f955273697ede50739d2e8fcfed90e965b65"),
        .init(path: "EMASound.mlpackage/Data/com.apple.CoreML/weights/weight.bin", bytes: 15_319_296, sha256: "3081db93ad88e7ed0fbe3864e66c3e0cbd8723adac88e0eada64a41ebe92b827"),
        .init(path: "EMASound.mlpackage/Manifest.json", bytes: 617, sha256: "e558217deb6ec55e3aaf91f587fc0743e1596c257db76de43797b4f7e951d872"),
        .init(path: "EMADecoder.mlpackage/Data/com.apple.CoreML/model.mlmodel", bytes: 171_303, sha256: "a952147fbfe0bd43e878a61eae3d8718e2307dcf8475664e8d75c06ca2883ce5"),
        .init(path: "EMADecoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin", bytes: 11_984_304, sha256: "ed8e4a1701ae8b7b2c1298db38ce34d83fbf107f13f82b77281c1a87f6fcbab9"),
        .init(path: "EMADecoder.mlpackage/Manifest.json", bytes: 617, sha256: "bbf2a573a9fc80dddc5d80f03e9a35d33b6d2a291dfa741aa1fd6bd8290b1acd"),
        .init(path: "voice.json", bytes: 1_443, sha256: "fba9baf368a1176f9b43d9c37c56a8497684c0d0c14c4ef074732523f14300d8")
    ]

    static var directory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("EMA-Lightning-CoreML-v1", isDirectory: true)
    }

    static var isInstalled: Bool {
        let root = directory
        return FileManager.default.fileExists(atPath: root.appendingPathComponent("verified").path) &&
            ["EMAText", "EMASound", "EMADecoder"].allSatisfy {
                FileManager.default.fileExists(atPath: root.appendingPathComponent("\($0).mlmodelc").path)
            }
    }

    /// Download to a staging directory, verify every byte, compile, then atomically install.
    /// The app's LFM .mercan file is neither moved nor unloaded.
    static func install(progress: @escaping @Sendable (Double) -> Void) async throws {
        guard #available(iOS 26.0, *) else {
            throw EMAError.unsupportedSystem
        }
        let manager = FileManager.default
        let staging = manager.temporaryDirectory.appendingPathComponent("ema-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: staging) }

        for (index, asset) in assets.enumerated() {
            try Task.checkCancellation()
            let urlString = "https://huggingface.co/elphi-si/elphi/resolve/main/\(asset.path)?download=true"
            guard let url = URL(string: urlString) else { throw EMAError.badDownload(asset.path) }
            let (temporary, response) = try await URLSession.shared.download(from: url)
            guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
                throw EMAError.badDownload(asset.path)
            }
            let data = try Data(contentsOf: temporary)
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            guard data.count == asset.bytes && digest == asset.sha256 else {
                throw EMAError.corruptAsset(asset.path)
            }
            let destination = staging.appendingPathComponent(asset.path)
            try manager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: destination, options: .atomic)
            progress(Double(index + 1) / Double(assets.count + 3))
        }

        // Core ML compilation may be expensive; never run it on the SwiftUI actor.
        for (index, name) in ["EMAText", "EMASound", "EMADecoder"].enumerated() {
            try Task.checkCancellation()
            let compiled = try await MLModel.compileModel(at: staging.appendingPathComponent("\(name).mlpackage"))
            try manager.moveItem(at: compiled, to: staging.appendingPathComponent("\(name).mlmodelc"))
            progress(Double(assets.count + index + 1) / Double(assets.count + 3))
        }
        try "verified".write(to: staging.appendingPathComponent("verified"), atomically: true, encoding: .utf8)
        let target = directory
        try manager.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        let backup = target.deletingLastPathComponent().appendingPathComponent("EMA-backup-\(UUID().uuidString)")
        let existing = manager.fileExists(atPath: target.path)
        if existing { try manager.moveItem(at: target, to: backup) }
        do {
            try manager.moveItem(at: staging, to: target)
            if existing { try? manager.removeItem(at: backup) }
        } catch {
            if existing { try? manager.moveItem(at: backup, to: target) }
            throw error
        }
    }
}

enum EMAError: LocalizedError {
    case unsupportedSystem, missingModels, badDownload(String), corruptAsset(String)
    case badMetadata, incompatibleModel(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedSystem: return "EMA Lightning Core ML requires iOS 26 or later."
        case .missingModels: return "Install the EMA Lightning voice from Settings first."
        case .badDownload(let file): return "Could not download \(file)."
        case .corruptAsset(let file): return "EMA model hash mismatch: \(file)."
        case .badMetadata: return "EMA voice.json could not be read."
        case .incompatibleModel(let reason): return "EMA Core ML contract mismatch: \(reason)."
        }
    }
}
