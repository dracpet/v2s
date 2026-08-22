import AVFoundation
import Foundation

/// Settings for the optional cloud ASR backend (OpenAI-compatible audio
/// transcription API — Groq whisper-large-v3-turbo by default). Used when
/// on-device/Apple-server speech recognition is unavailable, quota-limited,
/// or unreliable (e.g. zh-Hans on Intel Macs has no on-device model and the
/// server path is quota-bound).
struct CloudASRSettings: Codable, Equatable, Sendable {
    var enabled: Bool
    var baseURL: String // e.g. https://api.groq.com/openai/v1
    var apiKey: String
    var model: String   // e.g. whisper-large-v3-turbo

    static let disabled = CloudASRSettings(
        enabled: false,
        baseURL: "https://api.groq.com/openai/v1",
        apiKey: "",
        model: "whisper-large-v3-turbo"
    )
}

enum CloudASRError: LocalizedError {
    case missingAPIKey
    case invalidBaseURL(String)
    case httpError(Int, String)
    case malformedResponse

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "Cloud ASR is enabled but no API key is configured."
        case .invalidBaseURL(let url):
            return "Cloud ASR base URL is invalid: \(url)"
        case .httpError(let code, let detail):
            return "Cloud ASR request failed (HTTP \(code)): \(detail)"
        case .malformedResponse:
            return "Cloud ASR returned an unreadable response."
        }
    }
}

/// Chunked-upload ASR engine: accumulates 16 kHz mono PCM from the capture
/// pipeline, and every `stepSeconds` uploads the trailing `windowSeconds`
/// as a WAV to an OpenAI-compatible transcription endpoint. The newly
/// recognized tail (overlap-merged against the previously emitted text) is
/// emitted as committed sentences. Mirrors the proven live_pipe.py design:
/// sliding window + RMS silence gate + suffix/prefix overlap de-dup.
final class CloudASREngine: @unchecked Sendable {
    private let settings: CloudASRSettings
    private let languageCode: String
    private let emitSentence: @MainActor (String) -> Void
    private let reportFatal: @MainActor (String) -> Void

    // Sliding-window parameters (seconds @ 16 kHz int16 mono)
    private let sampleRate = 16_000
    private let windowSeconds = 8
    private let stepSeconds = 4
    private let rmsGate: Float = 0.008  // ~ int16 RMS 260; below → skip upload

    private let queue = DispatchQueue(label: "v2s.cloudASR", qos: .userInitiated)
    private var pcm = Data()          // int16 LE mono @16k
    private var bytesSinceUpload = 0
    private var printed = ""          // last emitted transcript text
    private var converter: AVAudioConverter?
    private var converterInputSignature: AudioFormatSignature?
    private var stopped = false
    private var consecutiveFailures = 0
    private var uploadInFlight = false

    private struct AudioFormatSignature: Equatable {
        let sampleRate: Double
        let channels: UInt32
        let commonFormat: AVAudioCommonFormat
    }

    private let contextPrompt: String

    init(
        settings: CloudASRSettings,
        localeIdentifier: String,
        contextualHints: [String] = [],
        emitSentence: @escaping @MainActor (String) -> Void,
        reportFatal: @escaping @MainActor (String) -> Void
    ) {
        // whisper-style `prompt` biases recognition toward domain terms —
        // the ASR-level fix for "TofinTV" → DolphinDB.
        self.contextPrompt = String(contextualHints.joined(separator: ", ").prefix(200))
        self.settings = settings
        // whisper-style APIs want ISO-639-1 ("zh", "en", …)
        self.languageCode = localeIdentifier.split(separator: "-").first.map(String.init) ?? localeIdentifier
        self.emitSentence = emitSentence
        self.reportFatal = reportFatal
    }

    func stop() {
        queue.sync { stopped = true }
    }

    /// Called on the capture queue with the post-processing buffer
    /// (already gain-boosted; format may vary by source).
    func ingest(_ buffer: AVAudioPCMBuffer) {
        queue.async { [weak self] in
            guard let self, self.stopped == false else { return }
            guard let chunk = self.convertToMono16kInt16(buffer) else { return }
            self.pcm.append(chunk)
            self.bytesSinceUpload += chunk.count

            let windowBytes = self.windowSeconds * self.sampleRate * 2
            if self.pcm.count > windowBytes {
                self.pcm = self.pcm.suffix(windowBytes)
            }
            let stepBytes = self.stepSeconds * self.sampleRate * 2
            guard self.bytesSinceUpload >= stepBytes else { return }
            self.bytesSinceUpload = 0

            guard self.rms(self.pcm) >= self.rmsGate else { return }
            guard self.uploadInFlight == false else { return }
            self.uploadInFlight = true
            let window = self.pcm
            Task { [weak self] in
                await self?.upload(window)
            }
        }
    }

    private func upload(_ window: Data) async {
        defer {
            queue.async { [weak self] in self?.uploadInFlight = false }
        }
        do {
            let text = try await transcribe(wavData: Self.wavWrap(window, sampleRate: sampleRate))
            queue.async { [weak self] in
                guard let self else { return }
                self.consecutiveFailures = 0
                let merged = self.overlapMergedNew(text)
                guard merged.isEmpty == false else { return }
                let sentence = merged
                Task { @MainActor in self.emitSentence(sentence) }
            }
        } catch {
            queue.async { [weak self] in
                guard let self else { return }
                self.consecutiveFailures += 1
                if self.consecutiveFailures >= 5 {
                    let message = "Cloud ASR keeps failing (\(self.consecutiveFailures) attempts): "
                        + error.localizedDescription
                    Task { @MainActor in self.reportFatal(message) }
                }
            }
        }
    }

    /// Drop the prefix that repeats the tail of what we already emitted
    /// (the sliding window re-hears the last few seconds).
    private func overlapMergedNew(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.isEmpty == false else { return "" }
        var k = 0
        let maxOverlap = min(printed.count, trimmed.count)
        if maxOverlap > 0 {
            for i in 1...maxOverlap {
                if printed.hasSuffix(String(trimmed.prefix(i))) {
                    k = i
                }
            }
        }
        let new = String(trimmed.dropFirst(k)).trimmingCharacters(in: .whitespacesAndNewlines)
        if new.isEmpty == false {
            printed = trimmed
        }
        return new
    }

    private func transcribe(wavData: Data) async throws -> String {
        guard settings.apiKey.isEmpty == false else {
            throw CloudASRError.missingAPIKey
        }
        let base = settings.baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let url = URL(string: base + "/audio/transcriptions") else {
            throw CloudASRError.invalidBaseURL(settings.baseURL)
        }

        let boundary = "----v2scloudasr\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))"
        var body = Data()
        func field(_ name: String, _ value: String) {
            body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n".data(using: .utf8)!)
        }
        field("model", settings.model.isEmpty ? "whisper-large-v3-turbo" : settings.model)
        field("language", languageCode)
        if contextPrompt.isEmpty == false {
            field("prompt", contextPrompt)
        }
        field("response_format", "json")
        body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
        body.append(wavData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 90
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(settings.apiKey)", forHTTPHeaderField: "Authorization")
        request.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            let detail = String(data: data.prefix(300), encoding: .utf8) ?? ""
            throw CloudASRError.httpError(http.statusCode, detail)
        }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = object["text"] as? String
        else {
            throw CloudASRError.malformedResponse
        }
        return text
    }

    // MARK: - Audio conversion helpers

    private func convertToMono16kInt16(_ input: AVAudioPCMBuffer) -> Data? {
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: true
        )!
        let signature = AudioFormatSignature(
            sampleRate: input.format.sampleRate,
            channels: input.format.channelCount,
            commonFormat: input.format.commonFormat
        )
        if converter == nil || converterInputSignature != signature {
            converter = AVAudioConverter(from: input.format, to: targetFormat)
            converterInputSignature = signature
        }
        guard let converter else { return nil }

        let capacity = AVAudioFrameCount(
            ceil(Double(input.frameLength) * targetFormat.sampleRate / input.format.sampleRate)
        ) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else {
            return nil
        }
        var consumed = false
        var conversionError: NSError?
        converter.convert(to: out, error: &conversionError) { _, status in
            if consumed {
                status.pointee = .noDataNow
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return input
        }
        guard conversionError == nil, out.frameLength > 0,
              let channelData = out.int16ChannelData
        else { return nil }

        let byteCount = Int(out.frameLength) * MemoryLayout<Int16>.size
        return Data(bytes: channelData[0], count: byteCount)
    }

    private func rms(_ data: Data) -> Float {
        let count = data.count / 2
        guard count > 0 else { return 0 }
        return data.withUnsafeBytes { raw -> Float in
            let samples = raw.bindMemory(to: Int16.self)
            var sum: Double = 0
            for i in 0..<count {
                let v = Double(samples[i])
                sum += v * v
            }
            return Float(sqrt(sum / Double(count)) / 32768.0)
        }
    }

    private static func wavWrap(_ pcm: Data, sampleRate: Int) -> Data {
        var out = Data()
        let byteRate = UInt32(sampleRate * 2)
        let dataSize = UInt32(pcm.count)
        func str(_ s: String) { out.append(s.data(using: .ascii)!) }
        func u32(_ v: UInt32) { var x = v.littleEndian; out.append(Data(bytes: &x, count: 4)) }
        func u16(_ v: UInt16) { var x = v.littleEndian; out.append(Data(bytes: &x, count: 2)) }
        str("RIFF"); u32(36 + dataSize); str("WAVE")
        str("fmt "); u32(16); u16(1); u16(1); u32(UInt32(sampleRate)); u32(byteRate); u16(2); u16(16)
        str("data"); u32(dataSize); out.append(pcm)
        return out
    }
}
