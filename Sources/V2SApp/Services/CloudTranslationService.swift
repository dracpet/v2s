import Foundation

/// Settings for the optional OpenAI-compatible cloud translation backend
/// (DeepSeek by default). When enabled, v2s routes translation through the
/// configured endpoint instead of Apple's Translation framework — useful on
/// machines where on-device translation resources are unavailable, or where a
/// domain-tuned LLM gives better live-subtitle quality.
struct CloudTranslationSettings: Codable, Equatable, Sendable {
    var enabled: Bool
    var baseURL: String       // e.g. https://api.deepseek.com
    var apiKey: String
    var model: String         // e.g. deepseek-v4-flash
    var domainContext: String // optional free-text domain hint for the prompt
    var slideContextEnabled: Bool // vision-model reads the slide for terms
    var visionModel: String       // e.g. deepseek-v4-flash-vision-exp

    static let disabled = CloudTranslationSettings(
        enabled: false,
        baseURL: "https://api.deepseek.com",
        apiKey: "",
        model: "deepseek-chat",
        domainContext: "",
        slideContextEnabled: false,
        visionModel: "deepseek-v4-flash-vision-exp"
    )

    // Custom decoder so settings files written before slide-context fields
    // existed keep loading (a missing key must NOT reset the whole struct
    // and silently drop the user's API key).
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        enabled = (try? c.decodeIfPresent(Bool.self, forKey: .enabled)) ?? false
        baseURL = (try? c.decodeIfPresent(String.self, forKey: .baseURL))
            ?? CloudTranslationSettings.disabled.baseURL
        apiKey = (try? c.decodeIfPresent(String.self, forKey: .apiKey)) ?? ""
        model = (try? c.decodeIfPresent(String.self, forKey: .model))
            ?? CloudTranslationSettings.disabled.model
        domainContext = (try? c.decodeIfPresent(String.self, forKey: .domainContext)) ?? ""
        slideContextEnabled = (try? c.decodeIfPresent(Bool.self, forKey: .slideContextEnabled)) ?? false
        visionModel = (try? c.decodeIfPresent(String.self, forKey: .visionModel))
            ?? CloudTranslationSettings.disabled.visionModel
    }

    init(enabled: Bool, baseURL: String, apiKey: String, model: String,
         domainContext: String, slideContextEnabled: Bool, visionModel: String) {
        self.enabled = enabled
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.model = model
        self.domainContext = domainContext
        self.slideContextEnabled = slideContextEnabled
        self.visionModel = visionModel
    }
}

enum CloudTranslationError: LocalizedError {
    case missingAPIKey
    case invalidBaseURL(String)
    case httpError(Int, String)
    case malformedResponse

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "Cloud translation is enabled but no API key is configured."
        case .invalidBaseURL(let url):
            return "Cloud translation base URL is invalid: \(url)"
        case .httpError(let code, let detail):
            return "Cloud translation request failed (HTTP \(code)): \(detail)"
        case .malformedResponse:
            return "Cloud translation returned an unreadable response."
        }
    }
}

/// OpenAI-compatible chat-completions translation client.
///
/// Mirrors the proven design of a live-subtitle pipeline: a domain-aware
/// system prompt, the user glossary injected as exact term mappings, and a
/// short rolling window of previous source→target pairs so sentence fragments
/// translate coherently.
final class CloudTranslationService: Sendable {
    private static let maxInputChars = 400
    // Reasoning-style models (deepseek-v4-flash etc.) burn most of the budget
    // on hidden reasoning before emitting content — 300 starves them into
    // returning empty translations. 1024 leaves room for reasoning + output.
    private static let maxOutputTokens = 1024

    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    func translate(
        _ text: String,
        from sourceLanguageID: String,
        to targetLanguageID: String,
        settings: CloudTranslationSettings,
        glossary: [String: String],
        history: [(source: String, target: String)],
        slideContext: String = ""
    ) async throws -> String {
        guard settings.apiKey.isEmpty == false else {
            throw CloudTranslationError.missingAPIKey
        }
        let base = settings.baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let url = URL(string: base + "/chat/completions") else {
            throw CloudTranslationError.invalidBaseURL(settings.baseURL)
        }

        let sourceName = Self.languageName(for: sourceLanguageID)
        let targetName = Self.languageName(for: targetLanguageID)

        var system = "You are translating live speech from \(sourceName) into \(targetName) "
            + "for real-time subtitles. The input comes from live speech recognition and "
            + "may contain stutters, duplicated fragments, and misheard names. Translate "
            + "into natural, precise \(targetName). Output ONLY the translation."
        let context = settings.domainContext.trimmingCharacters(in: .whitespacesAndNewlines)
        if context.isEmpty == false {
            system += " Context: \(context)"
        }
        let slide = slideContext.trimmingCharacters(in: .whitespacesAndNewlines)
        if slide.isEmpty == false {
            system += " The speaker's current slide contains these terms — they hold "
                + "the CORRECT spelling of what is being discussed. Lock these exact "
                + "spellings in your output (never re-transliterate or vary them): \(slide)."
        }
        if glossary.isEmpty == false {
            let mappings = glossary
                .sorted { $0.key.count > $1.key.count }
                .prefix(40)
                .map { "\($0.key) = \($0.value)" }
                .joined(separator: "; ")
            system += " Use these term mappings exactly: \(mappings)."
        }

        var messages: [[String: String]] = [["role": "system", "content": system]]
        for pair in history.suffix(3) {
            messages.append(["role": "user", "content": pair.source])
            messages.append(["role": "assistant", "content": pair.target])
        }
        messages.append(["role": "user", "content": String(text.prefix(Self.maxInputChars))])

        let model = settings.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let payload: [String: Any] = [
            "model": model.isEmpty ? "deepseek-chat" : model,
            "messages": messages,
            "stream": false,
            "temperature": 0.2,
            "max_tokens": Self.maxOutputTokens,
            // Live subtitles want speed over depth: cap the reasoning budget.
            // Measured 2026-08-22: 3.0s→1.2s/line, reasoning_tokens 172→15,
            // translation quality unchanged.
            "reasoning_effort": "low",
        ]

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(settings.apiKey)", forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)

        var (data, response) = try await session.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode == 429 {
            // Rate-limit transient: one short retry beats a leaked CN row.
            try await Task.sleep(nanoseconds: 2_000_000_000)
            (data, response) = try await session.data(for: request)
        }
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            let detail = String(data: data.prefix(300), encoding: .utf8) ?? ""
            throw CloudTranslationError.httpError(http.statusCode, detail)
        }

        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = object["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? String
        else {
            throw CloudTranslationError.malformedResponse
        }

        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        // Source-echo guard: with low reasoning effort the model sometimes
        // echoes Chinese back when the target is English. Detect CJK-heavy
        // output and retry once with a firmer instruction.
        if Self.isCJKHeavy(trimmed), targetLanguageID.hasPrefix("zh") == false,
           messages.last?["content"]?.contains("entirely in the target language") == false {
            var retryMessages = messages
            retryMessages[retryMessages.count - 1] = [
                "role": "user",
                "content": (messages.last?["content"] ?? "")
                    + "\n\nTranslate; the reply must be entirely in the target language, no Chinese.",
            ]
            let retryPayload: [String: Any] = [
                "model": payload["model"]!, "messages": retryMessages, "stream": false,
                "temperature": 0.2, "max_tokens": Self.maxOutputTokens,
                "reasoning_effort": "low",
            ]
            var retryRequest = URLRequest(url: url)
            retryRequest.httpMethod = "POST"
            retryRequest.timeoutInterval = 60
            retryRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
            retryRequest.setValue("Bearer \(settings.apiKey)", forHTTPHeaderField: "Authorization")
            retryRequest.httpBody = try JSONSerialization.data(withJSONObject: retryPayload)
            let (retryData, retryResponse) = try await session.data(for: retryRequest)
            if let http = retryResponse as? HTTPURLResponse, http.statusCode == 200,
               let obj = try JSONSerialization.jsonObject(with: retryData) as? [String: Any],
               let ch = obj["choices"] as? [[String: Any]],
               let msg = ch.first?["message"] as? [String: Any],
               let retryContent = msg["content"] as? String {
                let retryTrimmed = retryContent.trimmingCharacters(in: .whitespacesAndNewlines)
                if retryTrimmed.isEmpty == false {
                    Self.logEntry(text: text, translated: retryTrimmed, sourceLanguageID: sourceLanguageID,
                                  targetLanguageID: targetLanguageID, model: settings.model,
                                  slideContext: slideContext, domainContext: settings.domainContext)
                    return retryTrimmed
                }
            }
        }

        Self.logEntry(text: text, translated: trimmed, sourceLanguageID: sourceLanguageID,
                      targetLanguageID: targetLanguageID, model: settings.model,
                      slideContext: slideContext, domainContext: settings.domainContext)
        return trimmed
    }

    /// Appends one src→tgt pair to translation_log.jsonl in Application
    /// Support/v2s — the ground-truth sample stream for quality analysis.
    private static func logEntry(
        text: String, translated: String,
        sourceLanguageID: String, targetLanguageID: String,
        model: String, slideContext: String, domainContext: String
    ) {
        let ts = ISO8601DateFormatter().string(from: Date())
        let obj: [String: Any] = [
            "ts": ts, "src": text, "tgt": translated,
            "lang": "\(sourceLanguageID)->\(targetLanguageID)",
            "model": model, "ctxLen": slideContext.count, "domainLen": domainContext.count
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: obj),
              let line = String(data: data, encoding: .utf8) else { return }
        guard let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else { return }
        let url = dir.appendingPathComponent("v2s").appendingPathComponent("translation_log.jsonl")
        let lineWithNewline = line + "\n"
        if FileManager.default.fileExists(atPath: url.path) {
            if let fh = try? FileHandle(forWritingTo: url) {
                defer { try? fh.close() }
                _ = try? fh.seekToEnd()
                try? fh.write(contentsOf: lineWithNewline.data(using: .utf8) ?? Data())
            }
        } else {
            try? lineWithNewline.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    private static func isCJKHeavy(_ text: String) -> Bool {
        guard text.isEmpty == false else { return false }
        let cjk = text.unicodeScalars.filter {
            (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value)
        }.count
        return Double(cjk) / Double(text.count) > 0.2
    }

    private static func languageName(for identifier: String) -> String {
        let locale = Locale(identifier: "en")
        return locale.localizedString(forIdentifier: identifier)
            ?? locale.localizedString(forLanguageCode: identifier)
            ?? identifier
    }
}
