import Foundation

struct LanguageOption: Identifiable, Hashable {
    let id: String
    let displayName: String

    func localizedDisplayName(in interfaceLanguageID: String) -> String {
        LanguageCatalog.displayName(for: id, in: interfaceLanguageID)
    }
}

enum LanguageCatalog {
    static let common: [LanguageOption] = [
        LanguageOption(id: "en", displayName: "English"),
        LanguageOption(id: "zh-Hans", displayName: "Chinese (Simplified)"),
        LanguageOption(id: "es", displayName: "Spanish"),
        LanguageOption(id: "de", displayName: "German"),
        LanguageOption(id: "ja", displayName: "Japanese"),
        LanguageOption(id: "fr", displayName: "French"),
        LanguageOption(id: "ko", displayName: "Korean"),
        LanguageOption(id: "ar", displayName: "Arabic"),
        LanguageOption(id: "pt", displayName: "Portuguese"),
        LanguageOption(id: "ru", displayName: "Russian"),
    ]

    static let speechInput: [LanguageOption] = [
        LanguageOption(id: "en", displayName: "English"),
        LanguageOption(id: "zh-Hans", displayName: "Chinese (Simplified)"),
        LanguageOption(id: "yue", displayName: "Cantonese"),
        LanguageOption(id: "es", displayName: "Spanish"),
        LanguageOption(id: "de", displayName: "German"),
        LanguageOption(id: "ja", displayName: "Japanese"),
        LanguageOption(id: "fr", displayName: "French"),
        LanguageOption(id: "it", displayName: "Italian"),
        LanguageOption(id: "ko", displayName: "Korean"),
        LanguageOption(id: "pt", displayName: "Portuguese"),
    ]

    /// Whisper (Groq cloud ASR) supports ~99 languages; the Apple gate does
    /// not apply when cloudASR is enabled. Ids are BCP-47-ish; the ASR engine
    /// already reduces them to the primary subtag whisper expects.
    static let cloudSpeechInput: [LanguageOption] = {
        let ids = ["zh-Hans", "zh-Hant", "yue", "en", "ja", "ko", "fr", "de", "es", "pt",
                   "ru", "it", "nl", "pl", "sv", "da", "no", "fi", "cs", "sk",
                   "sl", "hr", "bs", "sr", "mk", "bg", "uk", "be", "ro", "hu",
                   "el", "tr", "ar", "he", "fa", "ur", "hi", "bn", "ta", "te",
                   "kn", "ml", "mr", "gu", "pa", "ne", "si", "my", "km", "lo",
                   "th", "vi", "id", "ms", "tl", "sw", "am", "ha", "yo", "so",
                   "af", "sq", "hy", "az", "eu", "ca", "gl", "cy", "br", "oc",
                   "la", "lb", "lt", "lv", "et", "is", "ga", "mt", "ps", "sd",
                   "tk", "tt", "uz", "kk", "mn", "bo", "mi", "haw", "ln", "sn",
                   "su", "jv", "sa", "fo", "mg", "as"]
        return ids.map { LanguageOption(id: $0, displayName: $0) }
    }()

    /// DeepSeek translates between any languages it knows; offer a broad
    /// target list when cloudTranslation is enabled.
    static let cloudTranslationTargets: [LanguageOption] = {
        let ids = ["en", "zh-Hans", "zh-Hant", "ja", "ko", "fr", "de", "es", "pt", "ru",
                   "it", "nl", "pl", "sv", "da", "no", "fi", "cs", "sk", "sl",
                   "hr", "sr", "bg", "uk", "ro", "hu", "el", "tr", "ar", "he",
                   "fa", "ur", "hi", "bn", "ta", "te", "mr", "gu", "pa", "th",
                   "vi", "id", "ms", "tl", "sw", "af", "sq", "hy", "az", "ca",
                   "la", "lb", "lt", "lv", "et", "is", "ga", "yue", "mn", "ka"]
        return ids.map { LanguageOption(id: $0, displayName: $0) }
    }()

    static func sourceOptions(cloudASREnabled: Bool) -> [LanguageOption] {
        cloudASREnabled ? cloudSpeechInput : speechInput
    }

    static func targetOptions(cloudTranslationEnabled: Bool) -> [LanguageOption] {
        cloudTranslationEnabled ? cloudTranslationTargets : common
    }

    static func supportedSpeechInputLanguageID(for identifier: String, cloudASREnabled: Bool) -> String {
        if cloudASREnabled, cloudSpeechInput.contains(where: { $0.id == identifier }) {
            return identifier
        }
        return supportedSpeechInputLanguageID(for: identifier)
    }

    static func displayName(for identifier: String) -> String {
        if let known = (speechInput + common).first(where: { $0.id == identifier }) {
            return known.displayName
        }
        return Locale.current.localizedString(forIdentifier: identifier) ?? identifier
    }

    static func displayName(for identifier: String, in interfaceLanguageID: String) -> String {
        let locale = Locale(identifier: interfaceLanguageID)
        return locale.localizedString(forIdentifier: identifier)
            ?? displayName(for: identifier)
    }

    static func preferredInterfaceLanguageID(storedIdentifier: String?) -> String {
        AppLocalization.resolvedInterfaceLanguageID(storedIdentifier: storedIdentifier)
    }

    static func supportedSpeechInputLanguageID(for identifier: String) -> String {
        speechInput.contains(where: { $0.id == identifier }) ? identifier : "en"
    }

    static func speechLocaleIdentifier(for identifier: String) -> String {
        switch identifier {
        case "en": return "en-US"
        case "zh-Hans": return "zh-CN"
        case "yue": return "yue-CN"
        case "es": return "es-ES"
        case "de": return "de-DE"
        case "ja": return "ja-JP"
        case "fr": return "fr-FR"
        case "it": return "it-IT"
        case "ko": return "ko-KR"
        case "ar": return "ar-SA"
        case "pt": return "pt-BR"
        case "ru": return "ru-RU"
        default: return identifier
        }
    }
}
