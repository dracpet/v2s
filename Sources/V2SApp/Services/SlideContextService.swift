import AppKit
import Foundation
import ScreenCaptureKit

/// Slide-context service: periodically captures the source app's window,
/// and ONLY when the picture materially changes (cheap thumbnail diff gate)
/// sends it to a vision model (DeepSeek-vision by default) to extract the
/// slide's technical terms with exact spelling. The terms feed the cloud
/// translation prompt, so names that only exist on screen (FlowMQ, …) are
/// translated correctly. Ports the proven live-translate slide_live.py idea
/// from local OCR to cloud vision — and captures the whole window, so no
/// region calibration is needed.
@MainActor
final class SlideContextService: ObservableObject {
    /// Latest slide terms, ready to inject into the translation prompt.
    @Published private(set) var currentTerms: String = ""

    private let bundleIDProvider: () -> String?
    private let settingsProvider: () -> CloudTranslationSettings

    private var timer: Timer?
    private var lastThumb: [UInt8]?
    /// Rolling union of the last few slides' terms — talk vocabulary
    /// accumulates; a term introduced slides ago is still spoken.
    private var slideHistory: [String] = []
    private let maxSlidesKept = 5
    private var visionInFlight = false
    private var lastVisionCall = Date.distantPast

    private let pollInterval: TimeInterval = 4
    private let visionMinInterval: TimeInterval = 10
    private let thumbSize = 48          // 48x27 luma grid
    private let diffThreshold = 6.0     // mean abs luma delta (0-255)

    init(
        bundleIDProvider: @escaping () -> String?,
        settingsProvider: @escaping () -> CloudTranslationSettings
    ) {
        self.bundleIDProvider = bundleIDProvider
        self.settingsProvider = settingsProvider
    }

    func start() {
        guard timer == nil else { return }
        let timer = Timer(timeInterval: pollInterval, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in self?.tick() }
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    func stop() {
        timer?.invalidate()
        timer = nil
        currentTerms = ""
        lastThumb = nil
        slideHistory.removeAll()
    }

    private var requestedCaptureAccess = false

    private func tick() {
        let settings = settingsProvider()
        guard settings.enabled, settings.slideContextEnabled else { return }
        // Preflight BEFORE touching ScreenCaptureKit: without permission the
        // SCK access itself re-triggers the system prompt on every poll.
        guard CGPreflightScreenCaptureAccess() else {
            if requestedCaptureAccess == false {
                requestedCaptureAccess = true
                _ = CGRequestScreenCaptureAccess()
            }
            return
        }
        Task { await captureAndMaybeAnalyze(settings: settings) }
    }

    private func captureAndMaybeAnalyze(settings: CloudTranslationSettings) async {
        guard let bundleID = bundleIDProvider(),
              let image = await captureWindowImage(bundleID: bundleID),
              let thumb = thumbnailLuma(image)
        else { return }

        if let last = lastThumb, meanAbsDiff(last, thumb) < diffThreshold {
            return  // slide unchanged — the whole point: vision calls stay rare
        }
        lastThumb = thumb

        guard visionInFlight == false,
              Date().timeIntervalSince(lastVisionCall) >= visionMinInterval
        else { return }
        visionInFlight = true
        lastVisionCall = Date()
        defer { visionInFlight = false }

        guard let jpeg = jpegData(image, maxWidth: 1280, quality: 0.6) else { return }
        Self.saveSnapshot(jpeg)
        do {
            let terms = try await analyzeSlide(jpeg: jpeg, settings: settings)
            if terms.isEmpty == false {
                slideHistory.append(terms)
                if slideHistory.count > maxSlidesKept {
                    slideHistory.removeFirst(slideHistory.count - maxSlidesKept)
                }
                currentTerms = Self.mergedTerms(slideHistory)
            }
        } catch {
            // Silent: slide context is best-effort; translation must never
            // fail because the vision sidecar errored.
        }
    }

    // MARK: - Capture

    private func captureWindowImage(bundleID: String) async -> CGImage? {
        // SCShareableContent.current: sync caller-isolated snapshot (macOS 14.4+);
        // the async class-method variant is not exposed by the CLT SDK here.
        guard let content = try? SCShareableContent.current else { return nil }  // throws without Screen Recording permission

        guard let window = content.windows.first(where: {
            $0.owningApplication?.bundleIdentifier == bundleID && $0.isOnScreen
        }) else { return nil }

        let filter = SCContentFilter(desktopIndependentWindow: window)
        let config = SCStreamConfiguration()
        config.width = Int(window.frame.width)
        config.height = Int(window.frame.height)
        return try? await SCScreenshotManager.captureImage(
            contentFilter: filter, configuration: config
        )
    }

    // MARK: - Diff gate

    private func thumbnailLuma(_ image: CGImage) -> [UInt8]? {
        let w = thumbSize, h = max(1, thumbSize * image.height / max(image.width, 1))
        guard let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w, space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }
        ctx.interpolationQuality = .low
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let data = ctx.data else { return nil }
        let count = w * h
        let ptr = data.bindMemory(to: UInt8.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    private func meanAbsDiff(_ a: [UInt8], _ b: [UInt8]) -> Double {
        guard a.count == b.count, a.isEmpty == false else { return .infinity }
        var sum = 0
        for i in 0..<a.count {
            sum += abs(Int(a[i]) - Int(b[i]))
        }
        return Double(sum) / Double(a.count)
    }

    private func jpegData(_ image: CGImage, maxWidth: CGFloat, quality: Double) -> Data? {
        let scale = min(1.0, maxWidth / CGFloat(image.width))
        let w = Int(CGFloat(image.width) * scale)
        let h = Int(CGFloat(image.height) * scale)
        guard let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else { return nil }
        ctx.interpolationQuality = .medium
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let scaled = ctx.makeImage() else { return nil }
        let rep = NSBitmapImageRep(cgImage: scaled)
        return rep.representation(using: .jpeg, properties: [.compressionFactor: quality])
    }

    /// Merge slide term lists (newest last), dedup case-insensitively,
    /// preserve order, cap the prompt footprint.
    private static func mergedTerms(_ slides: [String]) -> String {
        var seen = Set<String>()
        var result: [String] = []
        for slide in slides {
            for raw in slide.split(separator: ",") {
                let term = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                let key = term.lowercased()
                guard term.isEmpty == false, seen.insert(key).inserted else { continue }
                result.append(term)
            }
        }
        let joined = result.suffix(60).joined(separator: ", ")
        return String(joined.prefix(600))
    }

    /// Snapshots persist only on slide change (the diff gate upstream), so
    /// storage stays tiny: one JPEG per slide shown.
    private static let snapshotDir: URL = {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("v2s/slides", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    private static func saveSnapshot(_ jpeg: Data) {
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        try? jpeg.write(to: snapshotDir.appendingPathComponent("slide-\(stamp).jpg"))
    }

    // MARK: - Vision call

    private func analyzeSlide(jpeg: Data, settings: CloudTranslationSettings) async throws -> String {
        let base = settings.baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let url = URL(string: base + "/chat/completions"),
              settings.apiKey.isEmpty == false
        else { return "" }

        // Slides are usually in Chinese: extract bilingual pairs so the
        // injected context pushes the translator toward English output
        // instead of biasing it toward echoing Chinese source text.
        let prompt = "This is a screenshot of the presentation slide the speaker is "
            + "currently showing during a tech talk. List the slide's technical terms, "
            + "product names and proper nouns with their EXACT spelling as bilingual "
            + "pairs: 中文术语 = English term. IMPORTANT: ignore any subtitles/captions "
            + "overlaid on the video (scrolling text at the frame edge) — extract only "
            + "the slide content itself. The slide spelling is ground "
            + "truth for a speech-recognition pipeline that keeps misspelling them). "
            + "Comma-separated pairs only, no commentary, at most 20 pairs."
        let payload: [String: Any] = [
            "model": settings.visionModel.isEmpty ? "deepseek-v4-flash-vision-exp" : settings.visionModel,
            "messages": [[
                "role": "user",
                "content": [
                    ["type": "text", "text": prompt],
                    ["type": "image_url", "image_url": [
                        "url": "data:image/jpeg;base64," + jpeg.base64EncodedString(),
                    ]],
                ],
            ]],
            "stream": false,
            "temperature": 0.0,
            "max_tokens": 250,
            "reasoning_effort": "low",
        ]

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(settings.apiKey)", forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)

        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            return ""
        }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = object["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? String
        else { return "" }
        return content.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
