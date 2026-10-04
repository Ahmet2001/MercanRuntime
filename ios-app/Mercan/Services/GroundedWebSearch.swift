import Foundation

struct GroundedWebSearchEvidence: Sendable {
    let result: WebSearchResult
    let pageText: String?
}

extension WebSearchService {
    /// Runs the ranked search first, then enriches the first few results with
    /// cleaned page text. Search snippets remain the fallback when a page
    /// blocks fetching, is not HTML, or cannot be decoded safely.
    func searchWithPageContent(
        query: String,
        maxResults: Int = 5,
        maxPages: Int = 3
    ) async throws -> [GroundedWebSearchEvidence] {
        let results = try await search(query: query, maxResults: maxResults)
        guard !results.isEmpty else { return [] }

        var evidence: [GroundedWebSearchEvidence] = []
        evidence.reserveCapacity(results.count)

        for (index, result) in results.enumerated() {
            let pageText: String?
            if index < maxPages {
                pageText = try? await fetchReadablePageText(urlString: result.url)
            } else {
                pageText = nil
            }
            evidence.append(GroundedWebSearchEvidence(result: result, pageText: pageText))
        }

        return evidence
    }

    /// Canonical synthetic tool call used when the user explicitly enables
    /// Web Search. The app, not the model, has already decided to search.
    static func forcedToolCallJSON(query: String, callID: String = "preflight_web_search") -> String {
        let object: [String: Any] = [
            "tool_calls": [[
                "id": callID,
                "name": "web_search",
                "arguments": ["query": query]
            ]]
        ]

        guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
              let text = String(data: data, encoding: .utf8)
        else {
            return #"{"tool_calls":[{"id":"preflight_web_search","name":"web_search","arguments":{"query":"web search"}}]}"#
        }
        return text
    }

    /// Formats enriched evidence as the raw `araç` message body. Instead of
    /// feeding long page prefixes, select short query-relevant sentences. This
    /// both improves grounding density and makes verbatim page-copy behaviour
    /// much less likely on the small local model.
    static func groundedToolResult(
        query: String,
        callID: String,
        evidence: [GroundedWebSearchEvidence]
    ) -> String {
        guard !evidence.isEmpty else {
            return "'\(query)' için arama sonucu bulunamadı."
        }

        var sections: [String] = []
        var totalCharacters = 0
        let totalLimit = 3_600

        for (index, item) in evidence.enumerated() {
            guard totalCharacters < totalLimit else { break }

            var lines: [String] = ["[\(index + 1)] \(item.result.url)"]
            if !item.result.title.isEmpty {
                lines.append("Başlık: \(item.result.title)")
            }

            let page = item.pageText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if !page.isEmpty {
                let excerpt = queryFocusedExcerpt(from: page, query: query, maxCharacters: 900)
                if !excerpt.isEmpty {
                    lines.append("İlgili kanıt:")
                    lines.append(excerpt)
                }
            } else if !item.result.snippet.isEmpty {
                lines.append("Arama özeti:")
                lines.append(String(item.result.snippet.prefix(550)))
            }

            var section = lines.joined(separator: "\n")
            let remaining = totalLimit - totalCharacters
            if section.count > remaining {
                section = String(section.prefix(remaining))
            }
            sections.append(section)
            totalCharacters += section.count
        }

        return sections.joined(separator: "\n\n")
    }

    /// Detects when a proposed answer is mostly a verbatim replay of the tool
    /// evidence. Legitimate short facts and titles are tolerated; repeated
    /// eight-word shingles covering a large part of the answer trigger a rewrite.
    static func looksLikeRawEvidenceDump(answer: String, toolBody: String) -> Bool {
        let answerWords = normalizedWords(answer)
        let evidence = normalizedWords(toolBody).joined(separator: " ")

        guard answerWords.count >= 24, !evidence.isEmpty else { return false }

        let shingleSize = 8
        let possible = answerWords.count - shingleSize + 1
        guard possible > 0 else { return false }

        var matched = 0
        var checked = 0
        let strideSize = 4
        var index = 0
        while index + shingleSize <= answerWords.count {
            let shingle = answerWords[index..<(index + shingleSize)].joined(separator: " ")
            checked += 1
            if evidence.contains(shingle) {
                matched += 1
            }
            index += strideSize
        }

        guard checked >= 4 else { return false }
        let ratio = Double(matched) / Double(checked)
        return matched >= 4 && ratio >= 0.55
    }

    private static func normalizedWords(_ text: String) -> [String] {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }

    private static func queryFocusedExcerpt(
        from page: String,
        query: String,
        maxCharacters: Int
    ) -> String {
        let queryTerms = Set(
            normalizedWords(query)
                .filter { $0.count >= 3 }
        )

        let rawSentences = page
            .replacingOccurrences(of: "\n", with: " ")
            .components(separatedBy: CharacterSet(charactersIn: ".!?;"))
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { $0.count >= 35 }

        guard !rawSentences.isEmpty else {
            return String(page.prefix(maxCharacters))
        }

        let scored: [(index: Int, score: Int, sentence: String)] = rawSentences.enumerated().map { index, sentence in
            let words = Set(normalizedWords(sentence))
            let overlap = queryTerms.intersection(words).count
            // Prefer query overlap first; keep a light early-page prior for ties.
            let earlyBonus = index < 6 ? 1 : 0
            return (index, overlap * 10 + earlyBonus, sentence)
        }

        let ranked = scored
            .sorted {
                if $0.score == $1.score { return $0.index < $1.index }
                return $0.score > $1.score
            }
            .prefix(5)
            .sorted { $0.index < $1.index }

        var selected: [String] = []
        var count = 0
        for item in ranked {
            let remaining = maxCharacters - count
            guard remaining > 0 else { break }
            let sentence = item.sentence.count > remaining
                ? String(item.sentence.prefix(remaining))
                : item.sentence
            selected.append(sentence)
            count += sentence.count + 2
        }

        let result = selected.joined(separator: ". ")
        return result.isEmpty ? String(page.prefix(maxCharacters)) : result
    }

    private func fetchReadablePageText(urlString: String) async throws -> String {
        guard let url = URL(string: urlString),
              let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https"
        else {
            throw WebSearchServiceError.invalidResponse
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 10
        request.setValue(
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1",
            forHTTPHeaderField: "User-Agent"
        )
        request.setValue("text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5", forHTTPHeaderField: "Accept")
        request.setValue("tr-TR,tr;q=0.9,en;q=0.7", forHTTPHeaderField: "Accept-Language")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode)
        else {
            throw WebSearchServiceError.invalidResponse
        }

        let contentType = (http.value(forHTTPHeaderField: "Content-Type") ?? "").lowercased()
        guard contentType.isEmpty ||
                contentType.contains("text/html") ||
                contentType.contains("application/xhtml") ||
                contentType.contains("text/plain")
        else {
            throw WebSearchServiceError.invalidResponse
        }

        // Avoid feeding an unexpectedly huge page to the parser/model.
        let cappedData = data.count > 1_500_000 ? data.prefix(1_500_000) : data[...]
        let rawData = Data(cappedData)
        let html = String(data: rawData, encoding: .utf8)
            ?? String(data: rawData, encoding: .isoLatin1)
        guard var html, !html.isEmpty else {
            throw WebSearchServiceError.invalidResponse
        }

        // Remove non-content regions before stripping markup. This deliberately
        // stays dependency-free so it can run inside the iOS app bundle.
        let removablePatterns = [
            #"<!--.*?-->"#,
            #"<script\b[^>]*>.*?</script>"#,
            #"<style\b[^>]*>.*?</style>"#,
            #"<noscript\b[^>]*>.*?</noscript>"#,
            #"<svg\b[^>]*>.*?</svg>"#,
            #"<nav\b[^>]*>.*?</nav>"#,
            #"<footer\b[^>]*>.*?</footer>"#
        ]
        for pattern in removablePatterns {
            html = html.replacingOccurrences(
                of: pattern,
                with: " ",
                options: [.regularExpression, .caseInsensitive]
            )
        }

        // Prefer article/main when available, otherwise use the cleaned body.
        let preferred = firstCapturedHTML(
            in: html,
            patterns: [
                #"<article\b[^>]*>(.*?)</article>"#,
                #"<main\b[^>]*>(.*?)</main>"#,
                #"<body\b[^>]*>(.*?)</body>"#
            ]
        ) ?? html

        var text = preferred.replacingOccurrences(
            of: #"<(?:br|p|div|li|h1|h2|h3|h4|h5|h6|tr|section|article)\b[^>]*>"#,
            with: "\n",
            options: [.regularExpression, .caseInsensitive]
        )
        text = text.replacingOccurrences(of: #"<[^>]+>"#, with: " ", options: .regularExpression)
        text = decodeBasicHTMLEntities(text)
        text = text.replacingOccurrences(of: #"[\t\r ]+"#, with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: #"\n\s*\n+"#, with: "\n", options: .regularExpression)
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard text.count >= 120 else {
            throw WebSearchServiceError.invalidResponse
        }
        return String(text.prefix(4_000))
    }

    private func firstCapturedHTML(in html: String, patterns: [String]) -> String? {
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(
                pattern: pattern,
                options: [.caseInsensitive, .dotMatchesLineSeparators]
            ) else { continue }
            let fullRange = NSRange(html.startIndex..<html.endIndex, in: html)
            guard let match = regex.firstMatch(in: html, range: fullRange),
                  match.numberOfRanges > 1,
                  let range = Range(match.range(at: 1), in: html)
            else { continue }
            let value = String(html[range])
            if !value.isEmpty { return value }
        }
        return nil
    }

    private func decodeBasicHTMLEntities(_ raw: String) -> String {
        let quote = String(UnicodeScalar(34)!)
        return raw
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&quot;", with: quote)
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&#x27;", with: "'")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&nbsp;", with: " ")
    }
}
