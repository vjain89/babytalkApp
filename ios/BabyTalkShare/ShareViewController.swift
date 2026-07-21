import UIKit
import UniformTypeIdentifiers

/**
 * Appears in Voice Memos (and other apps) Share sheet.
 * Copies shared audio into the App Group so the main app can import it.
 */
class ShareViewController: UIViewController {
  private let appGroupId = "group.org.reactjs.native.example.babytalkApp"

  override func viewDidAppear(_ animated: Bool) {
    super.viewDidAppear(animated)
    view.backgroundColor = .systemBackground
    Task { await handleShare() }
  }

  private func handleShare() async {
    guard let items = extensionContext?.inputItems as? [NSExtensionItem] else {
      finish()
      return
    }

    var saved = 0
    for item in items {
      guard let attachments = item.attachments else { continue }
      for provider in attachments {
        if let url = await loadFileURL(from: provider) {
          if saveToAppGroup(url) { saved += 1 }
        }
      }
    }

    // Open main app so it can import from the shared container.
    if saved > 0, let url = URL(string: "babytalk://shared-import") {
      _ = openURL(url)
    }
    finish()
  }

  private func loadFileURL(from provider: NSItemProvider) async -> URL? {
    let types = [UTType.audio.identifier, "public.mpeg-4-audio", "com.apple.m4a-audio", "public.mpeg-4", UTType.fileURL.identifier, "public.file-url", "public.url"]
    for typeId in types {
      if provider.hasItemConformingToTypeIdentifier(typeId) {
        return await withCheckedContinuation { cont in
          provider.loadItem(forTypeIdentifier: typeId, options: nil) { item, _ in
            if let url = item as? URL {
              cont.resume(returning: url)
            } else if let data = item as? Data {
              let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("share_\(UUID().uuidString).m4a")
              try? data.write(to: tmp)
              cont.resume(returning: tmp)
            } else {
              cont.resume(returning: nil)
            }
          }
        }
      }
    }
    return nil
  }

  private func saveToAppGroup(_ url: URL) -> Bool {
    guard let container = FileManager.default.containerURL(
      forSecurityApplicationGroupIdentifier: appGroupId
    ) else {
      NSLog("BabyTalkShare: missing app group container")
      return false
    }
    let incoming = container.appendingPathComponent("Incoming", isDirectory: true)
    try? FileManager.default.createDirectory(at: incoming, withIntermediateDirectories: true)

    let accessed = url.startAccessingSecurityScopedResource()
    defer { if accessed { url.stopAccessingSecurityScopedResource() } }

    var name = url.lastPathComponent
    if (name as NSString).pathExtension.isEmpty { name += ".m4a" }
    let dest = incoming.appendingPathComponent("\(Int(Date().timeIntervalSince1970 * 1000))_\(name)")
    do {
      if FileManager.default.fileExists(atPath: dest.path) {
        try FileManager.default.removeItem(at: dest)
      }
      do {
        try FileManager.default.copyItem(at: url, to: dest)
      } catch {
        let data = try Data(contentsOf: url)
        try data.write(to: dest, options: .atomic)
      }
      return true
    } catch {
      NSLog("BabyTalkShare: save failed \(error.localizedDescription)")
      return false
    }
  }

  @objc private func openURL(_ url: URL) -> Bool {
    var responder: UIResponder? = self
    while let r = responder {
      if let application = r as? UIApplication {
        application.open(url, options: [:], completionHandler: nil)
        return true
      }
      responder = r.next
    }
    // Fallback selector used by many share extensions on modern iOS.
    let selector = sel_registerName("openURL:")
    var resp: UIResponder? = self
    while let r = resp {
      if r.responds(to: selector) {
        r.perform(selector, with: url)
        return true
      }
      resp = r.next
    }
    return false
  }

  private func finish() {
    extensionContext?.completeRequest(returningItems: nil, completionHandler: nil)
  }
}
