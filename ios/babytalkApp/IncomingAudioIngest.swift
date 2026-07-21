import UIKit
import UniformTypeIdentifiers

/**
 * Immediately copy an inbound Share / Open-In / Copy-to file into Documents/Import.
 * Must run while the security-scoped URL is still valid (AppDelegate openURL).
 */
enum IncomingAudioIngest {
  static let importFolderName = "Import"
  static let appGroupId = "group.org.reactjs.native.example.babytalkApp"

  @discardableResult
  static func ingest(_ url: URL) -> URL? {
    let accessed = url.startAccessingSecurityScopedResource()
    defer {
      if accessed { url.stopAccessingSecurityScopedResource() }
    }

    guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
      return nil
    }
    let importDir = docs.appendingPathComponent(importFolderName, isDirectory: true)
    do {
      try FileManager.default.createDirectory(at: importDir, withIntermediateDirectories: true)
    } catch {
      NSLog("IncomingAudioIngest: mkdir Import failed: \(error.localizedDescription)")
      return nil
    }

    var name = url.lastPathComponent
    if name.isEmpty || name == "/" {
      name = "shared_\(Int(Date().timeIntervalSince1970)).m4a"
    }
    // Voice Memos sometimes hands us extension-less temp names.
    if (name as NSString).pathExtension.isEmpty {
      name += ".m4a"
    }

    let dest = importDir.appendingPathComponent(
      "shared_\(Int(Date().timeIntervalSince1970 * 1000))_\(name)"
    )

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
      NSLog("IncomingAudioIngest: saved \(dest.lastPathComponent) (\(dest.path))")
      UserDefaults.standard.set(dest.path, forKey: "pendingSharedAudioPath")
      UserDefaults.standard.set(Date().timeIntervalSince1970, forKey: "pendingSharedAudioAt")
      return dest
    } catch {
      NSLog("IncomingAudioIngest: failed \(name): \(error.localizedDescription)")
      return nil
    }
  }

  /// Pull any audio sitting in Documents root / Inbox into Import/ for JS to pick up.
  static func collectLooseAudioIntoImport() {
    guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
      return
    }
    let importDir = docs.appendingPathComponent(importFolderName, isDirectory: true)
    try? FileManager.default.createDirectory(at: importDir, withIntermediateDirectories: true)

    let inbox = docs.appendingPathComponent("Inbox", isDirectory: true)
    let candidates = [docs, inbox]
    let audioExts: Set<String> = ["m4a", "mp4", "aac", "wav", "caf", "aiff", "aif", "mp3", "flac"]

    for dir in candidates {
      guard let items = try? FileManager.default.contentsOfDirectory(
        at: dir,
        includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
        options: [.skipsHiddenFiles]
      ) else { continue }

      for item in items {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: item.path, isDirectory: &isDir), !isDir.boolValue else {
          continue
        }
        // Don't move files already under Import/, kits, or readmes.
        if dir.path == importDir.path { continue }
        let ext = item.pathExtension.lowercased()
        let name = item.lastPathComponent
        if name == "README.txt" { continue }
        let lower = name.lowercased()
        if lower.hasPrefix("recording_") || lower.hasPrefix("imported_") { continue }
        if !audioExts.contains(ext) { continue }

        let dest = importDir.appendingPathComponent(
          "loose_\(Int(Date().timeIntervalSince1970 * 1000))_\(name)"
        )
        do {
          try FileManager.default.moveItem(at: item, to: dest)
          NSLog("IncomingAudioIngest: moved loose \(name) → Import")
        } catch {
          do {
            try FileManager.default.copyItem(at: item, to: dest)
            try? FileManager.default.removeItem(at: item)
            NSLog("IncomingAudioIngest: copied loose \(name) → Import")
          } catch {
            NSLog("IncomingAudioIngest: could not relocate \(name): \(error.localizedDescription)")
          }
        }
      }
    }
  }

  /// Copy files dropped by the Share extension (App Group → Documents/Import).
  @discardableResult
  static func consumeAppGroupIncoming() -> Int {
    guard let container = FileManager.default.containerURL(
      forSecurityApplicationGroupIdentifier: appGroupId
    ) else {
      return 0
    }
    let incoming = container.appendingPathComponent("Incoming", isDirectory: true)
    guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
      return 0
    }
    let importDir = docs.appendingPathComponent(importFolderName, isDirectory: true)
    try? FileManager.default.createDirectory(at: importDir, withIntermediateDirectories: true)
    guard let items = try? FileManager.default.contentsOfDirectory(
      at: incoming,
      includingPropertiesForKeys: nil,
      options: [.skipsHiddenFiles]
    ) else {
      return 0
    }
    var n = 0
    for item in items {
      let dest = importDir.appendingPathComponent(item.lastPathComponent)
      do {
        if FileManager.default.fileExists(atPath: dest.path) {
          try FileManager.default.removeItem(at: dest)
        }
        try FileManager.default.moveItem(at: item, to: dest)
        n += 1
      } catch {
        do {
          try FileManager.default.copyItem(at: item, to: dest)
          try? FileManager.default.removeItem(at: item)
          n += 1
        } catch {
          NSLog("IncomingAudioIngest: app group move failed \(item.lastPathComponent)")
        }
      }
    }
    return n
  }
}
