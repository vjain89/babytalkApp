import UIKit
import React
import React_RCTAppDelegate
import ReactAppDependencyProvider

@main
class AppDelegate: RCTAppDelegate {
  override func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey : Any]? = nil) -> Bool {
    self.moduleName = "babytalkApp"
    self.dependencyProvider = RCTAppDependencyProvider()
    self.initialProps = [:]

    // Cold-start: Copy-to / Open-In may pass a file URL in launchOptions.
    if let url = launchOptions?[.url] as? URL {
      IncomingAudioIngest.ingest(url)
    }
    IncomingAudioIngest.consumeAppGroupIncoming()
    IncomingAudioIngest.collectLooseAudioIntoImport()

    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  override func sourceURL(for bridge: RCTBridge) -> URL? {
    self.bundleURL()
  }

  override func bundleURL() -> URL? {
#if DEBUG
    RCTBundleURLProvider.sharedSettings().jsBundleURL(forBundleRoot: "index")
#else
    Bundle.main.url(forResource: "main", withExtension: "jsbundle")
#endif
  }

  // Share / Copy to / Open In — ingest while the security-scoped URL is valid, then notify JS.
  override func application(
    _ app: UIApplication,
    open url: URL,
    options: [UIApplication.OpenURLOptionsKey: Any] = [:]
  ) -> Bool {
    if url.isFileURL {
      IncomingAudioIngest.ingest(url)
    }
    IncomingAudioIngest.consumeAppGroupIncoming()
    IncomingAudioIngest.collectLooseAudioIntoImport()
    return RCTLinkingManager.application(app, open: url, options: options)
  }
}
