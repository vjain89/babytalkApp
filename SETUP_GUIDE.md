# Setup Guide - Running BabyTalk App on iPhone

## Prerequisites Check ✅
- ✅ Node.js v23.10.0 installed
- ✅ npm 11.5.1 installed
- ✅ CocoaPods 1.16.2 installed
- ✅ Xcode 16.2 installed
- ✅ iOS Pods previously installed

## Step-by-Step Setup

### 1. Install JavaScript Dependencies
```bash
npm install
```

### 2. Update iOS Dependencies (if needed)
```bash
cd ios
bundle install  # Install Ruby gems (if using Bundler)
bundle exec pod install  # Install/update CocoaPods dependencies
cd ..
```

### 3. Start Metro Bundler
In one terminal window:
```bash
npm start
```

### 4. Run on iPhone Simulator
In another terminal window:
```bash
npm run ios
```

### 5. Run on Physical iPhone Device

#### Option A: Using Xcode (Recommended)
1. Open `ios/babytalkApp.xcworkspace` in Xcode (NOT .xcodeproj)
2. Connect your iPhone via USB
3. Select your device from the device dropdown (top toolbar)
4. Click the Play button or press `Cmd+R`

**Important**: You'll need to:
- Sign the app with your Apple Developer account (free account works for personal use)
- Trust your developer certificate on your iPhone (Settings > General > VPN & Device Management)

#### Option B: Using Command Line
```bash
npm run ios -- --device "Your iPhone Name"
```

### 6. Troubleshooting

#### If Metro Bundler fails:
- Clear cache: `npm start -- --reset-cache`
- Clear watchman: `watchman watch-del-all`

#### If iOS build fails:
- Clean build folder in Xcode: `Product > Clean Build Folder` (Shift+Cmd+K)
- Delete `ios/Pods` and `ios/Podfile.lock`, then reinstall:
  ```bash
  cd ios
  rm -rf Pods Podfile.lock
  bundle exec pod install
  cd ..
  ```

#### If you get permission errors:
- Ensure microphone permission is granted on your device
- Check Info.plist has `NSMicrophoneUsageDescription`

#### If dependencies are outdated:
- React Native 0.78.1 is relatively recent, but some packages might need updates
- Check for security vulnerabilities: `npm audit`

## Quick Start Commands

```bash
# Full setup (first time)
npm install
cd ios && bundle exec pod install && cd ..
npm start  # Terminal 1
npm run ios  # Terminal 2
```

## Notes

- The app uses a local audio recorder module (`react-native-audio-recorder-player`) - ensure this is properly linked
- Database is stored locally on device (SQLite)
- Recordings are saved in the app's document directory
- Make sure your iPhone is unlocked and trusted when connecting via USB

