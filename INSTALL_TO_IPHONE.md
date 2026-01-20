# Installing BabyTalk App on iPhone via USB

## Prerequisites

1. **iPhone connected via USB** ✅ (You mentioned it's already connected)
2. **Mac with Xcode installed** ✅
3. **Apple ID** (Free account works for personal development)
4. **iPhone unlocked and trusted** (You should see "Trust This Computer?" prompt if not already trusted)

## Method 1: Using Xcode (Recommended - Easiest)

### Step 1: Open the Project in Xcode

```bash
# From the project root directory
open ios/babytalkApp.xcworkspace
```

**Important**: Always open `.xcworkspace`, NOT `.xcodeproj` (because we use CocoaPods)

### Step 2: Select Your iPhone as the Build Target

1. In Xcode's top toolbar, find the device selector (next to the Play/Stop buttons)
2. Click the device dropdown
3. You should see your iPhone listed (e.g., "Vijay's iPhone" or similar)
4. Select your iPhone from the list

If your iPhone doesn't appear:
- Make sure it's unlocked
- Make sure you've tapped "Trust" when prompted
- Try unplugging and reconnecting the USB cable
- In Xcode: `Window > Devices and Simulators` - check if your device appears there

### Step 3: Configure Code Signing

1. In Xcode, click on **"babytalkApp"** in the Project Navigator (left sidebar)
2. Select the **"babytalkApp"** target (under TARGETS)
3. Go to the **"Signing & Capabilities"** tab
4. Check **"Automatically manage signing"**
5. Select your **Team** from the dropdown (your Apple ID)
   - If you don't have a team: Click "Add Account..." and sign in with your Apple ID
   - Free Apple ID works for personal development (valid for 7 days, then re-sign)

### Step 4: Build and Install

1. Click the **Play button** (▶️) in the top-left, or press `Cmd+R`
2. Xcode will:
   - Build the app (this may take a few minutes the first time)
   - Install it on your iPhone
   - Launch the app

### Step 5: Trust Developer on iPhone (First Time Only)

When you first install the app:

1. On your iPhone, you may see: **"Untrusted Developer"**
2. Go to: **Settings > General > VPN & Device Management** (or **Device Management**)
3. Tap on your Apple ID under "Developer App"
4. Tap **"Trust [Your Name]"**
5. Confirm by tapping **"Trust"**
6. Go back to your home screen and open the BabyTalk app

### Step 6: Grant Permissions

The app will request microphone permission:
1. Tap **"Allow"** when prompted
2. This is required for audio recording

## Method 2: Using Command Line

If you prefer the command line:

### Step 1: List Connected Devices

```bash
xcrun xctrace list devices
```

Look for your iPhone in the list (it should show as "Connected").

### Step 2: Build and Install

```bash
# Get your iPhone's name/UDID from the list above, then:
npm run ios -- --device "Your iPhone Name"

# Or use UDID:
npm run ios -- --device "00008110-XXXXXXXX"  # Use actual UDID from list
```

However, you'll still need to:
- Configure signing in Xcode (Method 1, Step 3)
- Trust the developer on your iPhone (Method 1, Step 5)

## Troubleshooting

### "No devices found" / iPhone not appearing

1. **Check USB connection**: Try a different cable or USB port
2. **Unlock iPhone**: Make sure it's unlocked
3. **Trust computer**: On iPhone, tap "Trust" when prompted
4. **Check in Xcode**: `Window > Devices and Simulators` - does it appear?
5. **Restart both**: Sometimes restarting Xcode and reconnecting helps

### Code Signing Errors

**Error: "No signing certificate found"**
- Solution: Follow Step 3 above (Add your Apple ID as a team)

**Error: "Provisioning profile not found"**
- Solution: Enable "Automatically manage signing" in Xcode

**Error: "Failed to register bundle identifier"**
- Solution: Change the bundle identifier in Xcode:
  - Project Navigator > babytalkApp > Signing & Capabilities
  - Change Bundle Identifier to something unique (e.g., `com.yourname.babytalkApp`)

### "Untrusted Developer" on iPhone

- Go to: Settings > General > VPN & Device Management
- Find your Apple ID under "Developer App"
- Tap it and select "Trust"

### App Crashes Immediately

1. **Check Metro bundler**: Make sure `npm start` is running
2. **Check logs in Xcode**: View the console output
3. **Check permissions**: Make sure microphone permission was granted

### Build Fails

**"Module not found" or CocoaPods errors:**
```bash
cd ios
pod install
cd ..
```

Then rebuild in Xcode.

**"Build failed" with Swift errors:**
- Clean build folder: In Xcode, `Product > Clean Build Folder` (Shift+Cmd+K)
- Delete Derived Data: `~/Library/Developer/Xcode/DerivedData`
- Rebuild

## Quick Reference

```bash
# 1. Ensure dependencies are installed
npm install
cd ios && pod install && cd ..

# 2. Start Metro bundler (in one terminal)
npm start

# 3. Open Xcode
open ios/babytalkApp.xcworkspace

# 4. In Xcode:
#    - Select your iPhone as device
#    - Configure signing (Add your Apple ID team)
#    - Click Play (▶️) or Cmd+R

# 5. On iPhone:
#    - Trust developer (Settings > General > VPN & Device Management)
#    - Grant microphone permission
```

## Notes

- **Free Apple ID works**: You don't need a paid developer account for personal use
- **7-day certificates**: Free accounts create certificates that expire after 7 days
  - Just rebuild/reinstall to refresh (Xcode will handle it automatically)
- **Metro bundler**: Keep `npm start` running while developing
- **Hot reload**: Changes to JavaScript code will auto-reload on your iPhone
- **Native changes**: Changes to Swift/Objective-C require rebuilding in Xcode

## Success Checklist

- [ ] iPhone appears in Xcode device list
- [ ] Code signing configured (team selected)
- [ ] App builds successfully
- [ ] App installs on iPhone
- [ ] Developer trusted on iPhone
- [ ] Microphone permission granted
- [ ] App launches and runs
- [ ] Metro bundler connected (shows "Connected" in terminal)

If all checkboxes are ✅, you're good to go! 🎉

